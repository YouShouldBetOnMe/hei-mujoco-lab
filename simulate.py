#!/usr/bin/env python3
"""Retained laboratory MuJoCo viewer, with adaptive architecture (no network).

Room geometry is retained apart from the removed laptop. The versioned MJCF
contains pinned upstream contact parameters and four dynamic tabletop props.
Robot cameras always render every architectural group.
"""
import argparse
import json
from pathlib import Path
import struct
import time
import zlib

import mujoco
import numpy as np

BUNDLE = Path(__file__).resolve().parent
SENSOR_CAMERAS = ("front", "left_wrist", "right_wrist")
ATTACHMENTS = {
    "VideoWall": "WallEdge0",
    "BlockedRedDoor": "WallEdge1",
    "BenchWallBrackets": "WallEdge1",
    "EntryShelves": "WallEdge3",
    "EastShelves": "WallEdge3",
    "WindowWallColumn": "WallEdge6",
    "Window0_Radiator": "WallEdge6",
    "Window1_Radiator": "WallEdge6",
    "Window2_Radiator": "WallEdge6",
    "CeilingLights": "RoomCeiling",
}


def inside_polygon(xy, polygon):
    """Even-odd point test, also valid for the room's concave southeast notch."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    result = np.zeros(len(xy), dtype=bool)
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        if abs(b[1] - a[1]) < 1e-12:
            continue
        cross = ((a[1] > xy[:, 1]) != (b[1] > xy[:, 1]))
        x = a[0] + (xy[:, 1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
        result ^= cross & (xy[:, 0] < x)
    return result


def polygon_clearance(xy, polygon):
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    distances = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        ab = b - a
        t = np.clip((xy - a) @ ab / (ab @ ab), 0, 1)
        distances.append(np.linalg.norm(xy - a - t[:, None] * ab, axis=1))
    return np.min(distances, axis=0)


def segments_hit_box(eye, points, low, high):
    """Finite eye->interior segments vs a slab; parallel axes handled explicitly."""
    if not len(points):
        return False
    delta = points - eye
    parallel = np.abs(delta) < 1e-10
    outside = np.any(parallel & ((eye < low) | (eye > high)), axis=1)
    inv = np.divide(1., delta, out=np.zeros_like(delta), where=~parallel)
    a, b = (low - eye) * inv, (high - eye) * inv
    enter = np.max(np.where(parallel, -np.inf, np.minimum(a, b)), axis=1)
    leave = np.min(np.where(parallel, np.inf, np.maximum(a, b)), axis=1)
    return bool(np.any(~outside & (leave >= np.maximum(enter, 0)) & (enter < .999) & (leave >= 0)))


def view_options(full_room=False):
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [0, 1, 1, 1, int(full_room), 1]
    return opt


def sensor_camera(model, camera):
    return (camera.type == mujoco.mjtCamera.mjCAMERA_FIXED
            and 0 <= camera.fixedcamid < model.ncam
            and model.camera(camera.fixedcamid).name in SENSOR_CAMERAS)


def free_camera(azimuth=-55, distance=13, elevation=-28, lookat=(-.1, -3.0, .8)):
    camera = mujoco.MjvCamera()
    camera.lookat[:] = lookat
    camera.azimuth = azimuth
    camera.distance = distance
    camera.elevation = elevation
    return camera


class AdaptiveShell:
    """Select whole wall/roof assemblies that obstruct visible room samples.

    Finite wall slabs, the actual concave floor polygon, camera FOV and viewport
    aspect are used, not simply front/back-facing infinite wall planes. Windows,
    doors and wall-mounted fixtures follow their host wall. A 3 cm spatial and
    small frustum hysteresis avoids rapid switching at a visibility boundary.
    Hidden group4 is exclusively a presentation group; no alpha/physics edits.
    """
    def __init__(self, model, spec):
        self.model = model
        self.polygon = np.asarray(spec["floor_polygon"], dtype=float)
        self.height = float(spec["height_m"])
        self.wall_names = tuple(f"WallEdge{i}" for i in range(len(self.polygon)))
        self.names = (*self.wall_names, "RoomCeiling")
        self.ids = {name: [] for name in self.names}
        for gid in range(model.ngeom):
            body = model.body(model.geom_bodyid[gid]).name
            if not body.startswith("env_"):
                continue
            asset = body[4:]
            group = ATTACHMENTS.get(asset, asset)
            model.geom_group[gid] = 3
            if group in self.ids:
                self.ids[group].append(gid)
        self.ids = {n: np.array(ids, dtype=int) for n, ids in self.ids.items()}
        if any(not len(ids) for ids in self.ids.values()):
            raise ValueError("Missing architectural group; refusing an incomplete room")
        area2 = np.sum(self.polygon[:, 0] * np.roll(self.polygon[:, 1], -1)
                       - self.polygon[:, 1] * np.roll(self.polygon[:, 0], -1))
        self.walls = []
        for a, b in zip(self.polygon, np.roll(self.polygon, -1, axis=0)):
            tangent = (b - a) / np.linalg.norm(b - a)
            outward = np.sign(area2) * np.array([tangent[1], -tangent[0]])
            rotation = np.array([[*tangent, 0], [*outward, 0], [0, 0, 1]]).T
            self.walls.append((np.r_[a, 0.], rotation, np.linalg.norm(b - a)))
        # Sample the interior volume, not just one look-at ray: foreground walls
        # can cover furniture near the edge of the viewport without hitting it.
        x = np.arange(self.polygon[:, 0].min(), self.polygon[:, 0].max(), .40)
        y = np.arange(self.polygon[:, 1].min(), self.polygon[:, 1].max(), .40)
        xy = np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)
        xy = xy[inside_polygon(xy, self.polygon) & (polygon_clearance(xy, self.polygon) > .18)]
        self.samples = np.concatenate([np.c_[xy, np.full(len(xy), z)]
                                       for z in (.15, .75, 1.35, 2.0, self.height - .18)])
        self.camera_scene = mujoco.MjvScene(model, maxgeom=1)
        self.hidden = set()
        self.last_signature = None
        self.evaluations = 0
        self.eye = np.zeros(3)

    def camera_frame(self, data, camera):
        mujoco.mjv_updateCamera(self.model, data, camera, self.camera_scene)
        eye, forward, up = [np.zeros(3) for _ in range(3)]
        mujoco.mjv_cameraInModel(eye, forward, up, self.camera_scene)
        gl = mujoco.mjv_averageCamera(self.camera_scene.camera[0], self.camera_scene.camera[1])
        return eye, forward, up, gl

    def select(self, eye, forward, up, gl, aspect, lookat):
        points = self.samples
        focus = np.asarray(lookat, dtype=float)
        if (inside_polygon(focus[:2], self.polygon)[0]
                and polygon_clearance(focus[:2], self.polygon)[0] > .10
                and .05 < focus[2] < self.height - .08):
            points = np.vstack([points, focus])
        delta = points - eye
        depth = delta @ forward
        right = np.cross(forward, up)
        # MuJoCo fills horizontal extent at render time using viewport aspect.
        vhalf = (gl.frustum_top - gl.frustum_bottom) / 2
        hhalf = (gl.frustum_width / 2) if gl.frustum_width > 0 else vhalf * aspect
        denominator = np.ones(len(points)) if gl.orthographic else depth / gl.frustum_near
        good = depth > max(gl.frustum_near * .1, .001)
        denominator = np.where(good, denominator, 1.)
        px, py = (delta @ right) / denominator, (delta @ up) / denominator
        vcenter = (gl.frustum_top + gl.frustum_bottom) / 2
        def visible(margin):
            return points[good & (np.abs(px - gl.frustum_center) <= hhalf * margin)
                          & (np.abs(py - vcenter) <= vhalf * margin)]
        normal, expanded = visible(1.015), visible(1.05)
        hidden = set()
        for name, (origin, rotation, length) in zip(self.wall_names, self.walls):
            pad = .04 if name in self.hidden else .01
            targets = expanded if name in self.hidden else normal
            if segments_hit_box((eye - origin) @ rotation, (targets - origin) @ rotation,
                                np.array([-pad, -pad, -pad]),
                                np.array([length + pad, .12 + pad, self.height + pad])):
                hidden.add(name)
        # The roof is a finite polygon, not an infinite Z plane. Eye in its
        # thickness must also work; indoor upward views retain the ceiling.
        pad = .04 if "RoomCeiling" in self.hidden else .01
        targets = expanded if "RoomCeiling" in self.hidden else normal
        if len(targets) and eye[2] >= self.height - pad:
            z = min(eye[2], self.height + .06)
            dz = targets[:, 2] - eye[2]
            t = np.divide(z - eye[2], dz, out=np.full(len(targets), -1.), where=np.abs(dz) > 1e-10)
            hits = eye + t[:, None] * (targets - eye)
            if np.any((t >= 0) & (t < .999) & inside_polygon(hits[:, :2], self.polygon)):
                hidden.add("RoomCeiling")
        return hidden

    def update(self, data, camera, option, aspect=1.5, mode="auto"):
        if mode not in ("auto", "full", "cutaway") or not np.isfinite(aspect) or aspect <= 0:
            raise ValueError("Invalid view mode or viewport aspect")
        eye, forward, up, gl = self.camera_frame(data, camera)
        full = mode == "full" or sensor_camera(self.model, camera)
        signature = (mode, full, int(camera.type), int(camera.fixedcamid), aspect,
                     *eye, *forward, *up, *camera.lookat, gl.frustum_top, gl.frustum_bottom,
                     gl.frustum_width, gl.frustum_near, gl.frustum_center, int(gl.orthographic))
        # Also enforce full visibility on every sensor-camera update, including
        # after a manual viewer group toggle. Group0 collision duplicates stay off.
        option.geomgroup[3] = 1
        option.geomgroup[4] = int(full)
        self.eye = eye
        if signature == self.last_signature:
            return self.hidden.copy()
        hidden = set() if full else set(self.names) if mode == "cutaway" else self.select(
            eye, forward, up, gl, aspect, camera.lookat)
        for name, ids in self.ids.items():
            self.model.geom_group[ids] = 4 if name in hidden else 3
        self.hidden = hidden
        self.last_signature = signature
        self.evaluations += 1
        return hidden.copy()

    def report(self):
        return dict(eye_m=self.eye.tolist(), hidden=sorted(self.hidden),
                    visible=sorted(set(self.names) - self.hidden), evaluations=self.evaluations)


def png(path, pixels):
    pixels = np.asarray(pixels, np.uint8)
    h, w, _ = pixels.shape
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    rows = b"".join(b"\0" + row.tobytes() for row in pixels[:, :, :3])
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def load():
    model = mujoco.MjModel.from_xml_path(str(BUNDLE / "model/LabRoom_HEI.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    shell = AdaptiveShell(model, json.loads((BUNDLE / "model/parameters.json").read_text()))
    return model, data, shell


def render(model, data, shell, mode, folder):
    folder = folder.resolve()
    if not folder.is_relative_to(BUNDLE):
        raise ValueError("Render output must stay within this bundle")
    folder.mkdir(parents=True, exist_ok=True)
    cases = [(f"orbit_{i:02d}", free_camera(azimuth=-180 + i * 10), mode) for i in range(37)]
    cases += [("indoor", free_camera(azimuth=-55, distance=1.2, elevation=-8, lookat=(-1.75, -2.25, 1.0)), mode),
              ("above", free_camera(azimuth=90, elevation=-89, distance=16), mode),
              ("full_room_comparison", free_camera(azimuth=90), "full")]
    cases += [("tabletop_detail", free_camera(azimuth=100, distance=3.0, elevation=-43,
                                             lookat=(-1.78, -2.51, .86)), mode),
              ("tabletop_overhead", free_camera(azimuth=90, distance=2.7, elevation=-89,
                                               lookat=(-1.78, -2.51, .82)), mode)]
    for name in SENSOR_CAMERAS:
        c = mujoco.MjvCamera(); c.type = mujoco.mjtCamera.mjCAMERA_FIXED
        c.fixedcamid = model.camera(name).id
        cases.append((name, c, mode))
    if any((folder / (name + ".png")).exists() for name, _, _ in cases):
        raise FileExistsError("Preserve existing renders; choose a fresh output subdirectory")
    report = {}
    with mujoco.Renderer(model, width=1000, height=750) as renderer:
        opt = view_options()
        for name, camera, camera_mode in cases:
            shell.update(data, camera, opt, aspect=4/3, mode=camera_mode)
            renderer.update_scene(data, camera=camera, scene_option=opt)
            pixels = renderer.render()
            png(folder / (name + ".png"), pixels)
            report[name] = shell.report()
            report[name]["pixel_standard_deviation"] = float(pixels.std())
    (folder / "visibility.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(rendered=len(cases), output=str(folder), mode=mode)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--render-dir", type=Path, default=BUNDLE / "previews/native")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--full-room", action="store_true", help="Keep all walls/roof visible")
    mode.add_argument("--cutaway", action="store_true", help="Legacy all-shell-hidden observer mode")
    ap.add_argument("--duration", type=float, default=0)
    ap.add_argument("--demo-orbit", action="store_true", help="Bounded 16-second camera orbit / sensor switch demo")
    ap.add_argument("--view-log", type=Path, help="Save GUI camera visibility transitions inside this bundle")
    ap.add_argument("--workbench", action="store_true", help="Start close to the new tabletop props")
    args = ap.parse_args()
    if args.steps < 0 or not np.isfinite(args.duration) or args.duration < 0:
        ap.error("steps and duration must be finite and nonnegative")
    if args.view_log and (not args.view_log.resolve().is_relative_to(BUNDLE) or args.view_log.exists()):
        ap.error("view-log must be a new file inside this bundle")
    selected_mode = "full" if args.full_room else "cutaway" if args.cutaway else "auto"
    model, data, shell = load()
    if args.render:
        render(model, data, shell, selected_mode, args.render_dir)
        return
    if args.headless:
        opt = view_options()
        camera = free_camera()
        for i in range(args.steps):
            if i % 20 == 0:
                camera.azimuth = -180 + 360 * i / max(1, args.steps)
                shell.update(data, camera, opt, mode=selected_mode)
            mujoco.mj_step(model, data)
        warnings = [int(w.number) for w in data.warning]
        if any(warnings) or not np.isfinite(data.qpos).all():
            raise RuntimeError(warnings)
        print(json.dumps(dict(status="passed", mujoco=mujoco.__version__, steps=args.steps,
                              warnings=warnings, camera=shell.report(), network=False, real_robot=False)))
        return
    from mujoco import viewer as viewer_module
    transitions = []
    with viewer_module.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.opt.geomgroup[:] = view_options().geomgroup
            viewer.cam.lookat[:] = [-.1, -3.0, .8]
            viewer.cam.distance = 13; viewer.cam.azimuth = -55; viewer.cam.elevation = -28
            if args.workbench and not args.demo_orbit:
                viewer.cam.lookat[:] = [-1.78, -2.51, .86]
                viewer.cam.distance = 3.0; viewer.cam.azimuth = 100; viewer.cam.elevation = -43
        started = time.monotonic()
        previous = None
        while viewer.is_running():
            now = time.monotonic()
            elapsed = now - started
            limit = args.duration or (16 if args.demo_orbit else 0)
            if limit and elapsed >= limit:
                break
            mujoco.mj_step(model, data)
            if any(int(w.number) for w in data.warning) or not np.isfinite(data.qpos).all():
                raise RuntimeError("Stopped viewer after physics warning")
            with viewer.lock():
                if args.demo_orbit:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                    viewer.cam.distance = 13
                    viewer.cam.azimuth = -180 + min(elapsed, 12) * 30
                    if 12 <= elapsed < 14:
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                        viewer.cam.fixedcamid = model.camera("front").id
                viewport = viewer.viewport
                aspect = viewport.width / max(1, viewport.height) if viewport is not None else 1.5
                hidden = shell.update(data, viewer.cam, viewer.opt, max(aspect, .1), selected_mode)
                state = (int(viewer.cam.type), int(viewer.cam.fixedcamid), tuple(sorted(hidden)))
                if state != previous:
                    transitions.append(dict(seconds=round(elapsed, 3), camera_type=state[0],
                                            fixed_camera_id=state[1], **shell.report()))
                    previous = state
            # Full sync propagates changed rendering groups. state_only would not.
            viewer.sync()
            time.sleep(max(0, model.opt.timestep - (time.monotonic() - now)))
    if args.view_log:
        args.view_log.write_text(json.dumps(transitions, indent=2) + "\n")
    print(json.dumps(dict(viewer_closed=True, mode=selected_mode, visibility_transitions=len(transitions),
                          warnings=[int(w.number) for w in data.warning])))


if __name__ == "__main__":
    main()
    # mjpython's AppKit loop can outlive a closed viewer on macOS.
    import os
    import sys
    if sys.platform == "darwin":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
