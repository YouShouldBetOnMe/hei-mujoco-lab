"""Gamepad adapter for the bundled laboratory profile."""
import importlib.util
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

WORK = Path(__file__).resolve().parents[1]
LAB = WORK


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def timestep(dt):
    if not np.isfinite(dt) or dt < 0:
        raise ValueError('Invalid control timestep')
    return min(float(dt), .05)


class LabBackend:
    def __init__(self):
        self.scene = load_file('gamepad_lab_scene', LAB/'scripts/scene.py')
        self.env = self.scene.Env()
        self.m, self.d = self.env.m, self.env.d
        self.desired = self.env.state().astype(float)
        self.moving = {s: False for s in ('right', 'left')}
        self.gripping = dict(self.moving)
        self.accumulator = 0.
        self.message = ''

    def _compensated(self):
        e = self.env
        e.ikd.qpos[:] = e.d.qpos
        e.ikd.qvel[:] = 0
        for side, start, offset in (('right', 0, 0), ('left', 7, 8)):
            e.ikd.qpos[e.qadr[offset:offset+6]] = self.desired[start:start+6]
        mujoco.mj_forward(e.m, e.ikd)
        action = self.desired.copy()
        for start, offset in ((0, 0), (7, 8)):
            aids = e.aids[offset:offset+6]
            bias = e.ikd.qfrc_bias[e.vadr[offset:offset+6]]
            lo, hi = e.m.actuator_ctrlrange[aids].T
            action[start:start+6] = np.clip(
                self.desired[start:start+6] + bias/e.m.actuator_gainprm[aids, 0], lo, hi)
        return action

    def step(self, intent, dt):
        dt = timestep(dt)
        e = self.env
        self.message = ''
        for side, start, offset in (('right', 0, 0), ('left', 7, 8)):
            selected = intent.enabled and intent.mode == side+'_arm'
            moving = selected and (any(intent.linear) or any(intent.angular))
            qa = e.qadr[offset:offset+6]
            if self.moving[side] and not moving:
                # Hold the measured pose once on release; never overwrite qpos.
                self.desired[start:start+6] = e.d.qpos[qa]
            if moving:
                seed = self.desired[start:start+6].copy()
                # Compute the commanded TCP in scratch data, preventing integral
                # windup on unreachable IK or when the arm is blocked by a desk.
                e.ikd.qpos[:] = e.d.qpos
                e.ikd.qpos[qa] = seed
                mujoco.mj_forward(e.m, e.ikd)
                body = e.ikd.body(self.scene.PREFIX[side]+'_tcp')
                pos = e.base(body.xpos)
                rot = e.R.T@body.xmat.reshape(3, 3)
                if np.linalg.norm(pos-e.tcp(side)) > .025:
                    seed = e.d.qpos[qa].copy()
                    pos, rot = e.tcp(side), e.tcp_R(side).copy()
                pos = pos + np.asarray(intent.linear)*.10*intent.precision*dt
                rot = Rotation.from_rotvec(np.asarray(intent.angular)*np.deg2rad(35)
                                          *intent.precision*dt).as_matrix()@rot
                q, error = e.solve(side, pos, rot, seed=seed, maxiter=30)
                if error['position_error_m'] < .003 and error['orientation_error_rad'] < .035:
                    self.desired[start:start+6] = seed + np.clip(q-seed, -.8*dt, .8*dt)
                else:
                    self.message = 'IK LIMIT - TRY ANOTHER DIRECTION'
            gripping = selected and intent.gripper != 0
            if gripping:
                # Increment total opening at 80 mm/s; actuator forces stay original.
                self.desired[start+6] = np.clip(
                    self.desired[start+6]+intent.gripper*.08*intent.precision*dt, 0, .1)
            elif self.gripping[side]:
                # Preserve a closing preload when touching a grasped object.
                measured = e.state()[start+6]
                self.desired[start+6] = np.clip(min(self.desired[start+6], measured), 0, .1)
            self.moving[side], self.gripping[side] = moving, gripping
        e.apply(self._compensated())
        self.accumulator += dt
        while self.accumulator+1e-12 >= self.m.opt.timestep:
            mujoco.mj_step(self.m, self.d)
            self.accumulator -= self.m.opt.timestep
        mujoco.mj_forward(self.m, self.d)
        if not np.isfinite(self.d.qpos).all() or any(w.number for w in self.d.warning):
            raise RuntimeError('Laboratory physics warning/nonfinite state')

    def setup_view(self, viewer):
        import json
        self.view = load_file('gamepad_lab_view', LAB/'simulate.py')
        self.shell = self.view.AdaptiveShell(self.m, json.loads((LAB/'model/parameters.json').read_text()))
        viewer.opt.geomgroup[:] = self.view.view_options().geomgroup
        viewer.cam.lookat[:] = [-1.78, -2.51, .87]
        viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 100, -43, 2.6

    def update_view(self, viewer):
        viewport = viewer.viewport
        aspect = viewport.width/max(1, viewport.height) if viewport else 4/3
        self.shell.update(self.d, viewer.cam, viewer.opt, aspect=max(.1, aspect), mode='auto')

    def close(self):
        pass
