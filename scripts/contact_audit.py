"""Collision checks retained from the validated room integration."""
import mujoco
import numpy as np


def geom_sets(model):
    robot = np.array([i for i in range(model.ngeom) if model.geom(i).name.startswith("collision_")], dtype=int)
    environment = np.array([i for i in range(model.ngeom)
                            if model.body(model.geom_bodyid[i]).name.startswith("env_")
                            and model.geom_contype[i]], dtype=int)
    return robot, environment


def world_aabb(model, data, gid):
    """Conservative enclosure, independently certifies positive separation."""
    rotation = data.geom_xmat[gid].reshape(3, 3)
    center = data.geom_xpos[gid] + rotation @ model.geom_aabb[gid, :3]
    half = abs(rotation) @ model.geom_aabb[gid, 3:]
    return center - half, center + half


def distance_audit(model, data, cutoff=0.15):
    """Includes fixed-base vs fixed furniture: solver contacts alone omit them.

    Conservative bounding spheres discard only pairs provably beyond cutoff.
    Narrow phase uses MuJoCo's collision convex hulls, not visual triangles.
    """
    robot, environment = geom_sets(model)
    near, bad, count, zero_fallbacks = [], [], 0, 0
    for i in robot:
        lower = (np.linalg.norm(data.geom_xpos[environment] - data.geom_xpos[i], axis=1)
                 - model.geom_rbound[environment] - model.geom_rbound[i])
        for j in environment[lower <= cutoff]:
            count += 1
            dist = float(mujoco.mj_geomDistance(model, data, int(i), int(j), cutoff, None))
            engine_dist = dist
            kind = "engine_signed_distance"
            if dist == 0.0:
                # Reproduced on MuJoCo 3.5: GJK occasionally returns zero for a
                # provably separated mesh/box. Do NOT call that a real contact
                # or silently widen the tolerance. Use an independent enclosing
                # AABB certificate and label the conservative lower bound.
                al, ah = world_aabb(model, data, i)
                bl, bh = world_aabb(model, data, j)
                lower = float(np.linalg.norm(np.maximum(np.maximum(al - bh, bl - ah), 0)))
                if lower > 1e-8:
                    dist = lower; kind = "aabb_separation_lower_bound"; zero_fallbacks += 1
            if dist >= cutoff:
                continue
            row = dict(robot=model.geom(i).name, environment=model.geom(j).name, distance_m=dist,
                       engine_distance_m=engine_dist, distance_kind=kind)
            near.append(row)
            if dist < -0.001:
                bad.append(row)
    obstacles = [row for row in near if not row["environment"].startswith("RoomFloor__")]
    return dict(candidate_pairs=count, near_pairs=sorted(near, key=lambda r: r["distance_m"]),
                unexpected_penetrations=bad,
                minimum_nonfloor_distance_m=min((r["distance_m"] for r in obstacles), default=cutoff),
                distance_cutoff_m=cutoff, independently_resolved_zero_queries=zero_fallbacks,
                method="bounding spheres + signed convex collision distance; a positive enclosing-AABB certificate resolves spurious GJK zeros; includes static links")
