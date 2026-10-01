"""Validate the published laboratory MuJoCo bundle without opening a viewer."""
from hashlib import sha256
import json
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "model" / "LabRoom_HEI.xml"


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(500):
        mujoco.mj_step(model, data)
    warnings = [int(item.number) for item in data.warning]
    assert model.nq == 48 and model.nv == 44 and model.nu == 16
    assert model.neq == 0 and model.nmocap == 0
    assert np.allclose(model.opt.gravity, [0, 0, -9.81], atol=1e-12)
    assert abs(float(model.opt.timestep) - 0.002) < 1e-12
    assert np.isfinite(data.qpos).all() and not any(warnings)

    hashes = json.loads((ROOT / "evidence" / "model_sha256.json").read_text())
    for relative, expected in hashes.items():
        actual = sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected, relative
    report = dict(
        status="passed",
        mujoco=mujoco.__version__,
        model=str(MODEL.relative_to(ROOT)),
        dimensions=dict(nq=model.nq, nv=model.nv, nu=model.nu, neq=model.neq, nmocap=model.nmocap),
        gravity=model.opt.gravity.tolist(), timestep=float(model.opt.timestep),
        steps=500, warnings=warnings, model_files_verified=len(hashes),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
