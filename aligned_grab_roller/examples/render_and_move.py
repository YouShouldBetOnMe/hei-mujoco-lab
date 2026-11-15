"""Smoke test: place the table and roller, move the right arm toward the roller for 3 s, save the three camera images.
    python aligned_grab_roller/examples/render_and_move.py        -> aligned_grab_roller/examples/out/*.png
"""
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from hei_aligned import READY, AlignedGrabRoller  # noqa: E402

env = AlignedGrabRoller()
env.reset()
env.set_table(shift_left=0.01, shift_fwd=-0.02, yaw_rad=np.radians(1.0))      # the real table moves between sessions
env.place_roller(left=-0.02, fwd=0.40)                                        # 40 cm ahead of J1, 2 cm right of the midline
right_goal = np.array([0.15, -1.8, -0.75, 0.45, 0.0, 1.6])                    # roughly where the real right arm hovers
for n in range(90):
    a = min(1.0, n / 60)
    action = np.r_[np.array(READY) + a * (right_goal - READY), 0.06 * a, READY, 0.0]
    env.step(action, substeps=17)
print('state after 3 s:', env.state().round(3).tolist())
out = HERE / 'out'
out.mkdir(exist_ok=True)
for cam, img in env.render().items():
    cv2.imwrite(str(out / f'{cam}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print('saved', sorted(p.name for p in out.iterdir()))
