"""Grab_Roller scene aligned to the real HEI robot (measured 2026-10-29 and 2026-11-14), as a standalone runtime.

    from hei_aligned import AlignedGrabRoller
    env = AlignedGrabRoller()                 # loads model/aligned_grab_roller.xml (MuJoCo 3.5.0)
    env.reset()                               # both arms at READY [0, -0.5, -0.5, 0, 0, 0], grippers closed
    env.step(action14, substeps=17)           # one 30 Hz frame = 1/30 s / 0.002 s ~ 16.7 physics steps
    images = env.render()                     # {'front', 'left_wrist', 'right_wrist'}: 480 x 640 RGB, real lens distortion

Conventions (the same as the real robot's joint readings):
  action / state: 14 values = right J1..J6, right gripper gap (m), left J1..J6, left gripper gap (m).
  Joint angles are the real robot's readings: the zero offsets are inside the model (joint ref).
  Robot frame for placing things: `left` (m, + = robot's left of the midline) and `fwd` (m, ahead of the J1 line);
  sim world: x = MID_X + left, y = J1_Y - fwd, z up.
"""
import json
from collections import deque
from pathlib import Path

import cv2
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
INTERFACE = json.loads((HERE / 'interface.json').read_text())
J1_Y, MID_X = INTERFACE['J1_Y'], INTERFACE['MID_X']
READY = [0.0, -0.5, -0.5, 0.0, 0.0, 0.0]


def lens_map(K, dist, size):
    """For each real (distorted) pixel, where to sample the ideal pinhole render."""
    u, v = np.meshgrid(np.arange(size[0], dtype=np.float32), np.arange(size[1], dtype=np.float32))
    ideal = cv2.undistortPoints(np.stack([u.ravel(), v.ravel()], -1)[None], np.array(K), np.array(dist), P=np.array(K)).reshape(-1, 2)
    return ideal[:, 0].reshape(size[1], size[0]).astype(np.float32), ideal[:, 1].reshape(size[1], size[0]).astype(np.float32)


class AlignedGrabRoller:
    def __init__(self, model_path=HERE / 'model' / 'aligned_grab_roller.xml'):
        model_path = Path(model_path)
        load = mujoco.MjModel.from_binary_path if model_path.suffix == '.mjb' else mujoco.MjModel.from_xml_path
        self.model = load(str(model_path))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.joint_names = INTERFACE['joint_names']
        self.aids = np.array([m.actuator('servo_' + n).id for n in self.joint_names])
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in self.joint_names])
        self.dofs = np.array([m.jnt_dofadr[m.joint(n).id] for n in self.joint_names])
        self.dyn = INTERFACE['dynamics']
        self.size = tuple(INTERFACE['image_size'])
        self.lens = {cam: lens_map(c['K'], c['dist'], self.size) for cam, c in INTERFACE['cameras'].items()}
        self.kv_over_kp = -m.actuator_biasprm[self.aids, 2] / m.actuator_gainprm[self.aids, 0]
        self.renderer = None
        self.desk = m.body('env_LaptopDesk').id
        self.desk_pos0, self.desk_quat0 = m.body_pos[self.desk].copy(), m.body_quat[self.desk].copy()
        self.roller_adr = m.jnt_qposadr[m.joint('hei_object_roller_free').id]
        self.table_top_z = INTERFACE['table']['top_z']
        self.reset()

    # ---- state and control
    def state(self):
        q = self.data.qpos[self.qadr]
        return np.array([*q[:6], q[6] - q[7], *q[8:14], q[14] - q[15]], np.float32)

    def reset(self):
        """Arms at READY, grippers closed, roller where the model file has it (call place_roller after reset)."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        self.target = np.array(READY + [0.0] + READY + [0.0])
        self.applied = self.target.copy()
        ctrl = []
        for i in (0, 7):
            ctrl += list(self.target[i:i + 6]) + [0.0, 0.0]
        d.ctrl[self.aids] = ctrl
        d.qpos[self.qadr] = ctrl
        mujoco.mj_forward(m, d)
        self.history = deque(maxlen=int(round(self.dyn['delay_s'] / m.opt.timestep)) + 1)
        for _ in range(250):
            d.qfrc_applied[self.dofs] = d.qfrc_bias[self.dofs]
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        d.time = 0

    def step(self, action, substeps):
        """Joints follow v = clip(k * (cmd(t - 40 ms) - q), +-0.5 rad/s) (k 1.38 /s J1-J3, 5.6 /s J4-J6), gravity compensated;
        the gripper gap moves at most 65 mm/s opening and 70 mm/s closing. Fitted on the real 30 Hz streams."""
        m, d = self.model, self.data
        self.target[:] = np.asarray(action, float)[:14]
        k, vmax = np.array(self.dyn['k']), self.dyn['vmax']
        bounds = m.actuator_ctrlrange[self.aids]
        h = m.opt.timestep
        for _ in range(int(substeps)):
            self.history.append(self.target.copy())
            cmd = self.history[0]
            q = d.qpos[self.qadr]
            ctrl = np.zeros(16)
            for i, a in ((0, 0), (7, 8)):
                goal = np.clip(cmd[i:i + 6], bounds[a:a + 6, 0], bounds[a:a + 6, 1])
                speed = np.clip(k * (goal - q[a:a + 6]), -vmax, vmax)
                ctrl[a:a + 6] = q[a:a + 6] + speed * self.kv_over_kp[a:a + 6]
                goal_gap = np.clip(cmd[i + 6], 0, .1)
                rate = np.clip(self.dyn['gripper_k'] * (goal_gap - self.applied[i + 6]), -self.dyn['gripper_close_m_s'], self.dyn['gripper_open_m_s'])
                self.applied[i + 6] += rate * h
                ctrl[a + 6], ctrl[a + 7] = self.applied[i + 6] / 2, -self.applied[i + 6] / 2
            d.ctrl[self.aids] = np.clip(ctrl, bounds[:, 0], bounds[:, 1])
            d.qfrc_applied[self.dofs] = d.qfrc_bias[self.dofs]
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)

    # ---- cameras
    def render(self, cameras=('front', 'left_wrist', 'right_wrist')):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=self.size[1], width=self.size[0])
        out = {}
        for cam in cameras:
            self.renderer.update_scene(self.data, camera=cam)
            img = self.renderer.render().copy()
            if cam in self.lens:
                img = cv2.remap(img, *self.lens[cam], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            out[cam] = img
        return out

    # ---- scene placement
    def set_table(self, shift_left=0.0, shift_fwd=0.0, yaw_rad=0.0):
        """Move the table from its default pose (10-29: near edge 22.5 cm ahead of J1) by a shift in the robot frame and a
        turn about its centre (+ = counter-clockwise seen from above). The real table is moved between sessions:
        randomize this when generating data. Objects on it are not moved (use place_roller)."""
        m = self.model
        c = np.array([MID_X + INTERFACE['table']['centre_left_fwd'][0], J1_Y - INTERFACE['table']['centre_left_fwd'][1], 0.0])
        rot = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0], [np.sin(yaw_rad), np.cos(yaw_rad), 0], [0, 0, 1]])
        m.body_pos[self.desk] = c + rot @ (self.desk_pos0 - c) + [shift_left, -shift_fwd, 0.0]
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, np.array([np.cos(yaw_rad / 2), 0, 0, np.sin(yaw_rad / 2)]), self.desk_quat0)
        m.body_quat[self.desk] = q
        mujoco.mj_forward(m, self.data)

    def place_roller(self, left, fwd, yaw_rad=0.0):
        """Roller (40 x 2.8 cm) lying on the table, centre at (left, fwd) m in the robot frame, axis along the robot's
        left-right direction turned by yaw_rad."""
        a, d = self.roller_adr, self.data
        r = INTERFACE['roller']['radius']
        d.qpos[a:a + 3] = [MID_X + left, J1_Y - fwd, self.table_top_z + r]
        shaft = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0.0])
        R = np.stack([np.cross([0, 0, 1.0], shaft), [0, 0, 1.0], shaft], 1)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.flatten())
        d.qpos[a + 3:a + 7] = q
        d.qvel[:] = 0
        mujoco.mj_forward(self.model, d)
