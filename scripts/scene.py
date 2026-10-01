"""Offline fixed-lift HEI demo. No network, hardware, attachments or pose-follow."""
from pathlib import Path
import json

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

RUN=Path(__file__).resolve().parents[1]
ROOT=RUN.parents[1]
CAMERAS=('front','left_wrist','right_wrist')
COLORS=('red','green','blue')
PREFIX={'right':'a_right','left':'b_left'}
STATE_NAMES=sum(([*[f'{side}_joint_{j}.pos' for j in range(1,7)],f'{side}_gripper.pos'] for side in ('right','left')),[])+['x.vel','y.vel','theta.vel','height.pos']


def json_write(path,obj):
    assert path.resolve().is_relative_to(RUN)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')



class Env:
    fps=10
    def __init__(self):
        height=json.loads((RUN/'model/fixed_lift.json').read_text())['fixed_lift_m']
        self.m=mujoco.MjModel.from_xml_path(str(RUN/'model/LabRoom_HEI.xml'))
        self.height=height;self.d=mujoco.MjData(self.m);self.ikd=mujoco.MjData(self.m)
        self.joint_names=[]
        for side in ('right','left'):
            pre=PREFIX[side]
            self.joint_names += [f'{pre}_joint{i}' for i in range(1,7)]+[f'{pre}_end_joint_{f}_finger' for f in ('L','R')]
        self.aids=np.array([self.m.actuator('servo_'+n).id for n in self.joint_names])
        self.qadr=np.array([int(self.m.joint(n).qposadr[0]) for n in self.joint_names])
        self.vadr=np.array([int(self.m.joint(n).dofadr[0]) for n in self.joint_names])
        assert self.m.nu==16 and self.m.nq==48 and self.m.nv==44 and self.m.neq==0 and self.m.nmocap==0
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.m,self.d,self.m.key('home').id)
        mujoco.mj_forward(self.m,self.d)
        self.R=self.d.body('base_footprint').xmat.reshape(3,3).copy();self.origin=self.d.body('base_footprint').xpos.copy()
        self.target=self.state();self.apply(self.target)

    def state(self):
        q=self.d.qpos[self.qadr]
        return np.array([*q[:6],q[6]-q[7],*q[8:14],q[14]-q[15],0,0,0,self.height],np.float32)

    def apply(self,a):
        a=np.asarray(a,float)
        if a.shape!=(18,) or not np.isfinite(a).all():raise ValueError('18 finite absolute targets required')
        if np.any(a[14:17]!=0) or abs(a[17]-self.height)>1e-7:raise ValueError('Base and lift fixed')
        vals=[]
        for s in (0,7):
            if not -1e-7<=a[s+6]<=.1000001:raise ValueError('Total gripper opening range')
            opening=np.clip(a[s+6],0,.1)/2
            vals.extend(a[s:s+6]);vals.extend([opening,-opening])
        vals=np.asarray(vals);lim=self.m.actuator_ctrlrange[self.aids]
        if np.any(vals<lim[:,0]-1e-7) or np.any(vals>lim[:,1]+1e-7):raise ValueError('Target range')
        self.d.ctrl[self.aids]=np.clip(vals,lim[:,0],lim[:,1]);self.target=a.astype(np.float32)

    def step(self,a=None):
        if a is not None:self.apply(a)
        for _ in range(round(1/self.fps/self.m.opt.timestep)):mujoco.mj_step(self.m,self.d)
        mujoco.mj_forward(self.m,self.d)
        if not np.isfinite(self.d.qpos).all() or any(w.number for w in self.d.warning):raise RuntimeError('Physics warning/nonfinite')

    def world(self,pos):return self.origin+self.R@np.asarray(pos)
    def base(self,pos):return self.R.T@(np.asarray(pos)-self.origin)
    def cube(self,c):return self.base(self.d.body('hei_object_cube_'+c).xpos)
    def tcp(self,side):return self.base(self.d.body(PREFIX[side]+'_tcp').xpos)
    def tcp_R(self,side):return self.R.T@self.d.body(PREFIX[side]+'_tcp').xmat.reshape(3,3)

    def solve(self,side,pos,rot=None,seed=None,maxiter=350):
        """Damped full-pose IK in separate scratch data, not simulation data."""
        s=0 if side=='right' else 8;start=0 if side=='right' else 7
        qa=self.qadr[s:s+6];va=self.vadr[s:s+6]
        lo,hi=self.m.actuator_ctrlrange[self.aids[s:s+6]].T
        q=self.target[start:start+6].astype(float).copy() if seed is None else np.asarray(seed).copy()
        self.ikd.qpos[:]=self.d.qpos
        bid=self.m.body(PREFIX[side]+'_tcp').id
        goal=self.world(pos);goalR=self.R@(np.eye(3) if rot is None else rot)
        jp,jr=np.zeros((3,self.m.nv)),np.zeros((3,self.m.nv))
        best=(1e9,q.copy(),1e9,1e9)
        for i in range(maxiter):
            self.ikd.qpos[qa]=q
            mujoco.mj_kinematics(self.m,self.ikd);mujoco.mj_comPos(self.m,self.ikd)
            ep=goal-self.ikd.xpos[bid];er=Rotation.from_matrix(goalR@self.ikd.xmat[bid].reshape(3,3).T).as_rotvec()
            score=np.linalg.norm(ep)+.2*np.linalg.norm(er)
            if score<best[0]:best=(score,q.copy(),float(np.linalg.norm(ep)),float(np.linalg.norm(er)))
            if np.linalg.norm(ep)<.0003 and np.linalg.norm(er)<.002:break
            mujoco.mj_jacBody(self.m,self.ikd,jp,jr,bid)
            J=np.vstack([jp[:,va],.25*jr[:,va]]);error=np.r_[ep,.25*er]
            dq=J.T@np.linalg.solve(J@J.T+1e-5*np.eye(6),error)
            scale=min(1.,.12/max(abs(dq).max(),1e-12))
            q=np.clip(q+scale*dq,lo+1e-6,hi-1e-6)
        return best[1],dict(position_error_m=best[2],orientation_error_rad=best[3],score=best[0],iterations=i+1)
