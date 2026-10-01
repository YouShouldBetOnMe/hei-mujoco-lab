"""Pure input validation and dead-man state machine (testable without hardware)."""
from dataclasses import dataclass
import math

AXES = ('lx', 'ly', 'rx', 'ry', 'dx', 'dy', 'lt', 'rt')
BUTTONS = ('a', 'b', 'x', 'y', 'lb', 'rb', 'menu')
DEADZONE = .15
STALE_S = .25
MODES = {'a': 'left_arm', 'b': 'right_arm', 'x': 'chassis', 'y': 'lift'}


def radial(x, y):
    length = math.hypot(x, y)
    if length <= DEADZONE:
        return 0., 0.
    scale = (min(1., length) - DEADZONE) / (1. - DEADZONE) / length
    return x * scale, y * scale


def unit(v):
    scale = max(1., math.sqrt(sum(x*x for x in v)))
    return tuple(x/scale for x in v)


def valid(row):
    if not isinstance(row, dict) or row.get('connected') is not True:
        return False
    if not isinstance(row.get('device'), str) or not row['device']:
        return False
    for name in AXES:
        value = row.get(name)
        if type(value) not in (float, int) or not math.isfinite(value):
            return False
        low = 0 if name in ('lt', 'rt') else -1
        if not low <= value <= 1:
            return False
    return all(type(row.get(name)) is bool for name in BUTTONS)


@dataclass(frozen=True)
class Intent:
    mode: str = 'right_arm'
    enabled: bool = False
    linear: tuple = (0., 0., 0.)
    angular: tuple = (0., 0., 0.)
    gripper: int = 0  # +1 open, -1 close
    precision: float = 1.
    reason: str = 'WAITING FOR CONTROLLER'


class Gate:
    def __init__(self, profile):
        self.allowed = {'left_arm', 'right_arm'}
        if profile == 'upstream':
            self.allowed.update(('chassis', 'lift'))
        self.mode = 'right_arm'
        self.ready = False
        self.paused = False
        self.device = None
        self.buttons = set()

    def update(self, row, received, now, focused, stop=False):
        reason = None
        if row is None or not row.get('connected', False):
            reason = 'CONTROLLER DISCONNECTED'
        elif not valid(row):
            reason = 'INVALID CONTROLLER INPUT'
        elif not 0 <= now - received <= STALE_S:
            reason = 'CONTROLLER INPUT TIMEOUT'
        elif not focused:
            reason = 'CLICK MUJOCO WINDOW TO CONTROL'
        if reason:
            self.ready = False
            self.buttons = {k for k in BUTTONS if row and row.get(k) is True}
            if stop:
                self.paused = True
            return Intent(mode=self.mode, reason=reason)
        buttons = {k for k in BUTTONS if row[k]}
        edges = buttons - self.buttons
        self.buttons = buttons
        if self.device != row['device']:
            self.device = row['device']
            self.ready = False
            edges = set()  # A held button at connection is not a new command.
        if 'menu' in edges:
            self.paused = not self.paused
            self.ready = False
        if stop:
            self.paused = True
            self.ready = False
        for key, mode in MODES.items():
            if key in edges:
                self.ready = False
                if mode not in self.allowed:
                    return Intent(mode=self.mode, reason='LAB BASE AND LIFT ARE FIXED')
                self.mode = mode
        if self.paused:
            return Intent(mode=self.mode, reason='PAUSED - MENU TO RESUME')
        neutral = (all(abs(row[k]) <= DEADZONE for k in ('lx','ly','rx','ry','dx','dy'))
                   and row['rt'] < .1 and not buttons)
        if not self.ready:
            if neutral:
                self.ready = True
            return Intent(mode=self.mode, reason='RELEASE RT AND CENTER ALL CONTROLS')
        if row['rt'] < .5:
            return Intent(mode=self.mode, reason='READY - HOLD RT TO MOVE')
        lx, ly = radial(row['lx'], row['ly'])
        rx, ry = radial(row['rx'], row['ry'])
        dx, dy = radial(row['dx'], row['dy'])
        # Robot frame: +X forward, +Y left, +Z up; yaw positive turns left.
        linear = unit((ly, -lx, ry))
        angular = unit((-dx, dy, -rx))
        if self.mode == 'chassis':
            linear = (ly, -lx, 0.)
            angular = (0., 0., -rx)
        elif self.mode == 'lift':
            linear = (0., 0., ly)
            angular = (0., 0., 0.)
        return Intent(self.mode, True, linear, angular,
                      int(row['lb'])-int(row['rb']), 1.-.75*row['lt'], 'ACTIVE')
