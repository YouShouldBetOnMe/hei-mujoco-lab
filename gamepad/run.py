"""Native Xbox GameController control for the bundled laboratory profile."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

from backends import LabBackend, WORK
from controls import Gate, Intent
from input_mac import Reader, WindowInput


HELP = '''Xbox: A left arm | B right arm
Hold RT to move; LT precision. Left stick forward/sideways.
Right stick up/down: arm height; left/right: yaw.
D-pad: roll/pitch. LB open / RB close gripper.
Menu pause/resume | Space stop | Esc quit.
After switching modes/focus/reconnecting: release RT and center controls.
'''


def main():
    parser = argparse.ArgumentParser(description=HELP)
    parser.add_argument('--seconds', type=float, default=0, help='Optional bounded GUI check')
    parser.add_argument('--log', type=Path, help='Optional input/robot telemetry JSONL (not a training dataset)')
    args = parser.parse_args()
    if args.seconds < 0 or not np.isfinite(args.seconds):
        parser.error('seconds must be finite and nonnegative')
    if sys.platform != 'darwin':
        parser.error('This controller reader uses macOS GameController')
    backend = LabBackend()
    reader, window = Reader(), WindowInput()
    log = None
    frames = 0
    reason = 'normal_exit'
    try:
        if args.log:
            # Logs stay with the adapters, not accepted scene/data directories.
            path = args.log.resolve()
            if not path.is_relative_to(WORK/'evidence'):
                parser.error('Logs must be under evidence/')
            path.parent.mkdir(parents=True, exist_ok=True)
            log = path.open('x')
        print(HELP, flush=True)
        with mujoco.viewer.launch_passive(backend.m, backend.d,
                                          show_left_ui=False, show_right_ui=False) as viewer:
            with viewer.lock():
                backend.setup_view(viewer)
            window.start()
            reader.start()
            gate = Gate('lab')
            started = previous = time.monotonic()
            next_render = next_log = started
            last_status = None
            while viewer.is_running() and not window.exit.is_set():
                now = time.monotonic()
                if args.seconds and now-started >= args.seconds:
                    break
                row, received = reader.snapshot()
                stop = window.stop.is_set()
                window.stop.clear()
                focused = window.focused.is_set() and not window.lost_focus.is_set()
                window.lost_focus.clear()
                intent = gate.update(row, received, now, focused, stop)
                backend.step(intent, min(now-previous, .05))
                previous = now
                frames += 1
                status = (intent.mode, intent.reason, backend.message)
                if status != last_status:
                    print('[Gamepad] lab', *status, flush=True)
                    last_status = status
                if now >= next_render:
                    with viewer.lock():
                        backend.update_view(viewer)
                    text = f'LAB | {intent.mode.upper()}\n{intent.reason}'
                    if backend.message:
                        text += '\n'+backend.message
                    viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,
                                      mujoco.mjtGridPos.mjGRID_TOPLEFT, text,
                                      'RT enable | LT fine | A/B arms\nLB/RB gripper | Menu pause | Esc quit'))
                    viewer.sync()
                    next_render = now + .05
                if log and now >= next_log:
                    log.write(json.dumps(dict(elapsed=now-started, simulation_s=backend.d.time,
                        focused=window.focused.is_set(), input=row, intent=intent.__dict__,
                        qpos=backend.d.qpos.tolist(), ctrl=backend.d.ctrl.tolist()), allow_nan=False)+'\n')
                    log.flush()
                    next_log = now+.1
                delay = .02-(time.monotonic()-now)
                if delay > 0:
                    time.sleep(delay)
    except KeyboardInterrupt:
        reason = 'keyboard_interrupt'
    finally:
        reader.close()
        window.close()
        backend.close()
        if log:
            log.close()
    print(json.dumps(dict(status=reason, frames=frames, simulation_s=backend.d.time,
                          warnings=[int(w.number) for w in backend.d.warning])), flush=True)


if __name__ == '__main__':
    main()
    # mjpython owns an AppKit loop; end only after all resources closed normally.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
