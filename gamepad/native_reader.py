"""macOS GameController -> newline-delimited JSON, no network or global keys.

Run under ordinary python, on its main thread, separately from mjpython's UI.
"""
import argparse
import json
import time


def main():
    import objc
    from Foundation import NSDate, NSDefaultRunLoopMode, NSRunLoop
    objc.loadBundle('GameController', globals(),
                    bundle_path='/System/Library/Frameworks/GameController.framework')
    controller_class = objc.lookUpClass('GCController')
    controller_class.setShouldMonitorBackgroundEvents_(True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=0)
    args = parser.parse_args()
    started = time.monotonic()
    previous = None
    generation = 0
    sequence = 0
    while not args.seconds or time.monotonic() - started < args.seconds:
        NSRunLoop.currentRunLoop().runMode_beforeDate_(
            NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(1 / 60))
        devices = [c for c in controller_class.controllers() if c.extendedGamepad()]
        c = previous if previous in devices else (devices[0] if devices else None)
        if c != previous:
            generation += 1
            previous = c
        sequence += 1
        row = dict(connected=c is not None, device=str(generation), seq=sequence)
        if c is not None:
            g = c.extendedGamepad()
            # GC uses +Y up, +X right and normalized values, independent of HID layout.
            row.update(name=str(c.vendorName()).strip(),
                       lx=g.leftThumbstick().xAxis().value(),
                       ly=g.leftThumbstick().yAxis().value(),
                       rx=g.rightThumbstick().xAxis().value(),
                       ry=g.rightThumbstick().yAxis().value(),
                       dx=g.dpad().xAxis().value(), dy=g.dpad().yAxis().value(),
                       lt=g.leftTrigger().value(), rt=g.rightTrigger().value(),
                       a=bool(g.buttonA().isPressed()), b=bool(g.buttonB().isPressed()),
                       x=bool(g.buttonX().isPressed()), y=bool(g.buttonY().isPressed()),
                       lb=bool(g.leftShoulder().isPressed()), rb=bool(g.rightShoulder().isPressed()),
                       menu=bool(g.buttonMenu().isPressed()))
        print(json.dumps(row, allow_nan=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except (BrokenPipeError, KeyboardInterrupt):
        pass
