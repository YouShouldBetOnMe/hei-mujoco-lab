"""Native reader subprocess and main-UI-thread focus/stop monitor."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


class Reader:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest = None
        self.received = 0.
        self.process = None

    def start(self):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        # mjpython sets a dylib bootstrap variable for its child. A plain reader
        # must have its OWN Foundation run loop, without another MuJoCo UI.
        env.pop('DYLD_INSERT_LIBRARIES', None)
        env.pop('MJPYTHON_BIN', None)
        executable = Path(sys.prefix)/'bin/python'
        self.process = subprocess.Popen(
            [str(executable), '-u', str(Path(__file__).with_name('native_reader.py'))],
            stdout=subprocess.PIPE, text=True, bufsize=1, env=env)
        self.thread = threading.Thread(target=self._read, name='gamepad-json', daemon=True)
        self.thread.start()
        return self

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        row = None
                except (ValueError, TypeError):
                    row = None
                with self.lock:
                    self.latest, self.received = row, time.monotonic()
        finally:
            with self.lock:
                self.latest, self.received = None, 0.

    def snapshot(self):
        with self.lock:
            return self.latest, self.received

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
            self.thread.join(timeout=2)
            self.process.stdout.close()


class WindowInput:
    def __init__(self):
        self.focused = threading.Event()
        self.lost_focus = threading.Event()
        self.exit = threading.Event()
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.closed = threading.Event()
        self.running = False
        self.error = None
        self.monitor = None
        self.observers = []

    def start(self):
        from PyObjCTools import AppHelper
        self.running = True
        AppHelper.callAfter(self._install)
        if not self.ready.wait(10) or self.error:
            self.running = False
            raise RuntimeError('Gamepad window focus monitor failed') from self.error

    def _install(self):
        import AppKit
        from Foundation import NSNotificationCenter, NSThread
        try:
            assert NSThread.isMainThread()
            if not self.running:
                return
            self.monitor = AppKit.NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
                AppKit.NSEventMaskKeyDown, self._key)
            if self.monitor is None:
                raise RuntimeError('No AppKit local monitor')
            center = NSNotificationCenter.defaultCenter()
            for name in (AppKit.NSWindowDidResignKeyNotification,
                         AppKit.NSApplicationDidResignActiveNotification):
                self.observers.append(center.addObserverForName_object_queue_usingBlock_(
                    name, None, None, self._unfocus))
            self._poll()
        except BaseException as error:
            self.error = error
        finally:
            self.ready.set()

    def _unfocus(self, _notification):
        self.focused.clear()
        self.lost_focus.set()

    def _poll(self):
        import AppKit
        from PyObjCTools import AppHelper
        if not self.running:
            return
        app = AppKit.NSApplication.sharedApplication()
        window = app.keyWindow()
        if app.isActive() and window is not None and window.isKeyWindow():
            self.focused.set()
        else:
            self.focused.clear()
        AppHelper.callLater(.05, self._poll)

    def _key(self, event):
        if event.window() is None or not event.window().isKeyWindow():
            return event
        if int(event.keyCode()) == 53:
            self.exit.set()
            return None
        if int(event.keyCode()) == 49:
            self.stop.set()
            return None
        return event

    def close(self):
        from PyObjCTools import AppHelper
        self.running = False
        self.focused.clear()
        AppHelper.callAfter(self._uninstall)
        self.closed.wait(2)

    def _uninstall(self):
        import AppKit
        from Foundation import NSNotificationCenter
        if self.monitor is not None:
            AppKit.NSEvent.removeMonitor_(self.monitor)
        for observer in self.observers:
            NSNotificationCenter.defaultCenter().removeObserver_(observer)
        self.closed.set()
