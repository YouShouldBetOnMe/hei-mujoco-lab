"""Select the current environment's viewer executable on Mac, Linux or Windows."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gamepad', action='store_true', help='macOS Xbox control')
    args, remaining = parser.parse_known_args()
    if args.gamepad and sys.platform != 'darwin':
        parser.error('Gamepad input currently uses macOS GameController')
    if sys.platform == 'darwin':
        executable = Path(sys.prefix)/'bin/mjpython'
        if not executable.is_file():
            parser.error('Install requirements.txt in this Python environment first')
    else:
        executable = Path(sys.executable)
    script = ROOT/'gamepad/run.py' if args.gamepad else ROOT/'simulate.py'
    defaults = [] if args.gamepad else ['--workbench']
    try:
        return subprocess.run([str(executable), str(script), *defaults, *remaining], cwd=ROOT).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
