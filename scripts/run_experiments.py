"""List paper experiments, or execute a selected group sequentially."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('figure2', 'figure3', 'section5', 'section5_main', 'section5_ablations')


def select_experiments(manifest, group):
    """Select a paper experiment group independently of its calibration source."""
    selected = ('section5_main', 'section5_ablations') if group == 'section5' else (group,)
    return [entry for part in selected for entry in manifest if entry['group'] == part]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=GROUPS, default='section5_main',
                        help='Paper experiment group; section5 includes main results and ablations.')
    parser.add_argument('--run', action='store_true', help='Execute; by default only print commands.')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'configs/manifest.json').read_text())
    for entry in select_experiments(manifest, args.group):
        command = [sys.executable, str(ROOT / 'main.py'), '--config-path', entry['config']]
        print(shlex.join(command), flush=True)
        if args.run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
