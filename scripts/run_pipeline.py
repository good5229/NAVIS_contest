#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline orchestrator: collect ECOS data then recompute BDS.

Usage:
  ECOS_API_KEY=... python3 scripts/run_pipeline.py [CONFIG_PATH]

Steps:
  1) Run ECOS collector with given config (default: data/ecos/config.json)
  2) Recompute BDS baseline from official data
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DEFAULT = PROJECT_ROOT / 'data' / 'ecos' / 'config.json'


def run(cmd: list[str]) -> int:
    print('[pipeline] Running:', ' '.join(cmd))
    proc = subprocess.run(cmd)
    return proc.returncode


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG_DEFAULT
    # Step 1: ECOS collect (optional)
    run(['python3', str(PROJECT_ROOT / 'scripts' / 'ecos_collector.py'), str(config_path)])
    # Step 1b: KOSIS indicators collect (optional)
    run(['python3', str(PROJECT_ROOT / 'scripts' / 'kosis_indicator_collector.py')])
    # Step 2: Compute BDS
    run(['python3', str(PROJECT_ROOT / 'scripts' / 'compute_bds_from_official.py')])


if __name__ == '__main__':
    main()


