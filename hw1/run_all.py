"""Run the complete measurement, calibration/plotting and PDF pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def call(*arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


if __name__ == "__main__":
    call("measure.py", "--output", "results/measurements.csv", "--resume")
    call("calibrate.py")
    call("generate_derivations.py")
