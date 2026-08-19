"""Run the canonical three-object scene without bridge, VLA, or ROS control."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_extra.add_argument("--show-markers", action="store_true")
_extra.add_argument("--max-steps", type=int, default=0, help="0 runs until Ctrl+C")
_extra_args, _ = _extra.parse_known_args()

from vla_sim.isaac_app import boot_app, close_app, log

app = boot_app()

from vla_sim.runtime import RuntimeOptions, SimulationRuntime


def main() -> None:
    runtime = SimulationRuntime(
        RuntimeOptions(show_markers=_extra_args.show_markers)
    ).start()
    runtime.reset_targets()
    runtime.robot_controller.reset_home()
    log("Canonical scene running. Ctrl+C closes Isaac.")
    try:
        step = 0
        while app.is_running() and (_extra_args.max_steps <= 0 or step < _extra_args.max_steps):
            runtime.step()
            step += 1
    except KeyboardInterrupt:
        log("Ctrl+C received.")


if __name__ == "__main__":
    try:
        main()
    finally:
        close_app()
