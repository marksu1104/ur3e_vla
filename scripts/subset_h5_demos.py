#!/usr/bin/env python3
"""Create a smaller demonstration HDF5 file from an existing demos.h5."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def _demo_sort_key(name: str) -> int:
    if name.startswith("demo_"):
        suffix = name.removeprefix("demo_")
        if suffix.isdigit():
            return int(suffix)
    return 10**9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a fixed number of demos from one HDF5 file to another."
    )
    parser.add_argument("--input", required=True, type=Path, help="Source demos.h5")
    parser.add_argument("--output", required=True, type=Path, help="Output demos.h5")
    parser.add_argument("--episodes", required=True, type=int, help="Number of demos to copy")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="If set, randomly sample demos with this seed instead of taking the first N.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} already exists; pass --overwrite to replace it")

    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required. Install it in the active environment.") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    with h5py.File(args.input, "r") as src:
        if "data" not in src:
            raise KeyError(f"{args.input} does not contain a /data group")
        demo_names = sorted(src["data"].keys(), key=_demo_sort_key)
        if args.episodes > len(demo_names):
            raise ValueError(
                f"Requested {args.episodes} demos, but {args.input} only has {len(demo_names)}"
            )

        if args.seed is None:
            selected = demo_names[: args.episodes]
        else:
            rng = random.Random(args.seed)
            selected = sorted(rng.sample(demo_names, args.episodes), key=_demo_sort_key)

        with h5py.File(args.output, "w") as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            dst.attrs["num_demos"] = len(selected)
            dst.attrs["subset_source"] = str(args.input)
            if args.seed is not None:
                dst.attrs["subset_seed"] = args.seed

            dst_data = dst.create_group("data")
            for out_idx, demo_name in enumerate(selected):
                src.copy(src["data"][demo_name], dst_data, name=f"demo_{out_idx}")

    print(f"wrote {len(selected)} demos: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
