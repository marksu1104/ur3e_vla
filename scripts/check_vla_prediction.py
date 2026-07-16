"""Compare VLA predictions against actions stored in a demonstration HDF5 file."""

import argparse
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vla_sim.vla_client import encode_rgb_jpeg_base64


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--server", type=str, default="http://localhost:8000")
    parser.add_argument("--demo", type=str, default="demo_0")
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 5, 10, 20, 40, 60, 80, 100])
    parser.add_argument("--unnorm-key", type=str, default="ur3e_vla_dataset")
    return parser.parse_args()


def decode_task(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "decode"):
        return value.decode("utf-8")
    return str(value)


def main():
    args = parse_args()
    with h5py.File(args.h5, "r") as h5_file:
        demo = h5_file["data"][args.demo]
        instruction = decode_task(demo["task"][()])
        print(f"instruction: {instruction}")
        print(f"demo: {args.demo}")

        for step in args.steps:
            image = np.asarray(demo["image"][step], dtype=np.uint8)
            gt_action = np.asarray(demo["action"][step], dtype=np.float32)[:7]
            payload = {
                "image_b64": encode_rgb_jpeg_base64(image),
                "instruction": instruction,
                "unnorm_key": args.unnorm_key,
            }

            start = time.monotonic()
            response = requests.post(f"{args.server.rstrip('/')}/predict", json=payload, timeout=30)
            elapsed_ms = (time.monotonic() - start) * 1000

            print(f"\nstep: {step}")
            print(f"status: {response.status_code}, latency_ms: {elapsed_ms:.1f}")
            if response.status_code != 200:
                print(response.text[:1000])
                continue

            pred_action = np.asarray(response.json()["action"], dtype=np.float32)
            abs_error = np.abs(pred_action - gt_action)
            print(f"gt:       {np.round(gt_action, 5).tolist()}")
            print(f"pred:     {np.round(pred_action, 5).tolist()}")
            print(f"abs_err:  {np.round(abs_error, 5).tolist()}")
            print(f"mean_abs: {float(abs_error.mean()):.5f}")


if __name__ == "__main__":
    main()
