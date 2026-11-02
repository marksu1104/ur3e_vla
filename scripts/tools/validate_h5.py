"""Read-only validation of canonical and MultiView v1 demonstration files."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import h5py
import numpy as np


PROFILES = ("canonical_scene_v1", "multiview_scene_v1")
GRIPPER_ENCODING = "logical_binary_0_open_1_closed"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def numeric_array(demo, name: str, shape: tuple[int, ...]) -> np.ndarray:
    require(name in demo, f"{demo.name}: missing {name}")
    dataset = demo.get(name)
    require(isinstance(dataset, h5py.Dataset), f"{demo.name}/{name}: expected dataset")
    require(dataset.shape == shape, f"{dataset.name}: shape {dataset.shape}, expected {shape}")
    require(dataset.dtype == np.float32, f"{dataset.name}: expected float32")
    values = dataset[:]
    require(np.isfinite(values).all(), f"{dataset.name}: contains non-finite values")
    return values


def validate_h5(
    path: Path,
    *,
    expected_profile: str | None = None,
    expected_episodes: int | None = None,
) -> dict:
    """Validate stored schema and values; does not assess task or visual quality."""
    summaries = []
    with h5py.File(path, "r") as file:
        profile = file.attrs.get("scene_profile")
        require(profile in PROFILES, f"unsupported scene_profile: {profile!r}")
        if expected_profile is not None:
            require(profile == expected_profile, f"scene_profile {profile!r}, expected {expected_profile!r}")
        require(file.attrs.get("schema_version") == "v1", "expected schema_version=v1")
        require(file.attrs.get("format") == "vla_demo_hdf5", "expected format=vla_demo_hdf5")
        file_encoding = file.attrs.get("gripper_action_encoding")
        if file_encoding is not None:
            require(file_encoding == GRIPPER_ENCODING, "unsupported file gripper_action_encoding")
        require("data" in file and isinstance(file["data"], h5py.Group), "missing data group")
        count = len(file["data"])
        require(count > 0, "dataset has no episodes")
        require(file.attrs.get("num_demos") == count, "num_demos differs from stored episode count")
        if expected_episodes is not None:
            require(count == expected_episodes, f"stored {count} episodes, expected {expected_episodes}")

        for name, demo in file["data"].items():
            require(re.fullmatch(r"demo_\d+", name) is not None, f"unexpected episode name: {name}")
            require(isinstance(demo, h5py.Group), f"/data/{name}: expected group")
            require(demo.attrs.get("scene_profile") == profile, f"{demo.name}: scene_profile differs from file")
            success = demo.attrs.get("success")
            require(isinstance(success, (bool, np.bool_)) and bool(success), f"{demo.name}: missing/false success metadata")
            require(demo.attrs.get("record_hz") == 5.0, f"{demo.name}: expected record_hz=5.0 (metadata)")
            require(demo.attrs.get("gripper_action_encoding") == GRIPPER_ENCODING, f"{demo.name}: unsupported gripper_action_encoding")
            require("image" in demo and isinstance(demo["image"], h5py.Dataset), f"{demo.name}: missing image dataset")
            require(len(demo["image"].shape) == 4, f"{demo.name}/image: expected four dimensions")
            steps = demo["image"].shape[0]
            require(steps > 0, f"{demo.name}: empty episode")
            require(demo.attrs.get("num_steps") == steps, f"{demo.name}: num_steps differs from stored frames")
            cameras = ["image", "other/hand_image"]
            if profile == "multiview_scene_v1":
                cameras += ["other/top_image", "other/left_image", "other/right_image"]
            for camera in cameras:
                require(camera in demo, f"{demo.name}: missing {camera}")
                dataset = demo.get(camera)
                require(isinstance(dataset, h5py.Dataset), f"{demo.name}/{camera}: expected dataset")
                require(dataset.shape == (steps, 480, 640, 3), f"{dataset.name}: expected synchronized (T, 480, 640, 3) RGB")
                require(dataset.dtype == np.uint8, f"{dataset.name}: expected uint8")
                # Read one frame at a time, keeping memory bounded for large datasets.
                for index in range(steps):
                    require(np.any(dataset[index]), f"{dataset.name}: all-black frame at index {index}")

            state = numeric_array(demo, "robot_state", (steps, 15))
            action = numeric_array(demo, "action", (steps, 8))
            require(np.allclose(np.linalg.norm(state[:, 9:13], axis=1), 1, atol=1e-3), f"{demo.name}: invalid EE quaternion")
            require(np.isin(state[:, 13], [0, 1]).all(), f"{demo.name}: state gripper must be logical binary")
            require(np.all(state[:, 14] == 0), f"{demo.name}: reserved state column must be zero")
            require(np.isin(action[:, 6], [0, 1]).all(), f"{demo.name}: action gripper must be logical binary")
            require(np.all(action[:-1, 7] == 0) and action[-1, 7] == 1, f"{demo.name}: terminal flag must occur only at final step")
            require(np.all(action[-1, :6] == 0), f"{demo.name}: final motion action must be zero")
            require("task" in demo and isinstance(demo["task"], h5py.Dataset), f"{demo.name}: missing task")
            require(demo["task"].shape == (), f"{demo.name}: task must be scalar text")
            task = demo["task"].asstr()[()]
            require(bool(task.strip()), f"{demo.name}: empty task instruction")
            require(task == demo.attrs.get("instruction"), f"{demo.name}: task differs from instruction metadata")
            if profile == "multiview_scene_v1":
                for obj in ("spoon", "red_mug", "bowl"):
                    poses = numeric_array(demo, f"object_poses/{obj}", (steps, 7))
                    require(np.allclose(np.linalg.norm(poses[:, 3:7], axis=1), 1, atol=1e-3), f"{demo.name}: invalid {obj} quaternion")
                numeric_array(demo, "gripper_tip_position", (steps, 3))
            summaries.append({"episode": name, "steps": steps, "cameras": len(cameras)})
    return {"path": str(path), "scene_profile": profile, "episodes": count, "steps": sum(item["steps"] for item in summaries), "details": summaries}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5", type=Path)
    parser.add_argument("--expected-profile", choices=PROFILES)
    parser.add_argument("--expected-episodes", type=int)
    args = parser.parse_args()
    if args.expected_episodes is not None and args.expected_episodes < 1:
        parser.error("--expected-episodes must be at least 1")
    try:
        result = validate_h5(args.h5, expected_profile=args.expected_profile, expected_episodes=args.expected_episodes)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"FAIL: {args.h5}: {error}", file=sys.stderr)
        return 1
    print(f"PASS: {result['scene_profile']}; episodes={result['episodes']}, steps={result['steps']}; {args.h5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
