"""TFDS/RLDS builder that reads UR3e VLA demonstrations from HDF5."""

from __future__ import annotations

from pathlib import Path
import hashlib
import os
import random
from typing import Any, Iterator, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class Ur3eVlaDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for UR3e VLA simulation demonstrations."""

    VERSION = tfds.core.Version("1.1.0")
    RELEASE_NOTES = {
        "1.1.0": "Validate scene/action contracts and make source keys and splits deterministic.",
        "1.0.0": "Initial HDF5 RLDS builder.",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(480, 640, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Main camera RGB observation.",
                                    ),
                                    "hand_image": tfds.features.Image(
                                        shape=(480, 640, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Wrist camera RGB observation.",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(15,),
                                        dtype=np.float32,
                                        doc="Robot state.",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(8,),
                                dtype=np.float32,
                                doc="7D policy action plus terminal flag. OpenVLA transform drops the final flag.",
                            ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Discount, defaults to 1.0.",
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Reward, 1.0 on final demo step.",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on the first step.",
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on the last step.",
                            ),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on terminal steps.",
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Natural-language task instruction.",
                            ),
                            "language_embedding": tfds.features.Tensor(
                                shape=(512,),
                                dtype=np.float32,
                                doc="Universal Sentence Encoder embedding.",
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(
                                doc="Original HDF5 file path.",
                            ),
                            "demo_id": tfds.features.Text(
                                doc="HDF5 demo group name.",
                            ),
                        }
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        h5_paths = self._resolve_h5_paths()
        demo_refs = []
        scene_profiles = set()
        for h5_path in h5_paths:
            scene_profile, demo_names = self._inspect_h5(h5_path)
            scene_profiles.add(scene_profile)
            demo_refs.extend((h5_path, name) for name in demo_names)
        if len(scene_profiles) != 1:
            raise ValueError(
                "Cannot mix scene profiles in one build: "
                + ", ".join(sorted(scene_profiles))
            )
        if not demo_refs:
            raise ValueError("No successful episodes were found in the selected HDF5 datasets.")

        seed = int(os.environ.get("UR3E_VLA_SPLIT_SEED", "42"))
        val_ratio = float(os.environ.get("UR3E_VLA_VAL_RATIO", "0.1"))
        if not 0.0 <= val_ratio < 1.0:
            raise ValueError("UR3E_VLA_VAL_RATIO must be in [0, 1).")

        rng = random.Random(seed)
        shuffled = list(demo_refs)
        rng.shuffle(shuffled)

        n_val = 0
        if val_ratio > 0.0 and len(shuffled) > 1:
            n_val = min(len(shuffled) - 1, max(1, round(len(shuffled) * val_ratio)))
        val_refs = set(shuffled[:n_val])
        train_refs = [ref for ref in demo_refs if ref not in val_refs]
        val_refs = [ref for ref in demo_refs if ref in val_refs]

        splits = {
            "train": self._generate_examples(demo_refs=train_refs),
        }
        if val_refs:
            splits["val"] = self._generate_examples(demo_refs=val_refs)
        return splits

    def _generate_examples(
        self,
        demo_refs: list[tuple[Path, str]],
    ) -> Iterator[Tuple[str, Any]]:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError(
                "h5py is required to build this dataset. Install it in rlds_env."
            ) from exc

        grouped_refs: dict[Path, list[str]] = {}
        for h5_path, demo_name in demo_refs:
            grouped_refs.setdefault(h5_path, []).append(demo_name)

        for h5_path, demo_names in grouped_refs.items():
            with h5py.File(h5_path, "r") as h5_file:
                data_group = h5_file["data"]
                source_id = self._source_id(h5_path)
                for demo_name in demo_names:
                    demo = data_group[demo_name]
                    if not bool(demo.attrs.get("success", True)):
                        continue
                    example_key = f"{source_id}/{demo_name}"
                    yield example_key, self._parse_demo(h5_path, demo_name, demo)

    def _parse_demo(self, h5_path: Path, demo_name: str, demo) -> dict:
        image = np.asarray(demo["image"], dtype=np.uint8)
        hand_image = np.asarray(demo["other"]["hand_image"], dtype=np.uint8)
        robot_state = np.asarray(demo["robot_state"], dtype=np.float32)
        action = np.asarray(demo["action"], dtype=np.float32)
        task = self._decode_task(demo["task"][()])

        if len(action) == 0:
            raise ValueError(f"Empty episode in {demo.name}")
        if not (len(image) == len(hand_image) == len(robot_state) == len(action)):
            raise ValueError(f"Inconsistent step lengths in {demo.name}")

        language_embedding = self._embed([task])[0].numpy().astype(np.float32)
        last_idx = len(action) - 1
        episode = []

        for idx in range(len(action)):
            is_last = idx == last_idx
            episode.append(
                {
                    "observation": {
                        "image": image[idx],
                        "hand_image": hand_image[idx],
                        "state": robot_state[idx],
                    },
                    "action": action[idx],
                    "discount": np.float32(1.0),
                    "reward": np.float32(float(is_last)),
                    "is_first": idx == 0,
                    "is_last": is_last,
                    "is_terminal": is_last,
                    "language_instruction": task,
                    "language_embedding": language_embedding,
                }
            )

        return {
            "steps": episode,
            "episode_metadata": {
                "file_path": str(h5_path),
                "demo_id": demo_name,
            },
        }

    def _resolve_h5_paths(self) -> list[Path]:
        env_paths = os.environ.get("UR3E_VLA_H5_PATHS")
        if env_paths:
            h5_paths = [
                Path(path.strip()).expanduser()
                for path in env_paths.split(",")
                if path.strip()
            ]
        elif os.environ.get("UR3E_VLA_H5_PATH"):
            h5_paths = [Path(os.environ["UR3E_VLA_H5_PATH"]).expanduser()]
        else:
            raise ValueError(
                "Set UR3E_VLA_H5_PATH or UR3E_VLA_H5_PATHS explicitly; "
                "there is no implicit dataset input."
            )

        if not h5_paths:
            raise ValueError("UR3E_VLA_H5_PATHS did not contain any paths.")

        missing_paths = [h5_path for h5_path in h5_paths if not h5_path.exists()]
        if missing_paths:
            missing = ", ".join(str(path) for path in missing_paths)
            raise FileNotFoundError(
                f"Could not find HDF5 dataset(s): {missing}. "
                "Set UR3E_VLA_H5_PATH=/path/to/demos.h5 or "
                "UR3E_VLA_H5_PATHS=/path/a.h5,/path/b.h5."
            )
        # Resolve aliases before de-duplicating, and sort so comma-list ordering
        # cannot change the seeded episode split.
        return sorted({path.resolve() for path in h5_paths}, key=str)

    @staticmethod
    def _source_id(h5_path: Path) -> str:
        resolved = h5_path.resolve()
        source_hash = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()
        return f"{resolved.parent.name or resolved.stem}-{source_hash}"

    @staticmethod
    def _inspect_h5(h5_path: Path) -> tuple[str, list[str]]:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError(
                "h5py is required to build this dataset. Install it in rlds_env."
            ) from exc

        with h5py.File(h5_path, "r") as h5_file:
            if "data" not in h5_file:
                raise KeyError(f"{h5_path} does not contain a /data group.")
            profile = Ur3eVlaDataset._decode_task(
                h5_file.attrs.get("scene_profile", "")
            )
            if profile not in {"canonical_scene_v1", "multiview_scene_v1"}:
                raise ValueError(f"{h5_path}: unsupported scene_profile {profile!r}")
            encoding = "logical_binary_0_open_1_closed"
            file_encoding = h5_file.attrs.get("gripper_action_encoding", encoding)
            if Ur3eVlaDataset._decode_task(file_encoding) != encoding:
                raise ValueError(
                    f"{h5_path}: unsupported gripper_action_encoding {file_encoding!r}"
                )
            demo_names = []
            for name in sorted(
                h5_file["data"].keys(), key=Ur3eVlaDataset._demo_sort_key
            ):
                demo = h5_file["data"][name]
                if not bool(demo.attrs.get("success", True)):
                    continue
                demo_profile = Ur3eVlaDataset._decode_task(
                    demo.attrs.get("scene_profile", profile)
                )
                if demo_profile != profile:
                    raise ValueError(
                        f"{h5_path}:{name}: scene_profile does not match file profile"
                    )
                demo_encoding = Ur3eVlaDataset._decode_task(
                    demo.attrs.get("gripper_action_encoding", "")
                )
                if demo_encoding != encoding:
                    raise ValueError(
                        f"{h5_path}:{name}: unsupported gripper_action_encoding {demo_encoding!r}"
                    )
                demo_names.append(name)
            return profile, demo_names

    @staticmethod
    def _list_demo_names(h5_path: Path) -> list[str]:
        return Ur3eVlaDataset._inspect_h5(h5_path)[1]

    @staticmethod
    def _demo_sort_key(name: str) -> int:
        if name.startswith("demo_"):
            return int(name.split("_", 1)[1])
        return int(name)

    @staticmethod
    def _decode_task(task_value) -> str:
        if isinstance(task_value, bytes):
            return task_value.decode("utf-8")
        if hasattr(task_value, "decode"):
            return task_value.decode("utf-8")
        return str(task_value)
