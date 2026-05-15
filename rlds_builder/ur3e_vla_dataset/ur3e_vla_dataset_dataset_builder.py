"""TFDS/RLDS builder that reads UR3e VLA demonstrations from HDF5."""

from __future__ import annotations

from pathlib import Path
import os
import random
from typing import Any, Iterator, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class Ur3eVlaDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for UR3e VLA simulation demonstrations."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
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
                                doc="7D policy action plus terminal flag.",
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
        h5_path = self._resolve_h5_path()
        demo_names = self._list_demo_names(h5_path)

        seed = int(os.environ.get("UR3E_VLA_SPLIT_SEED", "42"))
        val_ratio = float(os.environ.get("UR3E_VLA_VAL_RATIO", "0.1"))
        if not 0.0 <= val_ratio < 1.0:
            raise ValueError("UR3E_VLA_VAL_RATIO must be in [0, 1).")

        rng = random.Random(seed)
        shuffled = list(demo_names)
        rng.shuffle(shuffled)

        n_val = int(round(len(shuffled) * val_ratio))
        val_names = set(shuffled[:n_val])
        train_names = [name for name in demo_names if name not in val_names]
        val_names = [name for name in demo_names if name in val_names]

        splits = {
            "train": self._generate_examples(h5_path=h5_path, demo_names=train_names),
        }
        if val_names:
            splits["val"] = self._generate_examples(h5_path=h5_path, demo_names=val_names)
        return splits

    def _generate_examples(
        self,
        h5_path: Path,
        demo_names: list[str],
    ) -> Iterator[Tuple[str, Any]]:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError(
                "h5py is required to build this dataset. Install it in rlds_env."
            ) from exc

        with h5py.File(h5_path, "r") as h5_file:
            data_group = h5_file["data"]
            for demo_name in demo_names:
                demo = data_group[demo_name]
                if not bool(demo.attrs.get("success", True)):
                    continue
                yield demo_name, self._parse_demo(h5_path, demo_name, demo)

    def _parse_demo(self, h5_path: Path, demo_name: str, demo) -> dict:
        image = np.asarray(demo["image"], dtype=np.uint8)
        hand_image = np.asarray(demo["other"]["hand_image"], dtype=np.uint8)
        robot_state = np.asarray(demo["robot_state"], dtype=np.float32)
        action = np.asarray(demo["action"], dtype=np.float32)
        task = self._decode_task(demo["task"][()])

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

    def _resolve_h5_path(self) -> Path:
        env_path = os.environ.get("UR3E_VLA_H5_PATH")
        if env_path:
            h5_path = Path(env_path).expanduser()
        else:
            builder_dir = Path(__file__).resolve().parent
            h5_path = (
                builder_dir
                / ".."
                / ".."
                / "outputs"
                / "data"
                / "mug"
                / "demos.h5"
            ).resolve()

        if not h5_path.exists():
            raise FileNotFoundError(
                f"Could not find HDF5 dataset: {h5_path}. "
                "Set UR3E_VLA_H5_PATH=/path/to/demos.h5."
            )
        return h5_path

    @staticmethod
    def _list_demo_names(h5_path: Path) -> list[str]:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError(
                "h5py is required to build this dataset. Install it in rlds_env."
            ) from exc

        with h5py.File(h5_path, "r") as h5_file:
            if "data" not in h5_file:
                raise KeyError(f"{h5_path} does not contain a /data group.")
            return sorted(h5_file["data"].keys(), key=Ur3eVlaDataset._demo_sort_key)

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
