"""Apply the patch to pinned OpenVLA and exercise its actual loop on CPU.

The source checkout is read with ``git show``; its working tree is never edited.
Set UR3E_OPENVLA_SOURCE to a local OpenVLA repository when using another layout.
No model downloads, CUDA execution or OpenVLA imports are needed.
"""

from __future__ import annotations

import ast
from collections import deque
from contextlib import nullcontext, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock


PROJECT = Path(__file__).resolve().parents[1]
REVISION = "c8f03f48af692657d3060c19588038c7220e9af9"
FILES = (
    "prismatic/vla/datasets/rlds/oxe/configs.py",
    "prismatic/vla/datasets/rlds/oxe/mixtures.py",
    "prismatic/vla/datasets/rlds/oxe/transforms.py",
    "vla-scripts/finetune.py",
)


def _function(source):
    return next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "finetune"
    )


def _compile(nodes):
    module = ast.Module(body=nodes, type_ignores=[])
    return compile(ast.fix_missing_locations(module), "<patched-openvla>", "exec")


class OpenVlaPatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        candidates = [
            Path(os.environ["UR3E_OPENVLA_SOURCE"])
            if os.environ.get("UR3E_OPENVLA_SOURCE") else None,
            PROJECT / "outputs/test/model/openvla_clean_20261004",
            PROJECT / "outputs/test/openvla_clean_20261004",
            PROJECT.parents[1] / "vla_ws/openvla/openvla",
        ]
        checkout = next((path for path in candidates if path and (path / ".git").exists()), None)
        if checkout is None:
            raise unittest.SkipTest("Set UR3E_OPENVLA_SOURCE to a local pinned OpenVLA checkout")
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.root = Path(directory.name)
        for name in FILES:
            source = subprocess.run(
                ["git", "-C", str(checkout), "show", f"{REVISION}:{name}"],
                check=True, capture_output=True,
            ).stdout
            destination = cls.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source)
        subprocess.run(["git", "init", "--quiet", str(cls.root)], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(cls.root), "apply", "--check", str(PROJECT / "scripts/tools/openvla_ur3e.patch")],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(cls.root), "apply", str(PROJECT / "scripts/tools/openvla_ur3e.patch")],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(cls.root), "diff", "--check"],
            check=True, capture_output=True,
        )
        cls.source = (cls.root / "vla-scripts/finetune.py").read_text()

    def run_loop(
        self,
        *,
        max_steps,
        save_steps,
        accumulation=1,
        keep_latest=True,
        use_lora=True,
        export_final=True,
    ):
        function = _function(self.source)
        start = next(
            index for index, node in enumerate(function.body)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "recent_losses" for target in node.targets)
        )
        actual_loop = _compile(function.body[start:])
        counts = {"updates": 0, "backwards": 0, "progress": 0}
        saves = []
        logs = []
        tensor = MagicMock()
        for method in ("to", "__getitem__", "argmax", "__gt__", "__eq__", "__and__", "sum", "float", "__truediv__", "cpu"):
            getattr(tensor, method).return_value = tensor
        tensor.item.return_value = 1.0
        tensor.numpy.return_value = [0., 0., 0., 0., 0., 0., 1.]
        tensor.device = "cpu"

        def increment(name):
            counts[name] += 1

        tensor.backward.side_effect = lambda: increment("backwards")

        def saver(kind, version=None):
            def save(path):
                path = Path(path)
                path.mkdir(parents=True, exist_ok=True)
                update = counts["updates"] if version is None else version
                (path / f"{kind}.json").write_text(json.dumps({"updates": update}))
                saves.append((kind, path.name, update))
            return save

        class Progress:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def update(self):
                increment("progress")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, model_dir, adapter_dir = root / "runs/run", root / "final", root / "adapters"
            run_dir.mkdir(parents=True)
            module = SimpleNamespace(
                vision_backbone=SimpleNamespace(featurizer=SimpleNamespace(patch_embed=SimpleNamespace(num_patches=0))),
                save_pretrained=saver("adapter" if use_lora else "weights"),
            )
            vla = MagicMock(return_value=SimpleNamespace(loss=tensor, logits=tensor))
            vla.module = module

            def merge(base, path):
                update = json.loads((Path(path) / "adapter.json").read_text())["updates"]
                return SimpleNamespace(merge_and_unload=lambda: SimpleNamespace(save_pretrained=saver("weights", update)))

            cfg = SimpleNamespace(
                max_steps=max_steps, save_steps=save_steps, grad_accumulation_steps=accumulation,
                use_lora=use_lora, save_latest_checkpoint_only=keep_latest, vla_path="offline/mock",
            )
            batch = {key: tensor for key in ("input_ids", "attention_mask", "pixel_values", "labels")}
            def save_statistics(stats, path):
                del stats
                saver("dataset_statistics")(path)

            def publish(model, processor, stats, path):
                path = Path(path)
                path.mkdir(parents=True, exist_ok=False)
                save_statistics(stats, path)
                processor.save_pretrained(path)
                model.save_pretrained(path)

            namespace = dict(
                cfg=cfg, vla=vla, deque=deque, CausalLMOutputWithPast=object,
                optimizer=SimpleNamespace(step=lambda: increment("updates"), zero_grad=lambda: None),
                dataloader=[batch] * ((max_steps + 2) * accumulation),
                torch=SimpleNamespace(
                    autocast=lambda *args, **kwargs: nullcontext(), bfloat16="bf16", tensor=lambda value: tensor,
                    nn=SimpleNamespace(functional=SimpleNamespace(l1_loss=lambda *args: tensor)),
                ),
                tqdm=SimpleNamespace(tqdm=lambda *args, **kwargs: Progress()),
                action_tokenizer=SimpleNamespace(action_token_begin_idx=256, decode_token_ids_to_actions=lambda value: value),
                device_id="cpu", distributed_state=SimpleNamespace(is_main_process=True),
                wandb=SimpleNamespace(log=lambda data, step: logs.append(step)),
                dist=SimpleNamespace(barrier=lambda: None),
                processor=SimpleNamespace(save_pretrained=saver("processor")),
                final_model_dir=model_dir if export_final else None,
                adapter_dir=adapter_dir,
                run_dir=run_dir,
                Path=Path, os=os,
                AutoModelForVision2Seq=SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()),
                PeftModel=SimpleNamespace(from_pretrained=merge),
                _require_complete_model_dir=lambda path: self.fail(
                    f"unexpected completeness check without publication: {path}"
                ),
                _publish_final_model=publish,
                save_dataset_statistics=save_statistics,
                vla_dataset=SimpleNamespace(dataset_statistics={}),
            )
            with redirect_stdout(io.StringIO()):
                exec(actual_loop, namespace)
            final = model_dir / "weights.json"
            if export_final:
                self.assertTrue(
                    final.exists(), "final fused weights were not exported"
                )
                self.assertEqual(
                    json.loads(final.read_text())["updates"], max_steps
                )
            else:
                self.assertFalse(model_dir.exists())
        self.assertEqual(counts, {"updates": max_steps, "backwards": max_steps * accumulation, "progress": max_steps})
        return saves, logs

    def test_one_update_is_exported_before_first_periodic_checkpoint(self):
        saves, _ = self.run_loop(max_steps=1, save_steps=250)
        self.assertEqual([update for kind, _, update in saves if kind == "weights"], [1])

    def test_final_export_includes_last_update_between_checkpoint_intervals(self):
        saves, _ = self.run_loop(max_steps=3, save_steps=2)
        self.assertEqual([update for kind, _, update in saves if kind == "weights"], [2, 3])

    def test_accumulation_finishes_exactly_before_checkpoint_and_stop(self):
        for accumulation in (2, 3):
            with self.subTest(accumulation=accumulation):
                saves, _ = self.run_loop(max_steps=4, save_steps=2, accumulation=accumulation)
                self.assertEqual([update for kind, _, update in saves if kind == "weights"], [2, 4])

    def test_retained_checkpoints_also_export_final_model(self):
        saves, _ = self.run_loop(max_steps=3, save_steps=2, accumulation=2, keep_latest=False)
        self.assertEqual(
            [(path, update) for kind, path, update in saves if kind == "weights"],
            [("run--2_chkpt", 2), ("run--3_chkpt", 3), ("final", 3)],
        )

    def test_full_finetuning_also_exports_last_update(self):
        saves, _ = self.run_loop(max_steps=3, save_steps=2, use_lora=False)
        self.assertEqual(
            [(path, update) for kind, path, update in saves if kind == "weights"],
            [("run", 2), ("run", 3), ("final", 3)],
        )

    def test_omitting_final_model_dir_preserves_run_directory_export(self):
        saves, _ = self.run_loop(
            max_steps=1,
            save_steps=250,
            export_final=False,
        )
        self.assertEqual(
            [(path, update) for kind, path, update in saves if kind == "weights"],
            [("run", 1)],
        )

    def test_metrics_log_once_per_completed_tenth_update(self):
        _, logs = self.run_loop(max_steps=11, save_steps=250, accumulation=3)
        self.assertEqual(logs, [10])

    def test_nonpositive_step_parameters_fail_before_gpu_setup(self):
        function = _function(self.source)
        gpu_check = next(index for index, node in enumerate(function.body) if isinstance(node, ast.Assert))
        validation = _compile(function.body[:gpu_check])
        for name in ("max_steps", "save_steps", "grad_accumulation_steps"):
            for value in (0, -1):
                cfg = SimpleNamespace(
                    max_steps=1, save_steps=250, grad_accumulation_steps=1,
                    vla_path="offline/mock", dataset_name="ur3e_vla_dataset",
                    final_model_dir=None,
                )
                setattr(cfg, name, value)
                with self.subTest(name=name, value=value), redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "must be positive"):
                    exec(validation, {"cfg": cfg})

    def export_helpers(self, statistics_writer=None):
        names = {
            "_is_nonempty_file",
            "_require_complete_model_dir",
            "_publish_final_model",
        }
        constants = {
            "_REQUIRED_FINAL_MODEL_FILES",
            "_TOKENIZER_PAYLOAD_FILES",
            "_WEIGHT_FILES",
            "_WEIGHT_INDEX_FILES",
        }
        nodes = []
        for node in ast.parse(self.source).body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                nodes.append(node)
            elif (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in constants
            ):
                nodes.append(node)
        self.assertEqual(
            {node.name for node in nodes if isinstance(node, ast.FunctionDef)},
            names,
        )

        def default_statistics_writer(statistics, directory):
            Path(directory, "dataset_statistics.json").write_text(
                json.dumps(statistics)
            )

        namespace = {
            "Path": Path,
            "json": json,
            "os": os,
            "shutil": shutil,
            "tempfile": tempfile,
            "save_dataset_statistics": (
                statistics_writer or default_statistics_writer
            ),
        }
        exec(_compile(nodes), namespace)
        return SimpleNamespace(**{name: namespace[name] for name in names})

    @staticmethod
    def complete_processor():
        def save(directory):
            directory = Path(directory)
            for name in (
                "preprocessor_config.json",
                "processor_config.json",
                "tokenizer_config.json",
                "tokenizer.json",
            ):
                (directory / name).write_text("{}")

        return SimpleNamespace(save_pretrained=save)

    @staticmethod
    def complete_model():
        def save(directory):
            directory = Path(directory)
            (directory / "config.json").write_text("{}")
            (directory / "model.safetensors").write_bytes(b"weights")

        return SimpleNamespace(save_pretrained=save)

    def test_final_publish_is_complete_and_atomic(self):
        helpers = self.export_helpers()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final = root / "model"
            helpers._publish_final_model(
                self.complete_model(),
                self.complete_processor(),
                {"ur3e_vla_dataset": {"action": {"q01": [0.0]}}},
                final,
            )
            self.assertTrue(final.is_dir())
            helpers._require_complete_model_dir(final)
            self.assertEqual(list(root.glob(".model.staging-*")), [])
            self.assertTrue((final / "model.safetensors").stat().st_size > 0)
            self.assertIn(
                "ur3e_vla_dataset",
                json.loads((final / "dataset_statistics.json").read_text()),
            )

    def test_existing_final_directory_is_rejected_without_changes(self):
        calls = []
        helpers = self.export_helpers(
            lambda statistics, directory: calls.append((statistics, directory))
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final = root / "model"
            final.mkdir()
            sentinel = final / "keep.txt"
            sentinel.write_text("existing model")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                helpers._publish_final_model(
                    self.complete_model(), self.complete_processor(), {}, final
                )
            self.assertEqual(sentinel.read_text(), "existing model")
            self.assertEqual(calls, [])
            self.assertEqual(list(root.glob(".model.staging-*")), [])

    def test_publish_failure_removes_staging_and_leaves_no_final_directory(self):
        helpers = self.export_helpers()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final = root / "model"

            def fail_after_partial_write(staging):
                staging = Path(staging)
                (staging / "config.json").write_text("{}")
                raise RuntimeError("synthetic model save failure")

            model = SimpleNamespace(save_pretrained=fail_after_partial_write)
            with self.assertRaisesRegex(RuntimeError, "synthetic model save failure"):
                helpers._publish_final_model(
                    model, self.complete_processor(), {}, final
                )
            self.assertFalse(final.exists())
            self.assertEqual(list(root.glob(".model.staging-*")), [])

    def test_incomplete_or_missing_weight_shard_is_never_published(self):
        helpers = self.export_helpers()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final = root / "model"
            incomplete = SimpleNamespace(
                save_pretrained=lambda staging: Path(
                    staging, "config.json"
                ).write_text("{}")
            )
            with self.assertRaisesRegex(RuntimeError, "Incomplete final model export"):
                helpers._publish_final_model(
                    incomplete, self.complete_processor(), {}, final
                )
            self.assertFalse(final.exists())
            self.assertEqual(list(root.glob(".model.staging-*")), [])

            candidate = root / "candidate"
            candidate.mkdir()
            self.complete_processor().save_pretrained(candidate)
            (candidate / "config.json").write_text("{}")
            (candidate / "dataset_statistics.json").write_text("{}")
            (candidate / "model.safetensors.index.json").write_text(
                json.dumps(
                    {"weight_map": {"layer": "model-00001-of-00002.safetensors"}}
                )
            )
            with self.assertRaisesRegex(
                RuntimeError, "model-00001-of-00002.safetensors"
            ):
                helpers._require_complete_model_dir(candidate)

    def test_existing_final_directory_fails_before_gpu_setup(self):
        function = _function(self.source)
        gpu_check = next(
            index
            for index, node in enumerate(function.body)
            if isinstance(node, ast.Assert)
        )
        validation = _compile(function.body[:gpu_check])
        with tempfile.TemporaryDirectory() as directory:
            final = Path(directory) / "existing"
            final.mkdir()
            (final / "sentinel").write_text("keep")
            cfg = SimpleNamespace(
                max_steps=1,
                save_steps=250,
                grad_accumulation_steps=1,
                vla_path="offline/mock",
                dataset_name="ur3e_vla_dataset",
                final_model_dir=final,
            )
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(
                FileExistsError, "already exists"
            ):
                exec(validation, {"cfg": cfg, "Path": Path, "os": os})
            self.assertEqual((final / "sentinel").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
