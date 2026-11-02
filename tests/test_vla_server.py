"""Check the actual prediction route with a model mock, without loading weights."""

from __future__ import annotations

import base64
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException
import numpy as np
from PIL import Image


class VLAServerTest(unittest.TestCase):
    def setUp(self):
        # Imports are substituted so even collecting these tests cannot load a
        # Transformers model or initialize a CUDA context.
        fake_torch = types.ModuleType("torch")
        fake_torch.no_grad = contextlib.nullcontext
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoModelForVision2Seq = object
        fake_transformers.AutoProcessor = object
        path = Path(__file__).resolve().parents[1] / "scripts/vla_server.py"
        spec = importlib.util.spec_from_file_location("vla_server_test_helper", path)
        self.server = importlib.util.module_from_spec(spec)
        with (
            patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers}),
            patch.object(sys, "argv", ["vla_server_test", "--device", "cpu"]),
        ):
            spec.loader.exec_module(self.server)
        self.model = Mock()
        self.model.dtype = None
        self.server.MODEL = self.model
        self.server.PROCESSOR = Mock(return_value={})
        self.server.DEVICE = "cpu"
        buffer = io.BytesIO()
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(buffer, format="JPEG")
        self.request = self.server.PredictRequest(
            image_b64=base64.b64encode(buffer.getvalue()).decode(),
            instruction="pick the mug",
            unnorm_key="test_dataset",
        )

    def test_finite_action_and_single_batch_return_exactly_seven_values(self):
        expected = [0.02, 0, 0, 0, 0, 0, 1]
        for action in (expected, [expected]):
            with self.subTest(action=action):
                self.model.predict_action.return_value = action
                result = self.server.predict(self.request)
                np.testing.assert_allclose(result.action, expected)
                self.assertEqual(len(result.action), 7)
                self.assertEqual(result.action_raw, result.action)
                self.assertTrue(np.isfinite(result.latency_ms))
                self.assertEqual(self.model.predict_action.call_args.kwargs["unnorm_key"], "test_dataset")

    def test_wrong_size_nonfinite_and_nonnumeric_model_outputs_raise_clear_500(self):
        outputs = (
            [],
            [0] * 6,
            [0] * 8,
            [[0] * 7, [0] * 7],
            [float("nan"), 0, 0, 0, 0, 0, 0],
            [0, 0, float("inf"), 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, float("-inf")],
            ["invalid", 0, 0, 0, 0, 0, 0],
        )
        for action in outputs:
            with self.subTest(action=action):
                self.model.predict_action.return_value = action
                with self.assertRaises(HTTPException) as caught:
                    self.server.predict(self.request)
                self.assertEqual(caught.exception.status_code, 500)
                self.assertIn("Invalid model action:", caught.exception.detail)

    def test_tensor_style_cpu_conversion_is_validated(self):
        action = Mock()
        action.cpu.return_value.numpy.return_value = np.zeros((1, 8), dtype=np.float32)
        self.model.predict_action.return_value = action
        with self.assertRaises(HTTPException) as caught:
            self.server.predict(self.request)
        self.assertIn("expected 7 action values, got 8", caught.exception.detail)
        action.cpu.assert_called_once_with()

    def test_unloaded_model_returns_503(self):
        self.server.MODEL = None
        with self.assertRaises(HTTPException) as caught:
            self.server.predict(self.request)
        self.assertEqual(caught.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
