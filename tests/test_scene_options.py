"""Fast checks for workflow scene selections; no Isaac app is required."""

from __future__ import annotations

import unittest

from vla_sim.scene import SceneOptions, get_scene_options


class SceneOptionsTest(unittest.TestCase):
    def test_known_workflows_select_expected_cameras(self) -> None:
        self.assertEqual(
            get_scene_options("canonical").cameras,
            ("camera_yolo", "camera_policy", "camera_wrist"),
        )
        self.assertEqual(get_scene_options("remote").cameras, ("camera_yolo",))
        self.assertEqual(
            get_scene_options("collection").cameras,
            ("camera_policy", "camera_wrist"),
        )

    def test_option_instance_is_preserved(self) -> None:
        option = SceneOptions(name="test", cameras=())
        self.assertIs(get_scene_options(option), option)

    def test_unknown_workflow_fails_with_choices(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown scene options"):
            get_scene_options("missing")


if __name__ == "__main__":
    unittest.main()
