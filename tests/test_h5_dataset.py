"""Fast checks for in-memory H5 collection state; no Isaac app is required."""

from __future__ import annotations

import unittest

from vla_sim.h5_dataset import EpisodeBuffer


class EpisodeBufferTest(unittest.TestCase):
    def test_default_buffers_do_not_share_mutable_lists(self) -> None:
        first = EpisodeBuffer()
        second = EpisodeBuffer()

        first.main_images.append("frame")

        self.assertEqual(first.main_images, ["frame"])
        self.assertEqual(second.main_images, [])


if __name__ == "__main__":
    unittest.main()
