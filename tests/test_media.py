import unittest

from frameseek.media import build_sample_timestamps, parse_fraction


class MediaTests(unittest.TestCase):
    def test_sampling_respects_interval_and_limit(self) -> None:
        timestamps = build_sample_timestamps(100.0, interval_seconds=30.0, max_frames=24)
        self.assertEqual(len(timestamps), 4)
        self.assertEqual(timestamps[0], 0.25)
        self.assertLess(timestamps[-1], 100.0)

        capped = build_sample_timestamps(100.0, interval_seconds=1.0, max_frames=3)
        self.assertEqual(len(capped), 3)
        self.assertEqual(capped, tuple(sorted(capped)))

    def test_sampling_rejects_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            build_sample_timestamps(0)
        with self.assertRaises(ValueError):
            build_sample_timestamps(10, interval_seconds=0)
        with self.assertRaises(ValueError):
            build_sample_timestamps(10, max_frames=0)

    def test_fraction_parser(self) -> None:
        self.assertAlmostEqual(parse_fraction("30000/1001") or 0, 29.97002997)
        self.assertEqual(parse_fraction("25"), 25.0)
        self.assertIsNone(parse_fraction("0/0"))
        self.assertIsNone(parse_fraction("bad"))


if __name__ == "__main__":
    unittest.main()
