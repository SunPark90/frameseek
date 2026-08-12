import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from frameseek.errors import MediaError
from frameseek.media import FFmpegMediaTool, build_sample_timestamps, parse_fraction


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

    def test_probe_uses_container_duration_when_stream_duration_is_unavailable(self) -> None:
        payload = {
            "streams": [
                {
                    "codec_type": "video",
                    "duration": "N/A",
                    "width": 640,
                    "height": 360,
                    "avg_frame_rate": "30/1",
                }
            ],
            "format": {"duration": "12.5"},
        }
        completed = subprocess.CompletedProcess(
            args=["ffprobe"],
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )
        media = FFmpegMediaTool()
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "sample.mp4"
            video.write_bytes(b"fake video")
            with (
                patch.object(media, "check_dependencies"),
                patch.object(media, "_run", return_value=completed),
            ):
                metadata = media.probe(video)

        self.assertEqual(metadata.duration_seconds, 12.5)

    def test_media_tool_rejects_invalid_timeout(self) -> None:
        with self.assertRaises(ValueError):
            FFmpegMediaTool(timeout_seconds=0)

    def test_media_command_times_out(self) -> None:
        media = FFmpegMediaTool(timeout_seconds=1.5)
        with (
            patch(
                "frameseek.media.subprocess.run",
                side_effect=subprocess.TimeoutExpired(["ffprobe"], 1.5),
            ) as run,
            self.assertRaisesRegex(MediaError, "timed out after 1.5 seconds"),
        ):
            media._run(["ffprobe"], "ffprobe")

        self.assertEqual(run.call_args.kwargs["timeout"], 1.5)


if __name__ == "__main__":
    unittest.main()
