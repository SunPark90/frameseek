import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from frameseek.errors import IndexFormatError
from frameseek.models import FrameRecord, VideoIndex, VideoMetadata, format_timestamp


class ModelTests(unittest.TestCase):
    def test_format_timestamp_includes_hours_and_milliseconds(self) -> None:
        self.assertEqual(format_timestamp(3661.25), "01:01:01.250")
        self.assertEqual(format_timestamp(5), "00:00:05")

    def test_index_round_trip(self) -> None:
        index = VideoIndex(
            video=VideoMetadata(source="sample.mp4", duration_seconds=10.0),
            frames=(FrameRecord(id="f000001", timestamp_seconds=1.0, path="frames/a.jpg"),),
            created_at="2026-08-07T00:00:00+00:00",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            index.save(path)
            self.assertEqual(VideoIndex.load(path), index)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)

    def test_index_rejects_duplicate_frame_ids(self) -> None:
        index = VideoIndex(
            video=VideoMetadata(source="sample.mp4", duration_seconds=10.0),
            frames=(
                FrameRecord(id="same", timestamp_seconds=1.0, path="a.jpg"),
                FrameRecord(id="same", timestamp_seconds=2.0, path="b.jpg"),
            ),
        )
        with self.assertRaisesRegex(IndexFormatError, "duplicate frame id"):
            index.validate()

    def test_index_rejects_frame_after_video_end(self) -> None:
        index = VideoIndex(
            video=VideoMetadata(source="sample.mp4", duration_seconds=2.0),
            frames=(FrameRecord(id="f1", timestamp_seconds=3.0, path="a.jpg"),),
        )
        with self.assertRaisesRegex(IndexFormatError, "outside the video duration"):
            index.validate()

    def test_index_rejects_malformed_frame_digest(self) -> None:
        index = VideoIndex(
            video=VideoMetadata(source="sample.mp4", duration_seconds=2.0),
            frames=(
                FrameRecord(
                    id="f1",
                    timestamp_seconds=1.0,
                    path="a.jpg",
                    sha256="not-a-sha256-digest",
                ),
            ),
        )
        with self.assertRaisesRegex(IndexFormatError, "invalid SHA-256 digest"):
            VideoIndex.from_dict(index.to_dict())

    def test_index_load_rejects_oversized_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            path.write_bytes(b" " * 33)
            with (
                patch("frameseek.models.MAX_INDEX_BYTES", 32),
                self.assertRaisesRegex(IndexFormatError, "exceeds 32 bytes"),
            ):
                VideoIndex.load(path)

    def test_index_load_rejects_invalid_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            path.write_bytes(b"\xff")
            with self.assertRaisesRegex(IndexFormatError, "not valid UTF-8"):
                VideoIndex.load(path)


if __name__ == "__main__":
    unittest.main()
