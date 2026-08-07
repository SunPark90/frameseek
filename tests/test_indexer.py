import tempfile
import unittest
from pathlib import Path

from frameseek.indexer import create_index
from frameseek.models import VideoMetadata


class FakeMediaTool:
    def probe(self, source: Path) -> VideoMetadata:
        return VideoMetadata(
            source=str(source),
            duration_seconds=60.0,
            width=640,
            height=360,
            fps=30,
        )

    def extract_frames(
        self,
        source: Path,
        output_dir: Path,
        timestamps: tuple[float, ...],
        max_width: int = 1280,
    ) -> tuple[Path, ...]:
        del source, max_width
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for number, timestamp in enumerate(timestamps, start=1):
            path = output_dir / f"frame_{number:06d}.jpg"
            path.write_bytes(f"frame:{timestamp}".encode())
            paths.append(path)
        return tuple(paths)


class FakeCaptioner:
    def caption_frame(self, path: Path, timestamp_seconds: float) -> str:
        return f"Visible frame {path.name} at {timestamp_seconds:.3f} seconds"


class IndexerTests(unittest.TestCase):
    def test_create_index_with_captioner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "sample.mp4"
            video.write_bytes(b"not a real video; fake media tool handles it")
            index, index_path = create_index(
                video,
                root / "sample.frameseek",
                interval_seconds=20,
                max_frames=10,
                media_tool=FakeMediaTool(),
                captioner=FakeCaptioner(),
            )
            self.assertTrue(index_path.is_file())
            self.assertEqual(len(index.frames), 3)
            self.assertTrue(all(frame.caption for frame in index.frames))
            self.assertTrue(all(frame.sha256 for frame in index.frames))
            self.assertTrue(all(not Path(frame.path).is_absolute() for frame in index.frames))


if __name__ == "__main__":
    unittest.main()
