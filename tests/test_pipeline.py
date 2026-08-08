import hashlib
import tempfile
import unittest
from pathlib import Path

from frameseek.backends.base import BackendAnswer, EvidenceRef, PreparedFrame, ResearchBackend
from frameseek.errors import EvidenceError
from frameseek.models import FrameRecord, VideoIndex, VideoMetadata
from frameseek.pipeline import research


class FakeBackend(ResearchBackend):
    name = "fake"
    model = "deterministic-test-model"

    def __init__(self, evidence_frame: str | None = "f000002") -> None:
        self.evidence_frame = evidence_frame
        self.received: tuple[PreparedFrame, ...] = ()

    def caption_frame(self, path: Path, timestamp_seconds: float) -> str:
        del path, timestamp_seconds
        return "unused"

    def answer(self, question: str, frames: tuple[PreparedFrame, ...]) -> BackendAnswer:
        del question
        self.received = frames
        evidence = ()
        if self.evidence_frame is not None:
            evidence = (EvidenceRef(self.evidence_frame, "A red ball is visible."),)
        return BackendAnswer(answer="The dog chases a red ball.", evidence=evidence)


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        frame_dir = self.root / "frames"
        frame_dir.mkdir()
        frames = []
        captions = ["kitchen", "dog with red ball in park", "city street"]
        for number, caption in enumerate(captions, start=1):
            path = frame_dir / f"{number}.jpg"
            path.write_bytes(f"image-{number}".encode())
            frames.append(
                FrameRecord(
                    id=f"f{number:06d}",
                    timestamp_seconds=float(number * 10),
                    path=f"frames/{number}.jpg",
                    caption=caption,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
        index = VideoIndex(
            video=VideoMetadata(source="sample.mp4", duration_seconds=40.0),
            frames=tuple(frames),
        )
        self.index_path = self.root / "index.json"
        index.save(self.index_path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_research_returns_timestamped_verified_citation(self) -> None:
        backend = FakeBackend()
        result = research(self.index_path, "red ball", backend, top_k=2)
        self.assertEqual(result.citations[0].frame_id, "f000002")
        self.assertEqual(result.citations[0].timestamp_seconds, 20.0)
        self.assertIn("f000002", {frame.id for frame in backend.received})

    def test_research_rejects_uninspected_frame(self) -> None:
        backend = FakeBackend(evidence_frame="f999999")
        with self.assertRaisesRegex(EvidenceError, "was not inspected"):
            research(self.index_path, "red ball", backend, top_k=2)

    def test_uncited_output_requires_explicit_override(self) -> None:
        backend = FakeBackend(evidence_frame=None)
        with self.assertRaisesRegex(EvidenceError, "no valid frame evidence"):
            research(self.index_path, "red ball", backend)
        result = research(self.index_path, "red ball", backend, allow_uncited=True)
        self.assertEqual(result.citations, ())

    def test_research_rejects_frame_outside_index_directory(self) -> None:
        nested = self.root / "nested"
        nested.mkdir()
        source_frame = self.root / "frames" / "2.jpg"
        index = VideoIndex(
            video=VideoMetadata(source="sample.mp4", duration_seconds=40.0),
            frames=(
                FrameRecord(
                    id="f000002",
                    timestamp_seconds=20.0,
                    path="../frames/2.jpg",
                    caption="dog with red ball in park",
                    sha256=hashlib.sha256(source_frame.read_bytes()).hexdigest(),
                ),
            ),
        )
        index_path = nested / "index.json"
        index.save(index_path)

        with self.assertRaisesRegex(EvidenceError, "escapes the index directory"):
            research(index_path, "red ball", FakeBackend(), top_k=1)

    def test_research_rejects_modified_frame(self) -> None:
        (self.root / "frames" / "2.jpg").write_bytes(b"tampered-image")

        with self.assertRaisesRegex(EvidenceError, "failed SHA-256 verification"):
            research(self.index_path, "red ball", FakeBackend(), top_k=1)


if __name__ == "__main__":
    unittest.main()
