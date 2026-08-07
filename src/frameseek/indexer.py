from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

from .media import FFmpegMediaTool, build_sample_timestamps
from .models import FrameRecord, VideoIndex


class FrameCaptioner(Protocol):
    def caption_frame(self, path: Path, timestamp_seconds: float) -> str:
        """Return a factual caption for one frame."""


def create_index(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    interval_seconds: float = 30.0,
    max_frames: int = 24,
    max_width: int = 1280,
    media_tool: FFmpegMediaTool | None = None,
    captioner: FrameCaptioner | None = None,
) -> tuple[VideoIndex, Path]:
    source = Path(video_path)
    destination = Path(output_dir)
    media = media_tool or FFmpegMediaTool()
    metadata = media.probe(source)
    timestamps = build_sample_timestamps(
        metadata.duration_seconds,
        interval_seconds=interval_seconds,
        max_frames=max_frames,
    )
    frame_dir = destination / "frames"
    paths = media.extract_frames(source, frame_dir, timestamps, max_width=max_width)
    records: list[FrameRecord] = []
    for number, (timestamp, path) in enumerate(zip(timestamps, paths, strict=True), start=1):
        caption = captioner.caption_frame(path, timestamp).strip() if captioner else None
        relative_path = path.relative_to(destination).as_posix()
        records.append(
            FrameRecord(
                id=f"f{number:06d}",
                timestamp_seconds=timestamp,
                path=relative_path,
                caption=caption or None,
                sha256=_sha256(path),
            )
        )
    index = VideoIndex(video=metadata, frames=tuple(records))
    index_path = destination / "index.json"
    index.save(index_path)
    return index, index_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
