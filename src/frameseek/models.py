from __future__ import annotations

import json
import math
import os
import re
import tempfile
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import IndexFormatError

SCHEMA_VERSION = 1
MAX_INDEX_BYTES = 16 * 1024 * 1024
SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def format_timestamp(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("timestamp must be a finite, non-negative number")
    total_milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    if milliseconds:
        return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}"


@dataclass(frozen=True)
class VideoMetadata:
    source: str
    duration_seconds: float
    width: int | None = None
    height: int | None = None
    fps: float | None = None

    def validate(self) -> None:
        if not self.source:
            raise IndexFormatError("video.source is required")
        if not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0:
            raise IndexFormatError("video.duration_seconds must be positive and finite")
        if self.width is not None and self.width <= 0:
            raise IndexFormatError("video.width must be positive")
        if self.height is not None and self.height <= 0:
            raise IndexFormatError("video.height must be positive")
        if self.fps is not None and (not math.isfinite(self.fps) or self.fps <= 0):
            raise IndexFormatError("video.fps must be positive and finite")


@dataclass(frozen=True)
class FrameRecord:
    id: str
    timestamp_seconds: float
    path: str
    caption: str | None = None
    sha256: str | None = None

    def validate(self, duration_seconds: float) -> None:
        if not self.id:
            raise IndexFormatError("frame.id is required")
        if not self.path:
            raise IndexFormatError(f"frame {self.id!r} has no path")
        if not math.isfinite(self.timestamp_seconds) or self.timestamp_seconds < 0:
            raise IndexFormatError(f"frame {self.id!r} has an invalid timestamp")
        if self.timestamp_seconds > duration_seconds + 0.001:
            raise IndexFormatError(f"frame {self.id!r} is outside the video duration")
        if self.sha256 is not None and SHA256_PATTERN.fullmatch(self.sha256) is None:
            raise IndexFormatError(f"frame {self.id!r} has an invalid SHA-256 digest")


@dataclass(frozen=True)
class VideoIndex:
    video: VideoMetadata
    frames: tuple[FrameRecord, ...]
    created_at: str = field(default_factory=utc_now)
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise IndexFormatError(
                f"unsupported schema version {self.schema_version}; expected {SCHEMA_VERSION}"
            )
        self.video.validate()
        if not self.frames:
            raise IndexFormatError("index must contain at least one frame")
        ids: set[str] = set()
        previous_timestamp = -1.0
        for frame in self.frames:
            frame.validate(self.video.duration_seconds)
            if frame.id in ids:
                raise IndexFormatError(f"duplicate frame id: {frame.id}")
            if frame.timestamp_seconds < previous_timestamp:
                raise IndexFormatError("frames must be sorted by timestamp")
            ids.add(frame.id)
            previous_timestamp = frame.timestamp_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "video": asdict(self.video),
            "frames": [asdict(frame) for frame in self.frames],
        }

    def save(self, path: str | Path) -> None:
        self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(destination)
        except OSError as exc:
            raise IndexFormatError(f"cannot write index {destination}: {exc}") from exc
        finally:
            if temporary is not None:
                with suppress(OSError):
                    temporary.unlink(missing_ok=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoIndex:
        try:
            video_data = data["video"]
            video = VideoMetadata(
                source=str(video_data["source"]),
                duration_seconds=float(video_data["duration_seconds"]),
                width=_optional_int(video_data.get("width")),
                height=_optional_int(video_data.get("height")),
                fps=_optional_float(video_data.get("fps")),
            )
            frames = tuple(
                FrameRecord(
                    id=str(item["id"]),
                    timestamp_seconds=float(item["timestamp_seconds"]),
                    path=str(item["path"]),
                    caption=_optional_string(item.get("caption")),
                    sha256=_optional_string(item.get("sha256")),
                )
                for item in data["frames"]
            )
            index = cls(
                schema_version=int(data.get("schema_version", 0)),
                created_at=str(data.get("created_at", "")),
                video=video,
                frames=frames,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise IndexFormatError(f"invalid index structure: {exc}") from exc
        index.validate()
        return index

    @classmethod
    def load(cls, path: str | Path) -> VideoIndex:
        source = Path(path)
        try:
            with source.open("rb") as handle:
                raw = handle.read(MAX_INDEX_BYTES + 1)
        except OSError as exc:
            raise IndexFormatError(f"cannot read index {source}: {exc}") from exc
        if len(raw) > MAX_INDEX_BYTES:
            raise IndexFormatError(f"index exceeds {MAX_INDEX_BYTES} bytes: {source}")
        try:
            data = json.loads(raw.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise IndexFormatError(f"index is not valid UTF-8: {source}") from exc
        except json.JSONDecodeError as exc:
            raise IndexFormatError(f"invalid JSON in {source}: {exc}") from exc
        if not isinstance(data, dict):
            raise IndexFormatError("index root must be a JSON object")
        return cls.from_dict(data)


@dataclass(frozen=True)
class ResearchCitation:
    frame_id: str
    timestamp_seconds: float
    claim: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp_seconds": self.timestamp_seconds,
            "timestamp": format_timestamp(self.timestamp_seconds),
            "claim": self.claim,
        }


@dataclass(frozen=True)
class ResearchAnswer:
    question: str
    answer: str
    citations: tuple[ResearchCitation, ...]
    backend: str
    model: str
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": [citation.to_dict() for citation in self.citations],
            "backend": self.backend,
            "model": self.model,
            "created_at": self.created_at,
        }


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
