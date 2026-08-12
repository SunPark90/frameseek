from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .errors import DependencyError, MediaError
from .models import VideoMetadata


def build_sample_timestamps(
    duration_seconds: float,
    interval_seconds: float = 30.0,
    max_frames: int = 24,
) -> tuple[float, ...]:
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive and finite")
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive and finite")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")

    estimated_count = max(1, math.ceil(duration_seconds / interval_seconds))
    count = min(max_frames, estimated_count)
    bucket_size = duration_seconds / count
    offset = min(0.25, bucket_size / 2)
    upper_bound = max(0.0, duration_seconds - 0.001)
    return tuple(round(min(upper_bound, index * bucket_size + offset), 3) for index in range(count))


def parse_fraction(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            denominator_value = float(denominator)
            if denominator_value == 0:
                return None
            result = float(numerator) / denominator_value
        else:
            result = float(value)
    except ValueError:
        return None
    if not math.isfinite(result) or result <= 0:
        return None
    return result


def _parse_positive_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or result <= 0:
        return None
    return result


class FFmpegMediaTool:
    def __init__(
        self,
        ffmpeg: str = "ffmpeg",
        ffprobe: str = "ffprobe",
        timeout_seconds: float = 300.0,
    ) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.timeout_seconds = timeout_seconds

    def check_dependencies(self) -> None:
        missing = [name for name in (self.ffmpeg, self.ffprobe) if shutil.which(name) is None]
        if missing:
            names = ", ".join(missing)
            raise DependencyError(
                f"missing media executable(s): {names}. Install FFmpeg and ensure they are on PATH."
            )

    def probe(self, source: str | Path) -> VideoMetadata:
        self.check_dependencies()
        source_path = Path(source)
        if not source_path.is_file():
            raise MediaError(f"video does not exist: {source_path}")
        command = [
            self.ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(source_path),
        ]
        completed = self._run(command, "ffprobe")
        try:
            payload: dict[str, Any] = json.loads(completed.stdout)
            stream = next(
                item
                for item in payload.get("streams", [])
                if item.get("codec_type") == "video"
            )
            format_data = payload.get("format", {})
            duration = _parse_positive_float(stream.get("duration"))
            if duration is None:
                duration = _parse_positive_float(format_data.get("duration"))
            if duration is None:
                raise ValueError("no positive finite duration")
            width = int(stream["width"]) if stream.get("width") else None
            height = int(stream["height"]) if stream.get("height") else None
            fps = parse_fraction(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
        except (KeyError, StopIteration, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise MediaError(f"ffprobe returned incomplete video metadata: {exc}") from exc
        metadata = VideoMetadata(
            source=str(source_path.resolve()),
            duration_seconds=duration,
            width=width,
            height=height,
            fps=fps,
        )
        metadata.validate()
        return metadata

    def extract_frames(
        self,
        source: str | Path,
        output_dir: str | Path,
        timestamps: tuple[float, ...],
        max_width: int = 1280,
    ) -> tuple[Path, ...]:
        self.check_dependencies()
        if max_width <= 0:
            raise ValueError("max_width must be positive")
        source_path = Path(source)
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        for number, timestamp in enumerate(timestamps, start=1):
            output = destination / f"frame_{number:06d}.jpg"
            command = [
                self.ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(source_path),
                "-map",
                "0:v:0",
                "-frames:v",
                "1",
                "-vf",
                f"scale='min({max_width},iw)':-2",
                "-q:v",
                "2",
                "-y",
                str(output),
            ]
            self._run(command, f"frame extraction at {timestamp:.3f}s")
            if not output.is_file() or output.stat().st_size == 0:
                raise MediaError(f"ffmpeg did not create frame at {timestamp:.3f}s")
            outputs.append(output)
        return tuple(outputs)

    def _run(self, command: list[str], operation: str) -> subprocess.CompletedProcess[str]:
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise MediaError(
                f"{operation} timed out after {self.timeout_seconds:g} seconds"
            ) from exc
        except OSError as exc:
            raise MediaError(f"cannot start {operation}: {exc}") from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
            raise MediaError(f"{operation} failed: {detail}")
        return completed
