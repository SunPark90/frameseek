from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreparedFrame:
    id: str
    timestamp_seconds: float
    path: Path
    caption: str | None
    retrieval_score: float


@dataclass(frozen=True)
class EvidenceRef:
    frame_id: str
    claim: str


@dataclass(frozen=True)
class BackendAnswer:
    answer: str
    evidence: tuple[EvidenceRef, ...]


class ResearchBackend(ABC):
    name: str
    model: str

    @abstractmethod
    def caption_frame(self, path: Path, timestamp_seconds: float) -> str:
        """Describe a frame without inferring facts that are not visible."""

    @abstractmethod
    def answer(self, question: str, frames: tuple[PreparedFrame, ...]) -> BackendAnswer:
        """Answer from inspected frames and return frame-level evidence."""
