from __future__ import annotations

from pathlib import Path

from .backends.base import PreparedFrame, ResearchBackend
from .errors import EvidenceError
from .models import ResearchAnswer, ResearchCitation, VideoIndex
from .retrieval import rank_frames


def research(
    index_path: str | Path,
    question: str,
    backend: ResearchBackend,
    *,
    top_k: int = 8,
    allow_uncited: bool = False,
) -> ResearchAnswer:
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("question cannot be empty")
    source_path = Path(index_path)
    index = VideoIndex.load(source_path)
    ranked = rank_frames(index, normalized_question, top_k=top_k)
    prepared = tuple(
        PreparedFrame(
            id=item.frame.id,
            timestamp_seconds=item.frame.timestamp_seconds,
            path=_resolve_frame_path(source_path.parent, item.frame.path),
            caption=item.frame.caption,
            retrieval_score=item.score,
        )
        for item in ranked
    )
    response = backend.answer(normalized_question, prepared)
    inspected = {frame.id: frame for frame in prepared}
    citations: list[ResearchCitation] = []
    seen: set[tuple[str, str]] = set()
    for evidence in response.evidence:
        frame = inspected.get(evidence.frame_id)
        claim = evidence.claim.strip()
        if frame is None:
            raise EvidenceError(
                f"backend cited frame {evidence.frame_id!r}, but that frame was not inspected"
            )
        if not claim:
            raise EvidenceError(f"backend returned an empty claim for frame {evidence.frame_id!r}")
        key = (frame.id, claim)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            ResearchCitation(
                frame_id=frame.id,
                timestamp_seconds=frame.timestamp_seconds,
                claim=claim,
            )
        )
    if not citations and not allow_uncited:
        raise EvidenceError(
            "backend returned no valid frame evidence; rerun with another model or --allow-uncited"
        )
    if not response.answer.strip():
        raise EvidenceError("backend returned an empty answer")
    return ResearchAnswer(
        question=normalized_question,
        answer=response.answer.strip(),
        citations=tuple(citations),
        backend=backend.name,
        model=backend.model,
    )


def _resolve_frame_path(index_dir: Path, frame_path: str) -> Path:
    candidate = Path(frame_path)
    resolved = candidate if candidate.is_absolute() else index_dir / candidate
    if not resolved.is_file():
        raise EvidenceError(f"indexed frame is missing: {resolved}")
    return resolved.resolve()
