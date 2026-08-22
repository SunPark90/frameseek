from __future__ import annotations

import hashlib
import hmac
from pathlib import Path

from .backends.base import PreparedFrame, ResearchBackend
from .errors import EvidenceError
from .models import ResearchAnswer, ResearchCitation, VideoIndex
from .retrieval import rank_frames


def load_verified_index(index_path: str | Path) -> VideoIndex:
    source_path = Path(index_path)
    index = VideoIndex.load(source_path)
    for frame in index.frames:
        if frame.sha256 is None:
            raise EvidenceError(
                f"cannot verify indexed frame without a SHA-256 digest: {frame.path}"
            )
        _resolve_frame_path(
            source_path.parent,
            frame.path,
            expected_sha256=frame.sha256,
        )
    return index


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
            path=_resolve_frame_path(
                source_path.parent,
                item.frame.path,
                expected_sha256=item.frame.sha256,
            ),
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


def _resolve_frame_path(
    index_dir: Path,
    frame_path: str,
    *,
    expected_sha256: str | None = None,
) -> Path:
    candidate = Path(frame_path)
    if candidate.is_absolute():
        raise EvidenceError(f"indexed frame path must be relative: {frame_path}")
    index_root = index_dir.resolve()
    resolved = (index_root / candidate).resolve()
    if not resolved.is_relative_to(index_root):
        raise EvidenceError(f"indexed frame escapes the index directory: {frame_path}")
    if not resolved.is_file():
        raise EvidenceError(f"indexed frame is missing: {resolved}")
    if expected_sha256 is not None:
        actual_sha256 = _sha256(resolved)
        if not hmac.compare_digest(actual_sha256, expected_sha256.casefold()):
            raise EvidenceError(f"indexed frame failed SHA-256 verification: {frame_path}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise EvidenceError(f"cannot verify indexed frame {path}: {exc}") from exc
    return digest.hexdigest()
