from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .models import FrameRecord, VideoIndex

TOKEN_PATTERN = re.compile(r"(?u)\b[^\W_]+\b")


@dataclass(frozen=True)
class RankedFrame:
    frame: FrameRecord
    score: float


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in TOKEN_PATTERN.findall(text))


def rank_frames(index: VideoIndex, question: str, top_k: int = 8) -> tuple[RankedFrame, ...]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    limit = min(top_k, len(index.frames))
    question_tokens = tokenize(question)
    captions = [tokenize(frame.caption or "") for frame in index.frames]
    if not question_tokens or not any(captions):
        return _temporal_fallback(index.frames, limit)

    document_frequency = Counter(token for tokens in captions for token in set(tokens))
    question_counts = Counter(question_tokens)
    scored: list[RankedFrame] = []
    for frame, tokens in zip(index.frames, captions, strict=True):
        counts = Counter(tokens)
        normalization = math.sqrt(max(1, len(tokens)))
        score = 0.0
        for token, query_frequency in question_counts.items():
            if token not in counts:
                continue
            inverse_frequency = (
                math.log((len(index.frames) + 1) / (document_frequency[token] + 1)) + 1
            )
            score += inverse_frequency * min(counts[token], 3) * min(query_frequency, 2)
        if question.casefold() in (frame.caption or "").casefold():
            score += 2.0
        scored.append(RankedFrame(frame=frame, score=score / normalization))

    if not any(item.score > 0 for item in scored):
        return _temporal_fallback(index.frames, limit)
    selected = sorted(scored, key=lambda item: (-item.score, item.frame.timestamp_seconds))[:limit]
    return tuple(sorted(selected, key=lambda item: item.frame.timestamp_seconds))


def _temporal_fallback(frames: tuple[FrameRecord, ...], limit: int) -> tuple[RankedFrame, ...]:
    if limit >= len(frames):
        return tuple(RankedFrame(frame=frame, score=0.0) for frame in frames)
    if limit == 1:
        return (RankedFrame(frame=frames[len(frames) // 2], score=0.0),)
    indexes = {
        round(position * (len(frames) - 1) / (limit - 1))
        for position in range(limit)
    }
    return tuple(RankedFrame(frame=frames[index], score=0.0) for index in sorted(indexes))
