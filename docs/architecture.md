# Architecture

FrameSeek separates media handling, retrieval, model inference, and evidence
validation. This keeps the core testable without downloading model weights.

```text
video file
  -> ffprobe metadata
  -> deterministic timestamp sampler
  -> ffmpeg frame extraction
  -> optional frame captions
  -> index.json
  -> lexical/temporal frame retrieval
  -> model backend receives selected frames
  -> strict JSON answer protocol
  -> evidence validator
  -> answer + timestamp citations
```

## Trust boundary

The model backend is not trusted to create citations. It can propose frame IDs,
but `pipeline.research` accepts only IDs from frames actually passed to that
backend. The pipeline derives timestamps from the saved index rather than from
model output.

Frame text and captions are also treated as untrusted input. Backend prompts
explicitly tell the model not to follow instructions found inside the video.

## Index schema

Version 1 stores immutable source metadata and ordered frame records. Frame paths
are relative to the index directory, which makes an index portable as one folder.
Each frame includes a SHA-256 digest so future releases can verify media integrity
and cache model results safely.

## Backend contract

A backend implements two methods:

- `caption_frame`: optional indexing-time factual description
- `answer`: answer one question from a tuple of prepared frames

`answer` returns an answer plus `(frame_id, claim)` evidence records. The core
pipeline owns evidence validation and timestamp rendering.

## Planned research extensions

- scene-change and embedding-based keyframe selection
- transcript indexing and audio/video evidence fusion
- web evidence with URL provenance separated from frame evidence
- VideoLLM-online streaming adapter
- reproducible VideoDR-Bench evaluation runner
- retrieval and model-call cache keyed by content digest
