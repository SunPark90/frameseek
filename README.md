# FrameSeek

FrameSeek is an evidence-grounded video research runtime. It extracts a portable
timeline index from a video, retrieves relevant frames for a question, and asks a
vision-language model to answer with citations to frames it actually inspected.

The project is inspired by the perception-before-exploration design in
[Vision-DeepResearch](https://github.com/Osilly/Vision-DeepResearch) and
[Video-DeepResearch](https://arxiv.org/abs/2608.03979). FrameSeek is an independent
implementation focused on low-cost inference, backend portability, and verifiable
timestamp citations. It is not affiliated with the paper authors.

> Status: alpha. The index format is versioned, but APIs may change before 1.0.

## What works

- deterministic frame extraction through FFmpeg
- portable JSON indexes with frame hashes and optional captions
- Korean and English lexical frame retrieval with temporal fallback
- strict validation that rejects citations to frames the model did not receive
- OpenAI-compatible vision API backend with no runtime SDK dependency
- optional local SmolVLM2 backend
- machine-readable JSON and human-readable CLI output

FrameSeek does not yet perform web search or transcript indexing. Those are
planned as separate evidence sources rather than hidden steps in model prompts.

## Requirements

- Python 3.10 or newer
- `ffmpeg` and `ffprobe` on `PATH` for indexing
- a vision-capable OpenAI-compatible model, or the optional SmolVLM2 dependencies

## Install

```bash
git clone https://github.com/SunPark90/frameseek.git
cd frameseek
python -m pip install -e .
frameseek doctor
```

For local SmolVLM2 inference:

```bash
python -m pip install -e .[smolvlm2]
```

## Quick start

Create an index without making any model calls:

```bash
frameseek index ./meeting.mp4 --interval 20 --max-frames 24
frameseek inspect ./meeting.frameseek/index.json
frameseek inspect ./meeting.frameseek/index.json --verify
```

Captions make retrieval more selective. With an OpenAI-compatible API:

```bash
export OPENAI_API_KEY="..."
frameseek index ./meeting.mp4 \
  --caption-backend openai \
  --model <vision-capable-model>

frameseek ask ./meeting.frameseek/index.json \
  "발표자가 성능 비교표를 설명하는 장면은 어디인가?" \
  --backend openai \
  --model <vision-capable-model>
```

PowerShell uses `$env:OPENAI_API_KEY = "..."` instead of `export`.

Use any local server that exposes `/v1/chat/completions`:

```bash
frameseek ask ./meeting.frameseek/index.json "What changed after the demo?" \
  --backend openai \
  --base-url http://127.0.0.1:8000/v1 \
  --model local-vision-model
```

Local SmolVLM2:

```bash
frameseek index ./meeting.mp4 --caption-backend smolvlm2
frameseek ask ./meeting.frameseek/index.json "Summarize the visible sequence" \
  --backend smolvlm2
```

## Evidence model

Backends return frame IDs and claims, not timestamps. FrameSeek verifies that
each ID belongs to the exact set of frames sent to the backend, then obtains the
timestamp from the index. A model cannot invent `f999999` and turn it into a
valid citation.

Run `frameseek inspect <index.json> --verify` after copying or restoring an
index to check every frame path and SHA-256 digest before inference.

Cloud backends receive the selected JPEG frames and the question, not the full
video. Review [SECURITY.md](SECURITY.md) before processing private material.

## Development

Core tests do not need FFmpeg, model weights, a GPU, or API credentials.

```bash
python -m pip install -e .[dev]
python -m unittest discover -s tests -v
ruff check .
python -m build
```

See [docs/architecture.md](docs/architecture.md) for module boundaries and the
research roadmap. Contributions should follow [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
