from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

from . import __version__
from .backends import create_backend
from .errors import FrameSeekError
from .indexer import create_index
from .models import VideoIndex, format_timestamp
from .pipeline import research


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frameseek",
        description="Evidence-grounded deep research over video with timestamp citations.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser(
        "index",
        help="extract and optionally caption video frames",
    )
    index_parser.add_argument("video", type=Path)
    index_parser.add_argument("--output", type=Path)
    index_parser.add_argument(
        "--interval",
        type=float,
        default=30.0,
        help="target seconds per frame",
    )
    index_parser.add_argument("--max-frames", type=int, default=24)
    index_parser.add_argument("--max-width", type=int, default=1280)
    index_parser.add_argument(
        "--caption-backend",
        choices=("none", "openai", "smolvlm2"),
        default="none",
    )
    _add_backend_options(index_parser)
    index_parser.set_defaults(handler=_handle_index)

    ask_parser = subparsers.add_parser("ask", help="answer a question from a FrameSeek index")
    ask_parser.add_argument("index", type=Path)
    ask_parser.add_argument("question")
    ask_parser.add_argument(
        "--backend",
        choices=("openai", "smolvlm2"),
        default=os.environ.get("FRAMESEEK_BACKEND", "openai"),
    )
    ask_parser.add_argument("--top-k", type=int, default=8)
    ask_parser.add_argument("--allow-uncited", action="store_true")
    ask_parser.add_argument("--json", action="store_true", dest="as_json")
    _add_backend_options(ask_parser)
    ask_parser.set_defaults(handler=_handle_ask)

    inspect_parser = subparsers.add_parser("inspect", help="show index metadata")
    inspect_parser.add_argument("index", type=Path)
    inspect_parser.add_argument("--json", action="store_true", dest="as_json")
    inspect_parser.set_defaults(handler=_handle_inspect)

    doctor_parser = subparsers.add_parser("doctor", help="check optional runtime dependencies")
    doctor_parser.set_defaults(handler=_handle_doctor)
    return parser


def _add_backend_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=os.environ.get("FRAMESEEK_MODEL"))
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")


def _handle_index(args: argparse.Namespace) -> int:
    destination = args.output or args.video.with_name(f"{args.video.stem}.frameseek")
    captioner = None
    if args.caption_backend != "none":
        captioner = create_backend(
            args.caption_backend,
            model=args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
        )
    index, index_path = create_index(
        args.video,
        destination,
        interval_seconds=args.interval,
        max_frames=args.max_frames,
        max_width=args.max_width,
        captioner=captioner,
    )
    captioned = sum(frame.caption is not None for frame in index.frames)
    print(f"index: {index_path}")
    print(f"duration: {format_timestamp(index.video.duration_seconds)}")
    print(f"frames: {len(index.frames)} ({captioned} captioned)")
    return 0


def _handle_ask(args: argparse.Namespace) -> int:
    backend = create_backend(
        args.backend,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
    )
    result = research(
        args.index,
        args.question,
        backend,
        top_k=args.top_k,
        allow_uncited=args.allow_uncited,
    )
    if args.as_json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0
    print(result.answer)
    if result.citations:
        print("\nEvidence:")
        for citation in result.citations:
            print(
                f"- [{format_timestamp(citation.timestamp_seconds)}] "
                f"{citation.frame_id}: {citation.claim}"
            )
    else:
        print("\nEvidence: none (uncited output was explicitly allowed)")
    return 0


def _handle_inspect(args: argparse.Namespace) -> int:
    index = VideoIndex.load(args.index)
    if args.as_json:
        print(json.dumps(index.to_dict(), ensure_ascii=False, indent=2))
        return 0
    print(f"source: {index.video.source}")
    print(f"duration: {format_timestamp(index.video.duration_seconds)}")
    print(f"frames: {len(index.frames)}")
    print(f"captioned: {sum(frame.caption is not None for frame in index.frames)}")
    print(f"schema: {index.schema_version}")
    return 0


def _handle_doctor(args: argparse.Namespace) -> int:
    del args
    checks = {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "ffprobe": shutil.which("ffprobe") is not None,
        "Pillow (SmolVLM2 extra)": importlib.util.find_spec("PIL") is not None,
        "torch (SmolVLM2 extra)": importlib.util.find_spec("torch") is not None,
        "transformers (SmolVLM2 extra)": importlib.util.find_spec("transformers") is not None,
    }
    for label, available in checks.items():
        print(f"{'ok' if available else 'missing':7} {label}")
    return 0 if checks["ffmpeg"] and checks["ffprobe"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        return int(args.handler(args))
    except (FrameSeekError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
