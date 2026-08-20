from __future__ import annotations

from pathlib import Path
from typing import Any

from ..errors import BackendError, DependencyError
from .base import BackendAnswer, PreparedFrame, ResearchBackend
from .openai_compatible import SYSTEM_PROMPT, parse_backend_answer


class SmolVLM2Backend(ResearchBackend):
    name = "smolvlm2"

    def __init__(
        self,
        *,
        model: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        max_new_tokens: int = 1200,
    ) -> None:
        self.model = model
        self.max_new_tokens = max_new_tokens
        self._processor: Any = None
        self._model_instance: Any = None
        self._torch: Any = None
        self._image_module: Any = None
        self._device = "cpu"

    def caption_frame(self, path: Path, timestamp_seconds: float) -> str:
        prompt = (
            f"Describe only what is visibly present at {timestamp_seconds:.3f} seconds. "
            "Include objects, actions, readable text, and scene context. Do not speculate."
        )
        text = self._generate((path,), prompt, max_new_tokens=300).strip()
        if not text:
            raise BackendError("SmolVLM2 returned an empty caption")
        return text

    def answer(self, question: str, frames: tuple[PreparedFrame, ...]) -> BackendAnswer:
        if not frames:
            raise BackendError("at least one frame is required")
        manifest = "\n".join(
            f"- {frame.id}: {frame.timestamp_seconds:.3f}s; caption={frame.caption or 'none'}"
            for frame in frames
        )
        prompt = (
            f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n\nAllowed frames:\n{manifest}\n\n"
            "The images follow in the same order as the manifest. Return the JSON object now."
        )
        text = self._generate(
            tuple(frame.path for frame in frames),
            prompt,
            max_new_tokens=self.max_new_tokens,
        )
        return parse_backend_answer(text)

    def _load(self) -> None:
        if self._model_instance is not None:
            return
        try:
            import torch
            from PIL import Image
            from transformers import AutoProcessor
            try:
                from transformers import AutoModelForImageTextToText as AutoVisionModel
            except ImportError:
                from transformers import AutoModelForVision2Seq as AutoVisionModel
        except ImportError as exc:
            raise DependencyError(
                "SmolVLM2 dependencies are missing. Install with: pip install -e .[smolvlm2]"
            ) from exc
        self._torch = torch
        self._image_module = Image
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        try:
            self._processor = AutoProcessor.from_pretrained(self.model)
            self._model_instance = AutoVisionModel.from_pretrained(
                self.model,
                torch_dtype=dtype,
            ).to(self._device)
            self._model_instance.eval()
        except Exception as exc:
            raise BackendError(f"cannot load SmolVLM2 model {self.model}: {exc}") from exc

    def _generate(self, paths: tuple[Path, ...], prompt: str, *, max_new_tokens: int) -> str:
        self._load()
        images = []
        try:
            for path in paths:
                images.append(_open_rgb_image(self._image_module, path))
            content = [{"type": "image"} for _ in images]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            rendered = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = self._processor(text=rendered, images=images, return_tensors="pt")
            moved = {
                key: value.to(self._device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            input_length = moved["input_ids"].shape[1]
            with self._torch.inference_mode():
                generated = self._model_instance.generate(
                    **moved,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            continuation = generated[:, input_length:]
            return self._processor.batch_decode(
                continuation,
                skip_special_tokens=True,
            )[0]
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"SmolVLM2 inference failed: {exc}") from exc
        finally:
            for image in images:
                image.close()


def _open_rgb_image(image_module: Any, path: Path) -> Any:
    with image_module.open(path) as source:
        return source.convert("RGB")
