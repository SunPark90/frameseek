import io
import math
import os
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

from frameseek.backends.openai_compatible import (
    OpenAICompatibleBackend,
    _image_data_url,
    _SameOriginRedirectHandler,
    parse_backend_answer,
)
from frameseek.errors import BackendError, BackendProtocolError


class BackendProtocolTests(unittest.TestCase):
    def test_parse_strict_json(self) -> None:
        result = parse_backend_answer(
            '{"answer":"A dog runs.","evidence":['
            '{"frame_id":"f000001","claim":"A dog is visible."}]}'
        )
        self.assertEqual(result.answer, "A dog runs.")
        self.assertEqual(result.evidence[0].frame_id, "f000001")

    def test_parse_markdown_fence_for_compatibility(self) -> None:
        result = parse_backend_answer(
            "```json\n{\"answer\":\"ok\",\"evidence\":[]}\n```"
        )
        self.assertEqual(result.answer, "ok")

    def test_reject_unstructured_text(self) -> None:
        with self.assertRaises(BackendProtocolError):
            parse_backend_answer("The answer is probably a dog.")

    def test_refuses_api_key_over_insecure_remote_http(self) -> None:
        backend = OpenAICompatibleBackend(
            model="test-model",
            base_url="http://models.example.com/v1",
        )
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "secret"}),
            patch("frameseek.backends.openai_compatible._MODEL_API_OPENER.open") as open_api,
            self.assertRaisesRegex(BackendError, "insecure HTTP"),
        ):
            backend._chat(messages=[], max_tokens=1)

        open_api.assert_not_called()

    def test_rejects_non_http_endpoint_before_network_request(self) -> None:
        backend = OpenAICompatibleBackend(
            model="test-model",
            base_url="file:///tmp/model-api",
        )
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "secret"}),
            patch("frameseek.backends.openai_compatible._MODEL_API_OPENER.open") as open_api,
            self.assertRaisesRegex(BackendError, "HTTP or HTTPS"),
        ):
            backend._chat(messages=[], max_tokens=1)

        open_api.assert_not_called()

    def test_rejects_oversized_model_response(self) -> None:
        backend = OpenAICompatibleBackend(
            model="test-model",
            response_limit_bytes=16,
        )
        response = io.BytesIO(b"{" + b"x" * 16)
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "secret"}),
            patch(
                "frameseek.backends.openai_compatible._MODEL_API_OPENER.open",
                return_value=response,
            ),
            self.assertRaisesRegex(BackendProtocolError, "exceeds 16 bytes"),
        ):
            backend._chat(messages=[], max_tokens=1)

    def test_rejects_invalid_response_limit(self) -> None:
        with self.assertRaises(ValueError):
            OpenAICompatibleBackend(model="test-model", response_limit_bytes=0)

    def test_rejects_invalid_request_timeout(self) -> None:
        for timeout_seconds in (0, -1, math.inf, math.nan):
            with (
                self.subTest(timeout_seconds=timeout_seconds),
                self.assertRaisesRegex(ValueError, "positive and finite"),
            ):
                OpenAICompatibleBackend(
                    model="test-model",
                    timeout_seconds=timeout_seconds,
                )

    def test_api_key_is_not_forwarded_across_redirects(self) -> None:
        backend = OpenAICompatibleBackend(model="test-model")
        response = io.BytesIO(b'{"choices":[]}')
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "secret"}),
            patch(
                "frameseek.backends.openai_compatible._MODEL_API_OPENER.open",
                return_value=response,
            ) as open_api,
        ):
            backend._chat(messages=[], max_tokens=1)

        request = open_api.call_args.args[0]
        self.assertEqual(request.unredirected_hdrs["Authorization"], "Bearer secret")
        self.assertNotIn("Authorization", request.headers)

    def test_rejects_cross_origin_model_api_redirect(self) -> None:
        handler = _SameOriginRedirectHandler()
        request = urllib.request.Request(
            "https://models.example.com/v1/chat/completions",
            data=b"{}",
            method="POST",
        )
        with self.assertRaisesRegex(BackendError, "different origin"):
            handler.redirect_request(
                request,
                None,
                302,
                "Found",
                {},
                "https://collector.example/upload",
            )

    def test_rejects_oversized_frame_before_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.jpg"
            path.write_bytes(b"image")
            with (
                patch("frameseek.backends.openai_compatible.MAX_FRAME_BYTES", 4),
                self.assertRaisesRegex(BackendError, "exceeds 4 bytes"),
            ):
                _image_data_url(path)


if __name__ == "__main__":
    unittest.main()
