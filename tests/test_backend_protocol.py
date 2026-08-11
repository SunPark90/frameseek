import os
import unittest
from unittest.mock import patch

from frameseek.backends.openai_compatible import OpenAICompatibleBackend, parse_backend_answer
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
            patch("frameseek.backends.openai_compatible.urllib.request.urlopen") as urlopen,
            self.assertRaisesRegex(BackendError, "insecure HTTP"),
        ):
            backend._chat(messages=[], max_tokens=1)

        urlopen.assert_not_called()

    def test_rejects_non_http_endpoint_before_network_request(self) -> None:
        backend = OpenAICompatibleBackend(
            model="test-model",
            base_url="file:///tmp/model-api",
        )
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "secret"}),
            patch("frameseek.backends.openai_compatible.urllib.request.urlopen") as urlopen,
            self.assertRaisesRegex(BackendError, "HTTP or HTTPS"),
        ):
            backend._chat(messages=[], max_tokens=1)

        urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
