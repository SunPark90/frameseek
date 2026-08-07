import unittest

from frameseek.backends.openai_compatible import parse_backend_answer
from frameseek.errors import BackendProtocolError


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


if __name__ == "__main__":
    unittest.main()
