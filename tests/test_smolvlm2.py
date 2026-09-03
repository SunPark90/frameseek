import unittest
from pathlib import Path

from frameseek.backends.smolvlm2 import SmolVLM2Backend, _open_rgb_image


class FakeSourceImage:
    def __init__(self, converted: object) -> None:
        self.converted = converted
        self.closed = False

    def __enter__(self) -> "FakeSourceImage":
        return self

    def __exit__(self, *args: object) -> None:
        self.closed = True

    def convert(self, mode: str) -> object:
        if mode != "RGB":
            raise AssertionError(f"unexpected mode: {mode}")
        return self.converted


class FakeImageModule:
    def __init__(self, source: FakeSourceImage) -> None:
        self.source = source
        self.opened_path: Path | None = None

    def open(self, path: Path) -> FakeSourceImage:
        self.opened_path = path
        return self.source


class SmolVLM2Tests(unittest.TestCase):
    def test_rejects_empty_model_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "model is required"):
            SmolVLM2Backend(model="  ")

    def test_rejects_invalid_max_new_tokens(self) -> None:
        for max_new_tokens in (0, -1, True, 1.5):
            with (
                self.subTest(max_new_tokens=max_new_tokens),
                self.assertRaisesRegex(ValueError, "positive integer"),
            ):
                SmolVLM2Backend(max_new_tokens=max_new_tokens)

    def test_rgb_conversion_closes_source_image(self) -> None:
        converted = object()
        source = FakeSourceImage(converted)
        image_module = FakeImageModule(source)
        path = Path("frame.jpg")

        result = _open_rgb_image(image_module, path)

        self.assertIs(result, converted)
        self.assertEqual(image_module.opened_path, path)
        self.assertTrue(source.closed)


if __name__ == "__main__":
    unittest.main()
