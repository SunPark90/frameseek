import unittest
from pathlib import Path

from frameseek.backends.smolvlm2 import _open_rgb_image


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
