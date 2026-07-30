import pathlib
import unittest


class TestReadme(unittest.TestCase):
    def test_product_name_spelled_correctly(self):
        readme = pathlib.Path(__file__).resolve().parent.parent / "README.md"
        content = readme.read_text(encoding="utf-8")
        self.assertNotIn("Night Copilot", content)
        self.assertIn("Nsight Copilot", content)


if __name__ == "__main__":
