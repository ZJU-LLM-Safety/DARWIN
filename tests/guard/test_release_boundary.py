from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ReleaseBoundaryTests(unittest.TestCase):
    def test_package_metadata_identifies_public_project(self):
        metadata = (ROOT / "pyproject.toml").read_text()
        self.assertIn("authors =", metadata)
        for author in ("Weiwei Qi", "Zefeng Wu", "Zhilin Guo", "Tianhang Zheng",
                       "Chaochao Lu", "Liang He", "Zhan Qin", "Kui Ren"):
            self.assertIn(f'name = "{author}"', metadata)
        for url in ("https://github.com/ZJU-LLM-Safety/DARWIN",
                    "https://arxiv.org/abs/2607.19829",
                    "https://huggingface.co/ZJU-Safety/DARWIN-Guard"):
            self.assertIn(url, metadata)

    def test_source_has_no_machine_home_paths_or_literal_credentials(self):
        for folder in ("src", "configs"):
            for path in (ROOT / folder).rglob("*"):
                if not path.is_file() or "__pycache__" in path.parts:
                    continue
                text = path.read_text()
                self.assertIsNone(re.search(r"/(?:Users|home|data/home)/[A-Za-z0-9_.-]+", text), str(path.relative_to(ROOT)))
                self.assertIsNone(re.search(r"sk-[A-Za-z0-9_-]{24,}", text), str(path.relative_to(ROOT)))
                self.assertIsNone(re.search(r"https?://(?:10\.|192\.168\.|127\.0\.0\.)", text), str(path.relative_to(ROOT)))

    def test_no_experiment_payload_is_distributed(self):
        data_files = sorted(str(path.relative_to(ROOT / "data")) for path in (ROOT / "data").rglob("*") if path.is_file())
        self.assertEqual(data_files, [".gitkeep"])
        for extension in ("*.sqlite", "*.db", "*.safetensors", "*.pt", "*.log"):
            self.assertEqual(list(ROOT.rglob(extension)), [])


if __name__ == "__main__":
    unittest.main()
