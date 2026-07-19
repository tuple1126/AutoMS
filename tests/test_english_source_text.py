"""Release safeguard: public source text must use ASCII English only."""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NON_ASCII_PATTERN = re.compile(r"[^\x00-\x7F]")
TEXT_SUFFIXES = {
    ".cff",
    ".env",
    ".ini",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_FILENAMES = {".env.example", ".gitignore", "LICENSE"}
EXCLUDED_DIRECTORIES = {".git", "__pycache__", ".venv", "venv"}


def public_text_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in EXCLUDED_DIRECTORIES for part in path.parts):
            continue
        if path.name in TEXT_FILENAMES or path.suffix.lower() in TEXT_SUFFIXES:
            yield path


class EnglishSourceTextTests(unittest.TestCase):
    def test_public_text_files_contain_ascii_only(self):
        matches = []
        for path in public_text_files():
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as error:
                self.fail(f"Public text file is not UTF-8: {path.relative_to(ROOT)} ({error})")

            match = NON_ASCII_PATTERN.search(content)
            if match:
                line = content.count("\n", 0, match.start()) + 1
                matches.append(f"{path.relative_to(ROOT)}:{line}")

        self.assertEqual(matches, [], f"Non-ASCII text remains in public files: {', '.join(matches)}")


if __name__ == "__main__":
    unittest.main()
