import unittest

from voicebot.providers.kokoro_tts import sanitize_for_tts


class SanitizationTests(unittest.TestCase):
    def test_sanitize_removes_non_bmp_and_control_chars(self) -> None:
        text = "Hello \x00 world 😀\n  with\tspaces"
        cleaned = sanitize_for_tts(text)
        self.assertEqual(cleaned, "Hello world with spaces")


if __name__ == "__main__":
    unittest.main()

