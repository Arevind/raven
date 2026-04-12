import unittest

from voicebot.text import SentenceChunker


class SentenceChunkerTests(unittest.TestCase):
    def test_sentence_boundaries_respect_punctuation(self) -> None:
        chunker = SentenceChunker(max_chars=120, min_chars=1, soft_chars=0)
        out = chunker.feed("Hello there. How are you? I am good.")
        self.assertEqual(out, ["Hello there. How are you?"])
        self.assertEqual(chunker.flush(), "I am good.")

    def test_long_text_without_punctuation_splits_by_max_chars(self) -> None:
        chunker = SentenceChunker(max_chars=15, min_chars=1, soft_chars=0)
        out = chunker.feed("word1 word2 word3 word4 word5")
        out_tail = chunker.flush()
        self.assertTrue(all(len(part) <= 15 for part in out))
        self.assertIsNotNone(out_tail)

    def test_soft_chars_emits_partial_phrase(self) -> None:
        chunker = SentenceChunker(max_chars=200, min_chars=1, soft_chars=10)
        out = chunker.feed("This should stream early with soft chunking")
        self.assertGreaterEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
