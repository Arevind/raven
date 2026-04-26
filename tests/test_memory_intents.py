import unittest

from voicebot.webapp.service import DashboardService


class MemoryIntentTests(unittest.TestCase):
    def test_add_note_mentioning_is_detected_as_note_intent(self) -> None:
        text = "Add a note mentioning I have completed 3 out of 5 questions for my HW"
        self.assertTrue(DashboardService._looks_like_note(text))

    def test_add_note_mentioning_prefix_is_trimmed_in_summary(self) -> None:
        text = "Add a note mentioning I have completed 3 out of 5 questions for my HW."
        summary = DashboardService._summarize_memory(text)
        self.assertEqual(summary, "I have completed 3 out of 5 questions for my HW")

    def test_create_new_note_telling_is_detected_as_note_intent(self) -> None:
        text = "create a new note telling I am a good kid"
        self.assertTrue(DashboardService._looks_like_note(text))

    def test_create_new_note_telling_prefix_is_trimmed_in_summary(self) -> None:
        text = "create a new note telling I am a good kid"
        summary = DashboardService._summarize_memory(text)
        self.assertEqual(summary, "I am a good kid")

    def test_add_reminder_for_is_detected_as_reminder_intent(self) -> None:
        text = "Add a reminder for buying groceries"
        self.assertTrue(DashboardService._looks_like_reminder(text))

    def test_add_reminder_for_prefix_is_trimmed_in_summary(self) -> None:
        text = "Add a reminder for buying groceries."
        summary = DashboardService._summarize_memory(text)
        self.assertEqual(summary, "buying groceries")


if __name__ == "__main__":
    unittest.main()
