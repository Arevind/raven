import unittest

from voicebot.domain.models import ChatTurn
from voicebot.providers.llama_cpp_client import LlamaCppChatClient


class LlamaCppClientMessageTests(unittest.TestCase):
    def test_build_messages_includes_system_prompt_and_memory_context(self) -> None:
        client = LlamaCppChatClient(
            model="m",
            url="http://localhost:8080/v1/chat/completions",
            system_prompt="You are Raven.",
            context_provider=lambda: "Memory context: Notes: - test",
        )
        history = [ChatTurn(user_text="u1", assistant_text="a1")]

        messages = client._build_messages(user_text="u2", history=history)

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Raven", messages[0]["content"])
        self.assertIn("Thinking mode is disabled", messages[0]["content"])
        self.assertIn("Memory context", messages[0]["content"])
        self.assertEqual(messages[-1], {"role": "user", "content": "u2"})

    def test_parse_tool_call_accepts_tagged_format(self) -> None:
        client = LlamaCppChatClient(model="m", url="http://localhost:8080/v1/chat/completions")
        parsed = client._parse_tool_call(
            '<tool_call>{"name":"add_note","arguments":{"summary":"x"}}</tool_call>'
        )
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed[0], "add_note")
        self.assertEqual(parsed[1]["summary"], "x")

    def test_parse_tool_call_accepts_shorthand_format(self) -> None:
        client = LlamaCppChatClient(model="m", url="http://localhost:8080/v1/chat/completions")
        parsed = client._parse_tool_call('add_reminder{"summary":"buy groceries","due_hint":"today"}')
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed[0], "add_reminder")
        self.assertEqual(parsed[1]["summary"], "buy groceries")

    def test_parse_tool_call_accepts_markdown_json_block(self) -> None:
        client = LlamaCppChatClient(model="m", url="http://localhost:8080/v1/chat/completions")
        text = '```json\n{"name":"list_notes","arguments":{"limit":5}}\n```'
        parsed = client._parse_tool_call(text)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed[0], "list_notes")

    def test_tool_result_to_text_for_list_notes(self) -> None:
        text = LlamaCppChatClient._tool_result_to_text(
            "list_notes",
            {"ok": True, "items": [{"summary": "finish hw", "due_hint": None}]},
        )
        self.assertIn("finish hw", text)


if __name__ == "__main__":
    unittest.main()
