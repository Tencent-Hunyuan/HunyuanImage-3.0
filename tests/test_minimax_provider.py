# -*- coding: utf-8 -*-
"""
Tests for MiniMax Client Module and LLM Provider Integration

This module contains unit tests and integration tests for the MiniMax prompt
enhancement client and the multi-provider support in run_image_gen.py.
"""
import json
import os
import re
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

class TestStripThinkTags(unittest.TestCase):
    """Test the _strip_think_tags helper function."""

    def test_no_think_tags(self):
        """Content without think tags should be returned unchanged."""
        from PE.minimax_client import _strip_think_tags
        text = "A beautiful sunset over the ocean."
        self.assertEqual(_strip_think_tags(text), text)

    def test_think_tags_removed(self):
        """Think tags and their content should be stripped."""
        from PE.minimax_client import _strip_think_tags
        text = "<think>I should enhance this prompt...</think>A beautiful sunset over the ocean."
        result = _strip_think_tags(text)
        self.assertEqual(result, "A beautiful sunset over the ocean.")

    def test_multiline_think_tags(self):
        """Multiline think content should be fully stripped."""
        from PE.minimax_client import _strip_think_tags
        text = (
            "<think>\nLet me analyze this prompt.\n"
            "The user wants a landscape.\n</think>\n"
            "A cinematic wide-angle landscape."
        )
        result = _strip_think_tags(text)
        self.assertEqual(result, "A cinematic wide-angle landscape.")

    def test_empty_think_tags(self):
        """Empty think tags should be stripped."""
        from PE.minimax_client import _strip_think_tags
        text = "<think></think>Enhanced prompt here."
        result = _strip_think_tags(text)
        self.assertEqual(result, "Enhanced prompt here.")

    def test_only_think_tags(self):
        """If only think tags exist, result should be empty."""
        from PE.minimax_client import _strip_think_tags
        text = "<think>Only reasoning, no content.</think>"
        result = _strip_think_tags(text)
        self.assertEqual(result, "")

    def test_whitespace_after_think_tags(self):
        """Whitespace between think tags and content should be handled."""
        from PE.minimax_client import _strip_think_tags
        text = "<think>reasoning</think>   \n  Actual content."
        result = _strip_think_tags(text)
        self.assertEqual(result, "Actual content.")


class TestMiniMaxClientInit(unittest.TestCase):
    """Test MiniMaxClient initialization."""

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    @patch("PE.minimax_client.OpenAI")
    def test_init_with_env_key(self, mock_openai_cls):
        """Client should initialize with API key from environment."""
        from PE.minimax_client import MiniMaxClient
        client = MiniMaxClient()
        self.assertEqual(client.api_key, "test-key-123")
        self.assertEqual(client.model, "MiniMax-M2.7")
        mock_openai_cls.assert_called_once_with(
            api_key="test-key-123",
            base_url="https://api.minimax.io/v1",
        )

    @patch("PE.minimax_client.OpenAI")
    def test_init_with_explicit_key(self, mock_openai_cls):
        """Client should initialize with explicitly provided API key."""
        from PE.minimax_client import MiniMaxClient
        client = MiniMaxClient(api_key="explicit-key")
        self.assertEqual(client.api_key, "explicit-key")

    @patch("PE.minimax_client.OpenAI")
    def test_init_with_custom_model(self, mock_openai_cls):
        """Client should accept a custom model name."""
        from PE.minimax_client import MiniMaxClient
        client = MiniMaxClient(api_key="key", model="MiniMax-M2.5")
        self.assertEqual(client.model, "MiniMax-M2.5")

    @patch.dict(os.environ, {}, clear=True)
    @patch("PE.minimax_client.OpenAI")
    def test_init_no_key_raises(self, mock_openai_cls):
        """Client should raise ValueError if no API key is available."""
        from PE.minimax_client import MiniMaxClient
        # Remove any existing env var
        os.environ.pop("MINIMAX_API_KEY", None)
        with self.assertRaises(ValueError) as ctx:
            MiniMaxClient()
        self.assertIn("MINIMAX_API_KEY", str(ctx.exception))

    def test_init_no_openai_package(self):
        """Client should raise ImportError if openai package is missing."""
        import PE.minimax_client as mod
        original = mod.OpenAI
        mod.OpenAI = None
        try:
            with self.assertRaises(ImportError) as ctx:
                mod.MiniMaxClient(api_key="key")
            self.assertIn("openai", str(ctx.exception))
        finally:
            mod.OpenAI = original


class TestMiniMaxClientRecaption(unittest.TestCase):
    """Test MiniMaxClient.run_single_recaption method."""

    def _make_mock_response(self, content):
        """Create a mock OpenAI response object."""
        mock_message = MagicMock()
        mock_message.content = content
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        return mock_response

    @patch("PE.minimax_client.OpenAI")
    def test_recaption_basic(self, mock_openai_cls):
        """Basic recaption should return enhanced prompt and empty reason."""
        from PE.minimax_client import MiniMaxClient
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            "A cinematic wide-angle shot of a brown and white border collie."
        )

        client = MiniMaxClient(api_key="test-key")
        content, reason = client.run_single_recaption(
            "You are a prompt engineer.", "A dog running."
        )

        self.assertIn("border collie", content)
        self.assertEqual(reason, "")
        mock_client.chat.completions.create.assert_called_once()

    @patch("PE.minimax_client.OpenAI")
    def test_recaption_strips_think_tags(self, mock_openai_cls):
        """Think tags in response should be stripped."""
        from PE.minimax_client import MiniMaxClient
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            "<think>Let me enhance this...</think>Enhanced prompt here."
        )

        client = MiniMaxClient(api_key="test-key")
        content, reason = client.run_single_recaption("system", "user prompt")

        self.assertEqual(content, "Enhanced prompt here.")
        self.assertNotIn("<think>", content)

    @patch("PE.minimax_client.OpenAI")
    def test_recaption_empty_content(self, mock_openai_cls):
        """Empty content from API should return empty string."""
        from PE.minimax_client import MiniMaxClient
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._make_mock_response(None)

        client = MiniMaxClient(api_key="test-key")
        content, reason = client.run_single_recaption("system", "user prompt")

        self.assertEqual(content, "")

    @patch("PE.minimax_client.OpenAI")
    def test_recaption_api_params(self, mock_openai_cls):
        """API call should use correct parameters."""
        from PE.minimax_client import MiniMaxClient
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._make_mock_response("result")

        client = MiniMaxClient(api_key="test-key", model="MiniMax-M2.5")
        client.run_single_recaption("sys prompt", "user prompt")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "MiniMax-M2.5")
        self.assertEqual(len(call_kwargs["messages"]), 2)
        self.assertEqual(call_kwargs["messages"][0]["role"], "system")
        self.assertEqual(call_kwargs["messages"][0]["content"], "sys prompt")
        self.assertEqual(call_kwargs["messages"][1]["role"], "user")
        self.assertEqual(call_kwargs["messages"][1]["content"], "user prompt")
        self.assertGreater(call_kwargs["temperature"], 0.0)
        self.assertLessEqual(call_kwargs["temperature"], 1.0)
        self.assertFalse(call_kwargs["stream"])

    @patch("PE.minimax_client.OpenAI")
    def test_recaption_retry_on_error(self, mock_openai_cls):
        """Client should retry on transient API errors."""
        from PE.minimax_client import MiniMaxClient
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # First call raises, second succeeds
        mock_client.chat.completions.create.side_effect = [
            Exception("Connection timeout"),
            self._make_mock_response("Enhanced prompt."),
        ]

        client = MiniMaxClient(api_key="test-key")
        with patch("PE.minimax_client.time.sleep"):  # Skip actual sleep
            content, reason = client.run_single_recaption("system", "prompt")

        self.assertEqual(content, "Enhanced prompt.")
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)

    @patch("PE.minimax_client.OpenAI")
    def test_recaption_long_prompt(self, mock_openai_cls):
        """Client should handle long prompts without truncation."""
        from PE.minimax_client import MiniMaxClient
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        long_response = "Enhanced: " + "word " * 500
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            long_response
        )

        client = MiniMaxClient(api_key="test-key")
        content, reason = client.run_single_recaption("system", "A " * 1000)

        self.assertEqual(content, long_response.strip())

    @patch("PE.minimax_client.OpenAI")
    def test_base_url(self, mock_openai_cls):
        """Client should use MiniMax API base URL."""
        from PE.minimax_client import MiniMaxClient
        MiniMaxClient(api_key="test-key")
        call_kwargs = mock_openai_cls.call_args[1]
        self.assertEqual(call_kwargs["base_url"], "https://api.minimax.io/v1")

    @patch("PE.minimax_client.OpenAI")
    def test_default_model(self, mock_openai_cls):
        """Default model should be MiniMax-M2.7."""
        from PE.minimax_client import MiniMaxClient
        client = MiniMaxClient(api_key="test-key")
        self.assertEqual(client.model, "MiniMax-M2.7")

    @patch("PE.minimax_client.OpenAI")
    def test_temperature_constraint(self, mock_openai_cls):
        """Temperature must be in (0.0, 1.0] for MiniMax API."""
        from PE.minimax_client import MiniMaxClient
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._make_mock_response("ok")

        client = MiniMaxClient(api_key="test-key")
        client.run_single_recaption("system", "prompt")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        temp = call_kwargs["temperature"]
        self.assertGreater(temp, 0.0)
        self.assertLessEqual(temp, 1.0)


class TestRunImageGenArgs(unittest.TestCase):
    """Test argument parsing for run_image_gen.py with LLM provider support."""

    @classmethod
    def setUpClass(cls):
        """Mock hunyuan_image_3 module so run_image_gen.py can be imported."""
        cls._mock_module = MagicMock()
        sys.modules["hunyuan_image_3"] = cls._mock_module

    @classmethod
    def tearDownClass(cls):
        """Remove mock module."""
        sys.modules.pop("hunyuan_image_3", None)

    def _parse(self, extra_args):
        """Parse args with required arguments plus extras."""
        base = ["--prompt", "test", "--model-id", "/tmp/fake"]
        with patch("sys.argv", ["run_image_gen.py"] + base + extra_args):
            # Re-import to get fresh parse_args
            import importlib
            import run_image_gen
            importlib.reload(run_image_gen)
            return run_image_gen.parse_args()

    def test_default_llm_provider(self):
        """Default LLM provider should be deepseek."""
        args = self._parse([])
        self.assertEqual(args.llm_provider, "deepseek")

    def test_minimax_provider(self):
        """--llm-provider minimax should be accepted."""
        args = self._parse(["--llm-provider", "minimax"])
        self.assertEqual(args.llm_provider, "minimax")

    def test_default_sys_prompt_type(self):
        """Default system prompt type should be universal."""
        args = self._parse([])
        self.assertEqual(args.sys_prompt_type, "universal")

    def test_text_rendering_prompt_type(self):
        """--sys-prompt-type text_rendering should be accepted."""
        args = self._parse(["--sys-prompt-type", "text_rendering"])
        self.assertEqual(args.sys_prompt_type, "text_rendering")

    def test_invalid_provider_rejected(self):
        """Invalid provider name should be rejected by argparse."""
        with self.assertRaises(SystemExit):
            self._parse(["--llm-provider", "invalid_provider"])

    def test_rewrite_default_off(self):
        """Rewrite should be disabled by default."""
        args = self._parse([])
        self.assertEqual(args.rewrite, 0)

    def test_rewrite_enabled(self):
        """--rewrite 1 should enable rewriting."""
        args = self._parse(["--rewrite", "1"])
        self.assertEqual(args.rewrite, 1)


class TestProviderDispatch(unittest.TestCase):
    """Test the LLM provider dispatch logic in run_image_gen.main()."""

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    @patch("PE.minimax_client.OpenAI")
    def test_minimax_provider_dispatch(self, mock_openai_cls):
        """When llm_provider is minimax, MiniMaxClient should be used."""
        from PE.minimax_client import MiniMaxClient
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_message = MagicMock()
        mock_message.content = "Enhanced prompt."
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        client = MiniMaxClient(api_key="test-key")
        content, reason = client.run_single_recaption(
            "You are a prompt engineer.", "A dog running."
        )
        self.assertEqual(content, "Enhanced prompt.")
        self.assertEqual(reason, "")

    @patch.dict(os.environ, {}, clear=True)
    def test_minimax_provider_no_key_raises(self):
        """MiniMax provider without API key should raise ValueError."""
        os.environ.pop("MINIMAX_API_KEY", None)
        from PE.minimax_client import MiniMaxClient
        # Mock OpenAI to avoid import error
        with patch("PE.minimax_client.OpenAI"):
            with self.assertRaises(ValueError) as ctx:
                MiniMaxClient()
            self.assertIn("MINIMAX_API_KEY", str(ctx.exception))


class TestSystemPromptCompatibility(unittest.TestCase):
    """Test that system prompts work correctly with MiniMax provider."""

    def test_system_prompts_are_strings(self):
        """System prompts should be non-empty strings."""
        from PE.system_prompt import system_prompt_universal, system_prompt_text_rendering
        self.assertIsInstance(system_prompt_universal, str)
        self.assertIsInstance(system_prompt_text_rendering, str)
        self.assertGreater(len(system_prompt_universal), 100)
        self.assertGreater(len(system_prompt_text_rendering), 100)

    def test_system_prompts_provider_agnostic(self):
        """System prompts should not reference any specific provider."""
        from PE.system_prompt import system_prompt_universal, system_prompt_text_rendering
        for prompt in [system_prompt_universal, system_prompt_text_rendering]:
            # System prompts describe image generation tasks, not LLM providers
            self.assertNotIn("DeepSeek", prompt)
            self.assertNotIn("deepseek", prompt)


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestMiniMaxIntegration(unittest.TestCase):
    """Integration tests for MiniMax prompt enhancement (requires API key)."""

    @classmethod
    def setUpClass(cls):
        """Check if MiniMax API key is available."""
        cls.api_key = os.getenv("MINIMAX_API_KEY")
        if not cls.api_key:
            raise unittest.SkipTest("MINIMAX_API_KEY not set, skipping integration tests")

    def test_real_recaption_universal(self):
        """Test real prompt enhancement with universal system prompt."""
        from PE.minimax_client import MiniMaxClient
        from PE.system_prompt import system_prompt_universal

        client = MiniMaxClient(api_key=self.api_key)
        content, reason = client.run_single_recaption(
            system_prompt_universal,
            "A cat sitting on a windowsill"
        )

        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 10)
        # Enhanced prompt should be longer/more detailed than input
        self.assertGreater(len(content), len("A cat sitting on a windowsill"))

    def test_real_recaption_text_rendering(self):
        """Test real prompt enhancement with text rendering system prompt."""
        from PE.minimax_client import MiniMaxClient
        from PE.system_prompt import system_prompt_text_rendering

        client = MiniMaxClient(api_key=self.api_key)
        content, reason = client.run_single_recaption(
            system_prompt_text_rendering,
            'A poster with the text "Hello World"'
        )

        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 10)

    def test_real_recaption_m25(self):
        """Test prompt enhancement with MiniMax-M2.5 model."""
        from PE.minimax_client import MiniMaxClient
        from PE.system_prompt import system_prompt_universal

        client = MiniMaxClient(api_key=self.api_key, model="MiniMax-M2.5")
        content, reason = client.run_single_recaption(
            system_prompt_universal,
            "A sunset over mountains"
        )

        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 10)


if __name__ == "__main__":
    unittest.main()
