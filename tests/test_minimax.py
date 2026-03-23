"""Unit tests for MiniMax engine and routes."""

import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---- Mock heavy modules before any project import ----
# Create a mock api.models module to avoid loading torch/transformers
_mock_models = types.ModuleType("api.models")
_mock_models.LLM_ENGINE = MagicMock()
_mock_models.app = MagicMock()
_mock_models.EMBEDDING_MODEL = None
_mock_models.RERANK_MODEL = None
sys.modules["api.models"] = _mock_models


class TestMiniMaxEngine(unittest.TestCase):
    """Test MiniMaxEngine class."""

    def test_clamp_temperature_within_range(self):
        """Temperature within [0, 1] should remain unchanged."""
        from api.engine.minimax import MiniMaxEngine
        self.assertEqual(MiniMaxEngine.clamp_temperature(0.5), 0.5)
        self.assertEqual(MiniMaxEngine.clamp_temperature(0.0), 0.0)
        self.assertEqual(MiniMaxEngine.clamp_temperature(1.0), 1.0)

    def test_clamp_temperature_above_range(self):
        """Temperature above 1 should be clamped to 1."""
        from api.engine.minimax import MiniMaxEngine
        self.assertEqual(MiniMaxEngine.clamp_temperature(1.5), 1.0)
        self.assertEqual(MiniMaxEngine.clamp_temperature(2.0), 1.0)

    def test_clamp_temperature_below_range(self):
        """Temperature below 0 should be clamped to 0."""
        from api.engine.minimax import MiniMaxEngine
        self.assertEqual(MiniMaxEngine.clamp_temperature(-0.5), 0.0)

    def test_clamp_temperature_none(self):
        """None temperature should return None."""
        from api.engine.minimax import MiniMaxEngine
        self.assertIsNone(MiniMaxEngine.clamp_temperature(None))

    def test_available_models(self):
        """Should return list of available MiniMax models."""
        from api.engine.minimax import MiniMaxEngine
        models = MiniMaxEngine.available_models()
        self.assertIn("MiniMax-M2.7", models)
        self.assertIn("MiniMax-M2.7-highspeed", models)
        self.assertIn("MiniMax-M2.5", models)
        self.assertIn("MiniMax-M2.5-highspeed", models)
        self.assertEqual(len(models), 4)

    def test_minimax_models_context_lengths(self):
        """Verify correct context lengths for MiniMax models."""
        from api.engine.minimax import MINIMAX_MODELS
        self.assertEqual(MINIMAX_MODELS["MiniMax-M2.7"], 1048576)
        self.assertEqual(MINIMAX_MODELS["MiniMax-M2.7-highspeed"], 1048576)
        self.assertEqual(MINIMAX_MODELS["MiniMax-M2.5"], 245760)
        self.assertEqual(MINIMAX_MODELS["MiniMax-M2.5-highspeed"], 204800)

    @patch("api.engine.minimax.OpenAI")
    def test_engine_initialization(self, mock_openai_cls):
        """Engine should initialize OpenAI client with correct params."""
        from api.engine.minimax import MiniMaxEngine
        engine = MiniMaxEngine(
            api_key="test-key",
            model_name="MiniMax-M2.7",
            api_base="https://api.minimax.io/v1",
        )
        mock_openai_cls.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.minimax.io/v1",
        )
        self.assertEqual(engine.model_name, "MiniMax-M2.7")

    @patch("api.engine.minimax.OpenAI")
    def test_engine_default_api_base(self, mock_openai_cls):
        """Engine should use default MiniMax API base URL."""
        from api.engine.minimax import MiniMaxEngine
        engine = MiniMaxEngine(api_key="test-key", model_name="MiniMax-M2.7")
        mock_openai_cls.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.minimax.io/v1",
        )

    @patch("api.engine.minimax.OpenAI")
    def test_engine_custom_api_base(self, mock_openai_cls):
        """Engine should respect custom API base URL."""
        from api.engine.minimax import MiniMaxEngine
        engine = MiniMaxEngine(
            api_key="test-key",
            model_name="MiniMax-M2.5",
            api_base="https://custom.api.example.com/v1",
        )
        mock_openai_cls.assert_called_once_with(
            api_key="test-key",
            base_url="https://custom.api.example.com/v1",
        )


class TestMiniMaxConfig(unittest.TestCase):
    """Test MiniMax configuration settings."""

    def test_minimax_settings_has_fields(self):
        """MiniMaxSettings should have expected fields."""
        from api.config import MiniMaxSettings
        settings = MiniMaxSettings()
        self.assertTrue(hasattr(settings, "minimax_api_key"))
        self.assertTrue(hasattr(settings, "minimax_api_base"))

    def test_minimax_settings_default_api_base(self):
        """Default API base should be MiniMax endpoint."""
        from api.config import MiniMaxSettings
        settings = MiniMaxSettings()
        self.assertIn("minimax", settings.minimax_api_base)

    def test_minimax_settings_with_explicit_values(self):
        """MiniMaxSettings should accept explicit values."""
        from api.config import MiniMaxSettings
        settings = MiniMaxSettings(
            minimax_api_key="explicit-key",
            minimax_api_base="https://custom.example.com/v1",
        )
        self.assertEqual(settings.minimax_api_key, "explicit-key")
        self.assertEqual(settings.minimax_api_base, "https://custom.example.com/v1")


class TestMiniMaxChatRoute(unittest.TestCase):
    """Test MiniMax chat route helper functions."""

    @patch("api.engine.minimax.OpenAI")
    def test_build_chat_kwargs_basic(self, mock_openai_cls):
        """Should build correct kwargs for basic chat request."""
        from api.engine.minimax import MiniMaxEngine
        from api.minimax_routes.chat import _build_chat_kwargs
        from api.protocol import ChatCompletionCreateParams

        engine = MiniMaxEngine(api_key="test", model_name="MiniMax-M2.7")
        request = ChatCompletionCreateParams(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.7,
            max_tokens=100,
        )

        kwargs = _build_chat_kwargs(request, engine)
        self.assertEqual(kwargs["model"], "MiniMax-M2.7")
        self.assertEqual(kwargs["messages"], [{"role": "user", "content": "hello"}])
        self.assertEqual(kwargs["temperature"], 0.7)
        self.assertEqual(kwargs["max_tokens"], 100)
        self.assertFalse(kwargs["stream"])

    @patch("api.engine.minimax.OpenAI")
    def test_build_chat_kwargs_temperature_clamping(self, mock_openai_cls):
        """Should clamp temperature to [0, 1] range."""
        from api.engine.minimax import MiniMaxEngine
        from api.minimax_routes.chat import _build_chat_kwargs
        from api.protocol import ChatCompletionCreateParams

        engine = MiniMaxEngine(api_key="test", model_name="MiniMax-M2.7")
        request = ChatCompletionCreateParams(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "hello"}],
            temperature=1.8,
        )

        kwargs = _build_chat_kwargs(request, engine)
        self.assertEqual(kwargs["temperature"], 1.0)

    @patch("api.engine.minimax.OpenAI")
    def test_build_chat_kwargs_with_tools(self, mock_openai_cls):
        """Should include tools when provided."""
        from api.engine.minimax import MiniMaxEngine
        from api.minimax_routes.chat import _build_chat_kwargs
        from api.protocol import ChatCompletionCreateParams

        engine = MiniMaxEngine(api_key="test", model_name="MiniMax-M2.7")
        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        request = ChatCompletionCreateParams(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "hello"}],
            tools=tools,
        )

        kwargs = _build_chat_kwargs(request, engine)
        self.assertEqual(kwargs["tools"], tools)

    @patch("api.engine.minimax.OpenAI")
    def test_build_chat_kwargs_stream(self, mock_openai_cls):
        """Should set stream flag correctly."""
        from api.engine.minimax import MiniMaxEngine
        from api.minimax_routes.chat import _build_chat_kwargs
        from api.protocol import ChatCompletionCreateParams

        engine = MiniMaxEngine(api_key="test", model_name="MiniMax-M2.7")
        request = ChatCompletionCreateParams(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )

        kwargs = _build_chat_kwargs(request, engine)
        self.assertTrue(kwargs["stream"])

    @patch("api.engine.minimax.OpenAI")
    def test_build_chat_kwargs_with_stop(self, mock_openai_cls):
        """Should include stop sequences when provided."""
        from api.engine.minimax import MiniMaxEngine
        from api.minimax_routes.chat import _build_chat_kwargs
        from api.protocol import ChatCompletionCreateParams

        engine = MiniMaxEngine(api_key="test", model_name="MiniMax-M2.7")
        request = ChatCompletionCreateParams(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "hello"}],
            stop=["###"],
        )

        kwargs = _build_chat_kwargs(request, engine)
        self.assertEqual(kwargs["stop"], ["###"])

    @patch("api.engine.minimax.OpenAI")
    def test_build_chat_kwargs_zero_temperature(self, mock_openai_cls):
        """Temperature=0 should be accepted (not clamped away)."""
        from api.engine.minimax import MiniMaxEngine
        from api.minimax_routes.chat import _build_chat_kwargs
        from api.protocol import ChatCompletionCreateParams

        engine = MiniMaxEngine(api_key="test", model_name="MiniMax-M2.7")
        request = ChatCompletionCreateParams(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.0,
        )

        kwargs = _build_chat_kwargs(request, engine)
        self.assertEqual(kwargs["temperature"], 0.0)

    @patch("api.engine.minimax.OpenAI")
    def test_build_chat_kwargs_no_optional_fields(self, mock_openai_cls):
        """Should not include optional fields when not set."""
        from api.engine.minimax import MiniMaxEngine
        from api.minimax_routes.chat import _build_chat_kwargs
        from api.protocol import ChatCompletionCreateParams

        engine = MiniMaxEngine(api_key="test", model_name="MiniMax-M2.7")
        request = ChatCompletionCreateParams(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "hello"}],
        )

        kwargs = _build_chat_kwargs(request, engine)
        self.assertNotIn("tools", kwargs)
        self.assertNotIn("tool_choice", kwargs)
        self.assertNotIn("response_format", kwargs)

    @patch("api.engine.minimax.OpenAI")
    def test_build_chat_kwargs_with_response_format(self, mock_openai_cls):
        """Should include response_format when set."""
        from api.engine.minimax import MiniMaxEngine
        from api.minimax_routes.chat import _build_chat_kwargs
        from api.protocol import ChatCompletionCreateParams

        engine = MiniMaxEngine(api_key="test", model_name="MiniMax-M2.7")
        request = ChatCompletionCreateParams(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "hello"}],
            response_format={"type": "json_object"},
        )

        kwargs = _build_chat_kwargs(request, engine)
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})


class TestMiniMaxModuleImports(unittest.TestCase):
    """Test that minimax modules import correctly."""

    def test_import_engine(self):
        """Should be able to import MiniMaxEngine."""
        from api.engine.minimax import MiniMaxEngine, MINIMAX_MODELS
        self.assertTrue(callable(MiniMaxEngine))
        self.assertIsInstance(MINIMAX_MODELS, dict)

    def test_import_config(self):
        """Should be able to import MiniMaxSettings."""
        from api.config import MiniMaxSettings
        self.assertTrue(callable(MiniMaxSettings))

    def test_import_chat_route(self):
        """Should be able to import chat route components."""
        from api.minimax_routes.chat import chat_router, _build_chat_kwargs
        self.assertIsNotNone(chat_router)
        self.assertTrue(callable(_build_chat_kwargs))

    def test_import_completion_route(self):
        """Should be able to import completion route module."""
        from api.minimax_routes.completion import completion_router
        self.assertIsNotNone(completion_router)

    def test_import_minimax_routes_init(self):
        """Should be able to import from minimax_routes package."""
        from api.minimax_routes import chat_router, completion_router
        self.assertIsNotNone(chat_router)
        self.assertIsNotNone(completion_router)


class TestMiniMaxTemperatureEdgeCases(unittest.TestCase):
    """Test temperature clamping edge cases."""

    def test_boundary_values(self):
        """Test exact boundary values."""
        from api.engine.minimax import MiniMaxEngine
        self.assertEqual(MiniMaxEngine.clamp_temperature(0.0), 0.0)
        self.assertEqual(MiniMaxEngine.clamp_temperature(1.0), 1.0)

    def test_just_over_boundary(self):
        """Test values just over the boundary."""
        from api.engine.minimax import MiniMaxEngine
        self.assertEqual(MiniMaxEngine.clamp_temperature(1.001), 1.0)

    def test_just_under_boundary(self):
        """Test values just under the boundary."""
        from api.engine.minimax import MiniMaxEngine
        self.assertEqual(MiniMaxEngine.clamp_temperature(-0.001), 0.0)

    def test_midrange_value(self):
        """Test midrange value passes through."""
        from api.engine.minimax import MiniMaxEngine
        self.assertAlmostEqual(MiniMaxEngine.clamp_temperature(0.42), 0.42)


if __name__ == "__main__":
    unittest.main()
