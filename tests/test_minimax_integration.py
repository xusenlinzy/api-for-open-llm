"""
Integration tests for MiniMax engine.

These tests call the real MiniMax API and require:
  - MINIMAX_API_KEY environment variable to be set

Run with:
  MINIMAX_API_KEY=your-key python -m pytest tests/test_minimax_integration.py -v

To skip these tests in CI, set SKIP_INTEGRATION_TESTS=true.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SKIP_REASON = "MINIMAX_API_KEY not set or SKIP_INTEGRATION_TESTS=true"
SHOULD_SKIP = (
    not os.environ.get("MINIMAX_API_KEY")
    or os.environ.get("SKIP_INTEGRATION_TESTS", "").lower() == "true"
)


@unittest.skipIf(SHOULD_SKIP, SKIP_REASON)
class TestMiniMaxIntegration(unittest.TestCase):
    """Integration tests that call the real MiniMax API."""

    @classmethod
    def setUpClass(cls):
        from api.engine.minimax import MiniMaxEngine
        cls.engine = MiniMaxEngine(
            api_key=os.environ["MINIMAX_API_KEY"],
            model_name="MiniMax-M2.5-highspeed",
        )

    def test_chat_completion(self):
        """Test non-streaming chat completion."""
        response = self.engine.client.chat.completions.create(
            model=self.engine.model_name,
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            temperature=0.0,
            max_tokens=10,
        )
        self.assertIsNotNone(response)
        self.assertTrue(len(response.choices) > 0)
        content = response.choices[0].message.content.lower()
        self.assertIn("hello", content)

    def test_chat_completion_streaming(self):
        """Test streaming chat completion."""
        stream = self.engine.client.chat.completions.create(
            model=self.engine.model_name,
            messages=[{"role": "user", "content": "Say 'world' and nothing else."}],
            temperature=0.0,
            max_tokens=256,
            stream=True,
        )
        chunks = list(stream)
        self.assertTrue(len(chunks) > 0)

        # Collect all content from chunks - verify streaming works
        full_content = ""
        for chunk in chunks:
            if chunk.choices and chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content
        # Verify we got some content back (model may include thinking tags)
        self.assertTrue(len(full_content) > 0)

    def test_temperature_clamping(self):
        """Test that clamped temperature produces valid results."""
        # Temperature 1.8 should be clamped to 1.0 for MiniMax
        clamped = self.engine.clamp_temperature(1.8)
        self.assertEqual(clamped, 1.0)

        response = self.engine.client.chat.completions.create(
            model=self.engine.model_name,
            messages=[{"role": "user", "content": "Say 'test' and nothing else."}],
            temperature=clamped,
            max_tokens=10,
        )
        self.assertIsNotNone(response.choices[0].message.content)


if __name__ == "__main__":
    unittest.main()
