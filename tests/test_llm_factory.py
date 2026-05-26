"""Tests for llm_factory — _CachedLLM wrapper and build_llm cache integration."""
from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

import pytest

from earnings_agents.llm_factory import _CachedLLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inner_llm(response: str = "cached-response") -> MagicMock:
    m = MagicMock()
    m.invoke.return_value = response
    return m


# ---------------------------------------------------------------------------
# _CachedLLM unit tests
# ---------------------------------------------------------------------------

class TestCachedLLM:
    def test_cache_miss_calls_inner_llm(self, tmp_path):
        inner = _make_inner_llm("hello")
        cached = _CachedLLM(inner, "ollama:test-model", str(tmp_path))
        result = cached.invoke("some prompt")
        assert result == "hello"
        inner.invoke.assert_called_once_with("some prompt")

    def test_cache_hit_skips_inner_llm(self, tmp_path):
        inner = _make_inner_llm("hello")
        cached = _CachedLLM(inner, "ollama:test-model", str(tmp_path))
        # Prime the cache
        cached.invoke("some prompt")
        inner.invoke.reset_mock()
        # Second call must be served from cache
        result = cached.invoke("some prompt")
        assert result == "hello"
        inner.invoke.assert_not_called()

    def test_different_prompts_get_different_entries(self, tmp_path):
        inner = MagicMock()
        inner.invoke.side_effect = ["resp-A", "resp-B"]
        cached = _CachedLLM(inner, "ollama:test-model", str(tmp_path))
        assert cached.invoke("prompt-A") == "resp-A"
        assert cached.invoke("prompt-B") == "resp-B"
        assert inner.invoke.call_count == 2

    def test_same_prompt_different_model_tag_is_cache_miss(self, tmp_path):
        """Different model tags produce different cache keys."""
        inner1 = _make_inner_llm("from-model-1")
        inner2 = _make_inner_llm("from-model-2")
        cached1 = _CachedLLM(inner1, "ollama:model-1", str(tmp_path))
        cached2 = _CachedLLM(inner2, "groq:model-2", str(tmp_path))
        r1 = cached1.invoke("same prompt")
        r2 = cached2.invoke("same prompt")
        assert r1 == "from-model-1"
        assert r2 == "from-model-2"
        inner1.invoke.assert_called_once()
        inner2.invoke.assert_called_once()

    def test_cache_key_is_sha256_of_model_tag_and_prompt(self, tmp_path):
        inner = _make_inner_llm()
        cached = _CachedLLM(inner, "ollama:my-model", str(tmp_path))
        prompt = "test prompt"
        expected_key = hashlib.sha256(f"ollama:my-model\n{prompt}".encode()).hexdigest()
        assert cached._cache_key(prompt) == expected_key

    def test_cache_persists_across_instances(self, tmp_path):
        """Two _CachedLLM instances sharing the same dir share cached responses."""
        inner1 = _make_inner_llm("original-response")
        cached1 = _CachedLLM(inner1, "ollama:m", str(tmp_path))
        cached1.invoke("hello")

        # Second instance — inner LLM should never be called (cache hit)
        inner2 = _make_inner_llm("should-not-be-returned")
        cached2 = _CachedLLM(inner2, "ollama:m", str(tmp_path))
        result = cached2.invoke("hello")
        assert result == "original-response"
        inner2.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# build_llm cache integration
# ---------------------------------------------------------------------------

class TestBuildLLMCacheIntegration:
    @patch("earnings_agents.llm_factory.LLM_CACHE_ENABLED", True)
    @patch("earnings_agents.llm_factory.LLM_PROVIDER", "ollama")
    def test_cache_enabled_wraps_ollama_in_cached_llm(self, tmp_path):
        with (
            patch("earnings_agents.llm_factory.LLM_CACHE_DIR", str(tmp_path)),
            patch("langchain_ollama.OllamaLLM", MagicMock()),
        ):
            from earnings_agents.llm_factory import build_llm
            llm = build_llm()
            assert isinstance(llm, _CachedLLM)

    @patch("earnings_agents.llm_factory.LLM_CACHE_ENABLED", False)
    @patch("earnings_agents.llm_factory.LLM_PROVIDER", "ollama")
    def test_cache_disabled_returns_raw_llm(self):
        with patch("langchain_ollama.OllamaLLM", MagicMock()) as mock_cls:
            from earnings_agents.llm_factory import build_llm
            llm = build_llm()
            assert not isinstance(llm, _CachedLLM)
