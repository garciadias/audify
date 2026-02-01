"""Tests for async API functionality."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from audify.utils.async_utils import AsyncBatcher, gather_with_limit, run_async


class TestAsyncBatcher:
    """Tests for AsyncBatcher class."""

    @pytest.mark.asyncio
    async def test_run_batch_basic(self):
        """Test basic batch execution."""

        async def simple_coro(x):
            return x * 2

        batcher = AsyncBatcher(max_concurrent=5)
        coros = [simple_coro(i) for i in range(10)]
        results = await batcher.run_batch(coros)

        assert results == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

    @pytest.mark.asyncio
    async def test_run_batch_with_exceptions(self):
        """Test batch execution handles exceptions."""

        async def maybe_fail(x):
            if x == 5:
                raise ValueError("Failed on 5")
            return x

        batcher = AsyncBatcher(max_concurrent=3)
        coros = [maybe_fail(i) for i in range(10)]
        results = await batcher.run_batch(coros)

        # Check that exception is returned, not raised
        assert isinstance(results[5], ValueError)
        assert results[0] == 0
        assert results[9] == 9

    @pytest.mark.asyncio
    async def test_run_batch_ordered(self):
        """Test ordered batch execution."""

        async def delayed(x):
            await asyncio.sleep(0.01 * (10 - x))  # Longer delay for smaller x
            return x

        batcher = AsyncBatcher(max_concurrent=10)
        coros = [delayed(i) for i in range(5)]
        results = await batcher.run_batch_ordered(coros, return_exceptions=False)

        # Results should be in original order despite varying delays
        assert results == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """Test that concurrency limit is respected."""
        concurrent_count = 0
        max_concurrent_seen = 0

        async def track_concurrency():
            nonlocal concurrent_count, max_concurrent_seen
            concurrent_count += 1
            max_concurrent_seen = max(max_concurrent_seen, concurrent_count)
            await asyncio.sleep(0.05)
            concurrent_count -= 1
            return True

        batcher = AsyncBatcher(max_concurrent=3)
        coros = [track_concurrency() for _ in range(10)]
        await batcher.run_batch(coros)

        assert max_concurrent_seen <= 3


class TestRunAsync:
    """Tests for run_async helper."""

    def test_run_async_simple(self):
        """Test running async function from sync context."""

        async def async_func():
            return "hello"

        result = run_async(async_func())
        assert result == "hello"

    def test_run_async_with_args(self):
        """Test running async function with arguments."""

        async def async_add(a, b):
            return a + b

        result = run_async(async_add(3, 4))
        assert result == 7


class TestGatherWithLimit:
    """Tests for gather_with_limit helper."""

    @pytest.mark.asyncio
    async def test_gather_basic(self):
        """Test gather_with_limit basic functionality."""

        async def double(x):
            return x * 2

        coros = [double(i) for i in range(5)]
        results = await gather_with_limit(coros, limit=3)

        assert results == [0, 2, 4, 6, 8]


class TestTTSConfigAsync:
    """Tests for async TTS config methods."""

    @pytest.mark.asyncio
    async def test_kokoro_synthesize_async(self):
        """Test KokoroTTSConfig.synthesize_async."""
        import tempfile

        from audify.utils.api_config import KokoroTTSConfig

        config = KokoroTTSConfig(voice="af_bella", language="en")

        with patch("audify.utils.api_config.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.content = b"audio_data"
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                output_path = Path(f.name)

            try:
                result = await config.synthesize_async("Hello", output_path)
                assert result is True
                assert output_path.read_bytes() == b"audio_data"
            finally:
                output_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_openai_synthesize_async(self):
        """Test OpenAITTSConfig.synthesize_async."""
        import tempfile

        from audify.utils.api_config import OpenAITTSConfig

        config = OpenAITTSConfig(api_key="test-key", voice="alloy")

        with patch("audify.utils.api_config.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.content = b"openai_audio"
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                output_path = Path(f.name)

            try:
                result = await config.synthesize_async("Hello OpenAI", output_path)
                assert result is True
                assert output_path.read_bytes() == b"openai_audio"
            finally:
                output_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_synthesize_batch_async(self):
        """Test TTSAPIConfig.synthesize_batch_async."""
        from audify.utils.api_config import KokoroTTSConfig

        config = KokoroTTSConfig(voice="af_bella")

        # Mock synthesize_async
        async def mock_synthesize(text, path):
            return True

        with patch.object(config, "synthesize_async", side_effect=mock_synthesize):
            texts = ["Hello", "World", "Test"]
            paths = [Path(f"/tmp/test_{i}.wav") for i in range(3)]

            results = await config.synthesize_batch_async(
                texts=texts, output_paths=paths, max_concurrent=2
            )

            assert results == [True, True, True]


class TestLLMConfigAsync:
    """Tests for async LLM config methods."""

    @pytest.mark.asyncio
    async def test_commercial_api_generate_async(self):
        """Test CommercialAPIConfig.generate_async."""
        from audify.utils.api_config import CommercialAPIConfig

        with patch("audify.utils.api_config.acompletion") as mock_acompletion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Generated text"
            mock_acompletion.return_value = mock_response

            config = CommercialAPIConfig(model="deepseek/deepseek-chat")
            result = await config.generate_async(
                system_prompt="You are helpful",
                user_prompt="Hello",
            )

            assert result == "Generated text"
            mock_acompletion.assert_called_once()

    @pytest.mark.asyncio
    async def test_ollama_api_generate_async(self):
        """Test OllamaAPIConfig.generate_async."""
        from audify.utils.api_config import OllamaAPIConfig

        with patch("audify.utils.api_config.acompletion") as mock_acompletion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Ollama response"
            mock_acompletion.return_value = mock_response

            config = OllamaAPIConfig(model="llama3.2:3b")
            result = await config.generate_async(user_prompt="Test prompt")

            assert result == "Ollama response"
            mock_acompletion.assert_called_once()


class TestTranslateAsync:
    """Tests for async translation."""

    @pytest.mark.asyncio
    async def test_translate_sentence_async(self):
        """Test translate_sentence_async function."""
        from audify.translate import translate_sentence_async

        with patch(
            "audify.utils.api_config.OllamaTranslationConfig.translate_async"
        ) as mock_translate:
            mock_translate.return_value = "Hola"

            result = await translate_sentence_async(
                "Hello", src_lang="en", tgt_lang="es"
            )

            assert result == "Hola"

    @pytest.mark.asyncio
    async def test_translate_same_language(self):
        """Test translation returns original when src == tgt."""
        from audify.translate import translate_sentence_async

        result = await translate_sentence_async("Hello", src_lang="en", tgt_lang="en")

        assert result == "Hello"


class TestLLMClientAsync:
    """Tests for async LLMClient."""

    @pytest.mark.asyncio
    async def test_generate_audiobook_script_async(self):
        """Test LLMClient.generate_audiobook_script_async."""
        from audify.audiobook_creator import LLMClient

        with patch(
            "audify.utils.api_config.OllamaAPIConfig.generate_async"
        ) as mock_generate:
            mock_generate.return_value = "Audiobook script content"

            client = LLMClient(model="llama3.2:3b")
            result = await client.generate_audiobook_script_async(
                "Chapter text here", language="en"
            )

            assert "script content" in result.lower() or "audiobook" in result.lower()
