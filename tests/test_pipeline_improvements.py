"""Tests for pipeline improvements: modes, preflight, validation, resumability."""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from audify.audiobook_creator import AudiobookCreator, _env_flag


# ---------------------------------------------------------------------------
# _env_flag
# ---------------------------------------------------------------------------
class TestEnvFlag:
    pass

# ---------------------------------------------------------------------------
# Helper to build a minimal AudiobookCreator without __init__
# ---------------------------------------------------------------------------
def _make_creator(base_path=None, **overrides):
    """Build an AudiobookCreator via __new__ with just enough state for testing.

    When *base_path* is None the path attributes are set to placeholders that
    should be overridden by each test that actually touches the filesystem.
    """
    creator = AudiobookCreator.__new__(AudiobookCreator)
    creator.chapter_titles = []
    creator.task_name = "audiobook"
    creator.tts_provider = "kokoro"
    creator.save_text = True
    creator.confirm = False
    creator.progress = MagicMock()
    base = base_path or Path("test_placeholder")
    creator.scripts_path = base / "scripts"
    creator.episodes_path = base / "episodes"
    creator.audiobook_path = base
    creator.file_name = Path("test")
    for k, v in overrides.items():
        setattr(creator, k, v)
    return creator


# ---------------------------------------------------------------------------
# _verify_tts_provider_available
# ---------------------------------------------------------------------------
class TestVerifyTTSProviderAvailable:
    def test_skip_when_env_flag_set(self, monkeypatch):
        monkeypatch.setenv("AUDIFY_SKIP_TTS_PREFLIGHT", "1")
        creator = _make_creator()
        # Should not raise even without _tts_config
        creator._verify_tts_provider_available()

    def test_warns_when_unavailable_and_non_audiobook_task(self, monkeypatch):
        monkeypatch.delenv("AUDIFY_SKIP_TTS_PREFLIGHT", raising=False)
        monkeypatch.delenv("AUDIFY_STRICT_TTS_PREFLIGHT", raising=False)

        mock_config = MagicMock()
        mock_config.is_available.return_value = False
        mock_config.provider_name = "kokoro"
        mock_config.base_url = "http://localhost:8887"

        creator = _make_creator(task_name="direct")
        creator._tts_config = mock_config
        creator._get_tts_config = Mock(return_value=mock_config)

        # Should not raise for non-audiobook task
        creator._verify_tts_provider_available()


    def test_passes_when_available(self, monkeypatch):
        monkeypatch.delenv("AUDIFY_SKIP_TTS_PREFLIGHT", raising=False)

        mock_config = MagicMock()
        mock_config.is_available.return_value = True
        mock_config.provider_name = "kokoro"

        creator = _make_creator()
        creator._tts_config = mock_config
        creator._get_tts_config = Mock(return_value=mock_config)

        creator._verify_tts_provider_available()

# ---------------------------------------------------------------------------
# _save_chapter_titles / _load_chapter_titles
# ---------------------------------------------------------------------------
class TestChapterTitlesPersistence:


    def test_load_corrupt_json(self, tmp_path):
        (tmp_path / "chapter_titles.json").write_text("{bad json", encoding="utf-8")
        creator = _make_creator(scripts_path=tmp_path)
        assert creator._load_chapter_titles() == []

# ---------------------------------------------------------------------------
# _validate_chapters
# ---------------------------------------------------------------------------
class TestValidateChapters:
    def test_empty_list(self):
        creator = _make_creator()
        assert creator._validate_chapters([]) == []

# ---------------------------------------------------------------------------
# _synthesize_from_existing_scripts
# ---------------------------------------------------------------------------
class TestSynthesizeFromExistingScripts:
    def test_no_scripts_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AUDIFY_SKIP_TTS_PREFLIGHT", raising=False)
        creator = _make_creator(scripts_path=tmp_path, mode="synthesize")
        # No _tts_config → preflight skipped
        result = creator._synthesize_from_existing_scripts()
        assert result == []

    def test_reads_scripts_and_synthesizes(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AUDIFY_SKIP_TTS_PREFLIGHT", raising=False)
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        episodes = tmp_path / "episodes"
        episodes.mkdir()

        (scripts / "episode_001_script.txt").write_text("Hello world")
        (scripts / "episode_002_script.txt").write_text("Goodbye world")

        titles_path = scripts / "chapter_titles.json"
        titles_path.write_text(json.dumps(["Ch1", "Ch2"]))

        creator = _make_creator(
            scripts_path=scripts,
            episodes_path=episodes,
            audiobook_path=tmp_path,
            mode="synthesize",
            file_name=Path("test"),
        )

        def fake_synthesize(script, num):
            path = episodes / f"episode_{num:03d}.mp3"
            path.write_bytes(b"fake-mp3")
            return path

        creator.synthesize_episode = Mock(side_effect=fake_synthesize)
        creator.create_m4b = Mock()

        result = creator._synthesize_from_existing_scripts()

        assert len(result) == 2
        creator.create_m4b.assert_called_once()

    def test_bad_filename_skipped(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AUDIFY_SKIP_TTS_PREFLIGHT", raising=False)
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        episodes = tmp_path / "episodes"
        episodes.mkdir()

        # Bad filename that can't be parsed for episode number
        (scripts / "episode_xyz_script.txt").write_text("content")

        creator = _make_creator(
            scripts_path=scripts, episodes_path=episodes, mode="synthesize"
        )
        result = creator._synthesize_from_existing_scripts()
        assert result == []

    def test_episode_not_created(self, tmp_path, monkeypatch):
        """Episode path returned by synthesize_episode doesn't exist."""
        monkeypatch.delenv("AUDIFY_SKIP_TTS_PREFLIGHT", raising=False)
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        episodes = tmp_path / "episodes"
        episodes.mkdir()

        (scripts / "episode_001_script.txt").write_text("Hello")

        creator = _make_creator(
            scripts_path=scripts, episodes_path=episodes, mode="synthesize"
        )

        ghost_path = episodes / "episode_001.mp3"
        creator.synthesize_episode = Mock(return_value=ghost_path)
        creator.create_m4b = Mock()

        result = creator._synthesize_from_existing_scripts()
        assert result == []
        creator.create_m4b.assert_not_called()


# ---------------------------------------------------------------------------
# synthesize_episode error path
# ---------------------------------------------------------------------------
class TestSynthesizeEpisodeError:
    def test_logs_provider_details_on_failure(self, tmp_path):
        creator = _make_creator(
            episodes_path=tmp_path,
        )

        mock_config = MagicMock()
        mock_config.provider_name = "kokoro"
        mock_config.voice = "af_bella"
        mock_config.language = "en"
        mock_config.base_url = "http://localhost:8887"
        creator._get_tts_config = Mock(return_value=mock_config)
        creator._synthesize_sentences = Mock(
            side_effect=RuntimeError("TTS down")
        )
        creator._break_script_into_segments = Mock(
            return_value=["Hello world."]
        )

        with pytest.raises(RuntimeError, match="TTS down"):
            creator.synthesize_episode("Hello world.", 1)

# ---------------------------------------------------------------------------
# Mode validation in __init__
# ---------------------------------------------------------------------------
class TestModeValidation:
    def test_invalid_mode_raises(self):
        with patch(
            "audify.audiobook_creator.EpubReader"
        ) as mock_reader_cls, patch(
            "audify.audiobook_creator.BaseSynthesizer.__init__",
            return_value=None,
        ):
            mock_reader = MagicMock()
            mock_reader.get_language.return_value = "en"
            mock_reader.title = "Test"
            mock_reader_cls.return_value = mock_reader

            with pytest.raises(ValueError, match="Invalid mode 'bogus'"):
                AudiobookCreator(
                    path="test.epub",
                    language="en",
                    voice="af_bella",
                    model_name="kokoro",
                    mode="bogus",
                )


# ---------------------------------------------------------------------------
# process-only mode in create_audiobook_series
# ---------------------------------------------------------------------------
class TestProcessOnlyMode:
    def test_process_mode_stops_before_synthesis(self, tmp_path):
        creator = _make_creator(
            mode="process",
            scripts_path=tmp_path / "scripts",
            episodes_path=tmp_path / "episodes",
            audiobook_path=tmp_path,
            max_chapters=None,
            language="en",
        )
        (tmp_path / "scripts").mkdir()
        (tmp_path / "episodes").mkdir()
        creator.confirm = False

        mock_reader = MagicMock()
        mock_reader.get_chapters.return_value = ["Chapter text here " * 50]
        mock_reader.get_chapter_title.return_value = "Ch1"
        mock_reader.cleaned_text = "Full text here " * 50
        creator.reader = mock_reader

        creator.generate_audiobook_script = Mock(return_value="Script text " * 50)
        creator.synthesize_episode = Mock()

        result = creator.create_audiobook_series()

        assert result == []
        creator.synthesize_episode.assert_not_called()


# ---------------------------------------------------------------------------
# CLI: container runtime path helpers
# ---------------------------------------------------------------------------
class TestContainerRuntimeExtras:
    def test_resolve_output_path_container_none(self, tmp_path, monkeypatch):
        """output=None in container → returns container output root."""
        import audify.cli as cli

        monkeypatch.setattr(cli, "_is_container_runtime", lambda: True)
        monkeypatch.setattr(cli, "_CONTAINER_OUTPUT_ROOT", tmp_path / "output")

        result = cli._resolve_output_path_for_runtime(None, MagicMock())
        assert result == str(tmp_path / "output")
        assert (tmp_path / "output").exists()

    def test_resolve_output_path_data_marker(self, tmp_path, monkeypatch):
        """Path containing /data/ is remapped."""
        import audify.cli as cli

        monkeypatch.setattr(cli, "_is_container_runtime", lambda: True)
        monkeypatch.setattr(cli, "_CONTAINER_DATA_ROOT", tmp_path)

        result = cli._resolve_output_path_for_runtime(
            "/home/user/data/output/book", MagicMock()
        )
        assert result == str(tmp_path / "output/book")

    def test_resolve_output_path_relative(self, tmp_path, monkeypatch):
        """Relative path in container → mapped to output root."""
        import audify.cli as cli

        monkeypatch.setattr(cli, "_is_container_runtime", lambda: True)
        monkeypatch.setattr(cli, "_CONTAINER_OUTPUT_ROOT", tmp_path / "output")

        result = cli._resolve_output_path_for_runtime("my_book", MagicMock())
        assert result == str(tmp_path / "output" / "my_book")

    def test_resolve_output_path_absolute_no_data(self, tmp_path, monkeypatch):
        """Absolute path without /data/ marker returns as-is."""
        import audify.cli as cli

        monkeypatch.setattr(cli, "_is_container_runtime", lambda: True)
        monkeypatch.setattr(cli, "_CONTAINER_DATA_ROOT", tmp_path)
        monkeypatch.setattr(cli, "_CONTAINER_OUTPUT_ROOT", tmp_path / "output")

        result = cli._resolve_output_path_for_runtime("/some/abs/path", MagicMock())
        assert result == "/some/abs/path"

    def test_stage_input_oserror_fallback(self, tmp_path, monkeypatch):
        """OSError during staging falls back to original path."""
        import audify.cli as cli

        monkeypatch.setattr(cli, "_is_container_runtime", lambda: True)
        monkeypatch.setattr(cli, "_CONTAINER_DATA_ROOT", tmp_path / "data")
        monkeypatch.setattr(
            cli, "_CONTAINER_INPUT_ROOT", tmp_path / "data" / "input"
        )

        external = tmp_path / "file.epub"
        external.write_bytes(b"data")

        with patch("audify.cli.shutil.copy2", side_effect=OSError("disk full")):
            result = cli._stage_input_to_host_data(external, MagicMock())
        assert result == external

    def test_ensure_output_sync_oserror_fallback(self, tmp_path, monkeypatch):
        """OSError during output sync falls back to original path."""
        import audify.cli as cli

        monkeypatch.setattr(cli, "_is_container_runtime", lambda: True)
        monkeypatch.setattr(cli, "_CONTAINER_DATA_ROOT", tmp_path / "data")
        monkeypatch.setattr(
            cli, "_CONTAINER_OUTPUT_ROOT", tmp_path / "data" / "output"
        )

        external = tmp_path / "output_dir"
        external.mkdir()
        (external / "ep.mp3").write_bytes(b"mp3")

        with patch(
            "audify.cli.shutil.copytree", side_effect=OSError("no space")
        ):
            result = cli._ensure_output_synced_to_host_data(
                external, MagicMock()
            )
        assert result == external

    def test_stage_input_dir(self, tmp_path, monkeypatch):
        """Stage a directory input."""
        import audify.cli as cli

        data_root = tmp_path / "data"
        data_root.mkdir()
        monkeypatch.setattr(cli, "_is_container_runtime", lambda: True)
        monkeypatch.setattr(cli, "_CONTAINER_DATA_ROOT", data_root)
        monkeypatch.setattr(cli, "_CONTAINER_INPUT_ROOT", data_root / "input")

        src_dir = tmp_path / "mybooks"
        src_dir.mkdir()
        (src_dir / "a.txt").write_text("hello")

        result = cli._stage_input_to_host_data(src_dir, MagicMock())
        assert result == data_root / "input" / "mybooks"
        assert (result / "a.txt").exists()

    def test_ensure_output_sync_file(self, tmp_path, monkeypatch):
        """Sync a single file output."""
        import audify.cli as cli

        data_root = tmp_path / "data"
        data_root.mkdir()
        monkeypatch.setattr(cli, "_is_container_runtime", lambda: True)
        monkeypatch.setattr(cli, "_CONTAINER_DATA_ROOT", data_root)
        monkeypatch.setattr(cli, "_CONTAINER_OUTPUT_ROOT", data_root / "output")

        src_file = tmp_path / "book.m4b"
        src_file.write_bytes(b"m4b")

        result = cli._ensure_output_synced_to_host_data(src_file, MagicMock())
        assert result == data_root / "output" / "book.m4b"
        assert result.read_bytes() == b"m4b"

# ---------------------------------------------------------------------------
# CLI: _contains_audio_artifacts
# ---------------------------------------------------------------------------
class TestContainsAudioArtifacts:
    def test_nonexistent_path(self, tmp_path):
        import audify.cli as cli

        assert cli._contains_audio_artifacts(tmp_path / "nope") is False

    def test_dir_with_m4b(self, tmp_path):
        import audify.cli as cli

        (tmp_path / "book.m4b").write_bytes(b"m4b")
        assert cli._contains_audio_artifacts(tmp_path) is True

# ---------------------------------------------------------------------------
# CLI: --process-only / --synthesize-only mutual exclusion
# ---------------------------------------------------------------------------
class TestProcessSynthesizeMutualExclusion:
    def test_both_flags_error(self):
        import tempfile

        from click.testing import CliRunner

        from audify.cli import cli

        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".epub") as f:
            result = runner.invoke(
                cli, [f.name, "--process-only", "--synthesize-only"]
            )
        assert result.exit_code == 1
        assert "mutually exclusive" in result.output


# ---------------------------------------------------------------------------
# text_to_speech: batch separator accounting
# ---------------------------------------------------------------------------
class TestBatchSeparatorAccounting:
    """Test _batch_sentences accounts for separator chars."""

    def _make_synth(self):
        from audify.text_to_speech import BaseSynthesizer

        synth = BaseSynthesizer.__new__(BaseSynthesizer)
        return synth

# ---------------------------------------------------------------------------
# ebook reader: TOC match counter fix
# ---------------------------------------------------------------------------
class TestTocMatchCounter:


    def test_flatten_toc_catches_type_error(self):
        """_flatten_toc_hrefs catches TypeError from bad TOC structure."""
        from audify.readers.ebook import EpubReader

        class BadHref:
            @property
            def href(self):
                raise TypeError("broken")

        reader = EpubReader.__new__(EpubReader)
        reader.book = MagicMock()
        # A 2-tuple whose nav_point.href raises TypeError triggers the
        # except (TypeError, AttributeError) handler in _walk.
        reader.book.toc = [(BadHref(), [])]
        result = reader._flatten_toc_hrefs()
        assert result == []


# ---------------------------------------------------------------------------
# api/app.py: version fallback (lines 41-42)
# ---------------------------------------------------------------------------
class TestApiVersionFallback:
    def test_version_fallback_when_package_not_found(self):
        """The app module falls back to '0.1.0' when package not installed."""
        import importlib

        import audify.api.app as app_mod

        with patch(
            "importlib.metadata.version",
            side_effect=importlib.metadata.PackageNotFoundError,
        ):
            importlib.reload(app_mod)
            assert app_mod._version == "0.1.0"

        # Reload again to restore normal state
        importlib.reload(app_mod)


# ---------------------------------------------------------------------------
# Byte-aware batching
# ---------------------------------------------------------------------------
class TestByteAwareBatching:
    """Test _batch_sentences with unit='bytes' for multi-byte UTF-8 text."""

    def _make_synth(self):
        from audify.text_to_speech import BaseSynthesizer

        synth = BaseSynthesizer.__new__(BaseSynthesizer)
        return synth

    def test_oversized_sentence_split_by_words(self):
        """A sentence exceeding byte limit is split at word boundaries."""
        synth = self._make_synth()
        sentence = "Olá mundo maravilhoso"  # multi-byte + long
        # Each word: "Olá"=4B, "mundo"=5B, "maravilhoso"=11B
        batches = synth._batch_sentences([sentence], 12, unit="bytes")
        # Should be split into parts that each fit in 12 bytes
        assert len(batches) >= 2
        for batch in batches:
            batch_text = " ".join(batch)
            assert len(batch_text.encode("utf-8")) <= 12

    def test_oversized_single_word_hard_split(self):
        """A single word exceeding limit is hard-split by characters."""
        synth = self._make_synth()
        # A long word that exceeds 5-byte limit
        word = "abcdefghij"  # 10 bytes (ASCII)
        batches = synth._batch_sentences([word], 5, unit="bytes")
        assert len(batches) >= 2
        for batch in batches:
            assert len(" ".join(batch).encode("utf-8")) <= 5


# ---------------------------------------------------------------------------
# Provider limit properties
# ---------------------------------------------------------------------------
class TestProviderLimits:
    """Test that each TTS provider has correct limit configuration."""

    def test_google_limit_unit_is_bytes(self):
        with patch.dict("os.environ", {"GOOGLE_APPLICATION_CREDENTIALS": "/fake"}):
            from audify.utils.api_config import GoogleTTSConfig

            config = GoogleTTSConfig.__new__(GoogleTTSConfig)
            assert config.limit_unit == "bytes"

    def test_google_max_text_length(self):
        with patch.dict("os.environ", {"GOOGLE_APPLICATION_CREDENTIALS": "/fake"}):
            from audify.utils.api_config import GoogleTTSConfig

            config = GoogleTTSConfig.__new__(GoogleTTSConfig)
            assert config.max_text_length == 4800

    def test_openai_max_text_length(self):
        from audify.utils.api_config import OpenAITTSConfig

        config = OpenAITTSConfig.__new__(OpenAITTSConfig)
        assert config.max_text_length == 4096

    def test_aws_limit_unit_is_bytes(self):
        from audify.utils.api_config import AWSTTSConfig

        config = AWSTTSConfig.__new__(AWSTTSConfig)
        assert config.limit_unit == "bytes"

    def test_kokoro_limit_unit_is_chars(self):
        from audify.utils.api_config import KokoroTTSConfig

        config = KokoroTTSConfig.__new__(KokoroTTSConfig)
        assert config.limit_unit == "chars"

    def test_kokoro_max_text_length(self):
        from audify.utils.api_config import KokoroTTSConfig

        config = KokoroTTSConfig.__new__(KokoroTTSConfig)
        assert config.max_text_length == 5000


# ---------------------------------------------------------------------------
# TTSSynthesisError threshold
# ---------------------------------------------------------------------------
class TestTTSSynthesisThreshold:
    """Test the failure-rate gate in _synthesize_with_provider."""

    @staticmethod
    def _make_synth(tmp_path):
        from audify.text_to_speech import BaseSynthesizer

        synth = BaseSynthesizer.__new__(BaseSynthesizer)
        synth.tmp_dir = tmp_path / "test_tts"
        return synth

# ---------------------------------------------------------------------------
# LLM token size validation
# ---------------------------------------------------------------------------
class TestLLMTokenValidation:
    """Test that _split_text_into_chunks warns on oversized chunks."""

    def _make_creator(self):
        creator = AudiobookCreator.__new__(AudiobookCreator)
        return creator


    def test_warns_for_oversized_chunk(self):
        creator = self._make_creator()
        # ~130k chars → ~32500 tokens, exceeds 90% of context window
        para = "word " * 26000
        big_text = para + "\n\n" + "small paragraph"
        with patch("audify.audiobook_creator.logger") as mock_logger:
            creator._split_text_into_chunks(big_text, max_words=26001)
        assert mock_logger.warning.called
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "context window" in warning_msg
