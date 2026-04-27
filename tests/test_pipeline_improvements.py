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
    def test_unset_returns_default_false(self, monkeypatch):
        monkeypatch.delenv("MY_FLAG", raising=False)
        assert _env_flag("MY_FLAG") is False

    def test_unset_returns_custom_default(self, monkeypatch):
        monkeypatch.delenv("MY_FLAG", raising=False)
        assert _env_flag("MY_FLAG", default=True) is True

    @pytest.mark.parametrize("val", ["1", "true", "True", "YES", "on", " 1 "])
    def test_truthy_values(self, val, monkeypatch):
        monkeypatch.setenv("MY_FLAG", val)
        assert _env_flag("MY_FLAG") is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "off", "random"])
    def test_falsy_values(self, val, monkeypatch):
        monkeypatch.setenv("MY_FLAG", val)
        assert _env_flag("MY_FLAG") is False


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

    def test_skip_when_no_tts_config(self, monkeypatch):
        monkeypatch.delenv("AUDIFY_SKIP_TTS_PREFLIGHT", raising=False)
        creator = _make_creator()
        # No _tts_config attribute → skip silently
        creator._verify_tts_provider_available()

    def test_raises_when_unavailable_and_audiobook_task(self, monkeypatch):
        monkeypatch.delenv("AUDIFY_SKIP_TTS_PREFLIGHT", raising=False)
        monkeypatch.delenv("AUDIFY_STRICT_TTS_PREFLIGHT", raising=False)

        mock_config = MagicMock()
        mock_config.is_available.return_value = False
        mock_config.provider_name = "kokoro"
        mock_config.base_url = "http://localhost:8887"

        creator = _make_creator(task_name="audiobook")
        creator._tts_config = mock_config
        creator._get_tts_config = Mock(return_value=mock_config)

        with pytest.raises(RuntimeError, match="not available"):
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

    def test_raises_when_strict_preflight_env_set(self, monkeypatch):
        monkeypatch.delenv("AUDIFY_SKIP_TTS_PREFLIGHT", raising=False)
        monkeypatch.setenv("AUDIFY_STRICT_TTS_PREFLIGHT", "1")

        mock_config = MagicMock()
        mock_config.is_available.return_value = False
        mock_config.provider_name = "kokoro"
        mock_config.base_url = "http://localhost:8887"

        creator = _make_creator(task_name="direct")
        creator._tts_config = mock_config
        creator._get_tts_config = Mock(return_value=mock_config)

        with pytest.raises(RuntimeError, match="not available"):
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

    def test_kokoro_specific_guidance_in_error(self, monkeypatch):
        monkeypatch.delenv("AUDIFY_SKIP_TTS_PREFLIGHT", raising=False)

        mock_config = MagicMock()
        mock_config.is_available.return_value = False
        mock_config.provider_name = "kokoro"
        mock_config.base_url = "http://localhost:8887"

        creator = _make_creator(tts_provider="kokoro", task_name="audiobook")
        creator._tts_config = mock_config
        creator._get_tts_config = Mock(return_value=mock_config)

        with pytest.raises(RuntimeError, match="Kokoro"):
            creator._verify_tts_provider_available()

    def test_no_base_url_in_error(self, monkeypatch):
        """When tts_config has no base_url, error message skips API URL line."""
        monkeypatch.delenv("AUDIFY_SKIP_TTS_PREFLIGHT", raising=False)

        mock_config = MagicMock(spec=["is_available", "provider_name"])
        mock_config.is_available.return_value = False
        mock_config.provider_name = "openai"

        creator = _make_creator(tts_provider="openai", task_name="audiobook")
        creator._tts_config = mock_config
        creator._get_tts_config = Mock(return_value=mock_config)

        with pytest.raises(RuntimeError) as exc_info:
            creator._verify_tts_provider_available()
        assert "Configured API URL" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# _save_chapter_titles / _load_chapter_titles
# ---------------------------------------------------------------------------
class TestChapterTitlesPersistence:
    def test_save_and_load(self, tmp_path):
        creator = _make_creator(scripts_path=tmp_path)
        creator.chapter_titles = ["Intro", "Chapter 1", "Epilogue"]
        creator._save_chapter_titles()

        loaded = creator._load_chapter_titles()
        assert loaded == ["Intro", "Chapter 1", "Epilogue"]

    def test_load_missing_file(self, tmp_path):
        creator = _make_creator(scripts_path=tmp_path)
        assert creator._load_chapter_titles() == []

    def test_load_corrupt_json(self, tmp_path):
        (tmp_path / "chapter_titles.json").write_text("{bad json", encoding="utf-8")
        creator = _make_creator(scripts_path=tmp_path)
        assert creator._load_chapter_titles() == []

    def test_save_io_error(self, tmp_path):
        """IOError during save should warn, not crash."""
        creator = _make_creator(scripts_path=tmp_path / "nonexistent")
        creator.chapter_titles = ["A"]
        # scripts_path doesn't exist → IOError
        creator._save_chapter_titles()  # should not raise


# ---------------------------------------------------------------------------
# _validate_chapters
# ---------------------------------------------------------------------------
class TestValidateChapters:
    def test_empty_list(self):
        creator = _make_creator()
        assert creator._validate_chapters([]) == []

    def test_all_sufficient(self):
        creator = _make_creator()
        counts = [("Ch1", 500), ("Ch2", 300)]
        flagged = creator._validate_chapters(counts)
        assert flagged == []

    def test_short_chapters_flagged(self):
        creator = _make_creator()
        counts = [("Intro", 50), ("Ch1", 500), ("Outro", 10)]
        flagged = creator._validate_chapters(counts)
        assert len(flagged) == 2
        assert flagged[0][0] == "Intro"
        assert flagged[1][0] == "Outro"

    def test_long_title_truncated(self):
        """Titles > 50 chars should be truncated in the table (no crash)."""
        creator = _make_creator()
        long_title = "A" * 60
        counts = [(long_title, 500)]
        flagged = creator._validate_chapters(counts)
        assert flagged == []


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

    def test_logs_without_base_url(self, tmp_path):
        """Error path when tts_config has no base_url attribute."""
        creator = _make_creator(episodes_path=tmp_path)

        mock_config = MagicMock(spec=["provider_name", "voice", "language"])
        mock_config.provider_name = "openai"
        mock_config.voice = "alloy"
        mock_config.language = "en"
        creator._get_tts_config = Mock(return_value=mock_config)
        creator._synthesize_sentences = Mock(
            side_effect=RuntimeError("API error")
        )
        creator._break_script_into_segments = Mock(
            return_value=["Some text."]
        )

        with pytest.raises(RuntimeError, match="API error"):
            creator.synthesize_episode("Some text.", 1)


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

    def test_resolve_input_candidates(self, tmp_path, monkeypatch):
        """Test input path resolution with multiple candidates."""
        import audify.cli as cli

        data_root = tmp_path / "data"
        (data_root / "ebooks").mkdir(parents=True)
        target = data_root / "ebooks" / "book.epub"
        target.write_bytes(b"epub")

        monkeypatch.setattr(cli, "_is_container_runtime", lambda: True)
        monkeypatch.setattr(cli, "_CONTAINER_DATA_ROOT", data_root)

        # Absolute path not containing /data/ but name matches
        result = cli._resolve_input_path_for_runtime(
            "/absolute/book.epub", MagicMock()
        )
        assert result == str(target)


# ---------------------------------------------------------------------------
# CLI: _contains_audio_artifacts
# ---------------------------------------------------------------------------
class TestContainsAudioArtifacts:
    def test_nonexistent_path(self, tmp_path):
        import audify.cli as cli

        assert cli._contains_audio_artifacts(tmp_path / "nope") is False

    def test_file_mp3(self, tmp_path):
        import audify.cli as cli

        f = tmp_path / "audio.mp3"
        f.write_bytes(b"mp3")
        assert cli._contains_audio_artifacts(f) is True

    def test_file_txt(self, tmp_path):
        import audify.cli as cli

        f = tmp_path / "notes.txt"
        f.write_text("hello")
        assert cli._contains_audio_artifacts(f) is False

    def test_dir_with_m4b(self, tmp_path):
        import audify.cli as cli

        (tmp_path / "book.m4b").write_bytes(b"m4b")
        assert cli._contains_audio_artifacts(tmp_path) is True

    def test_dir_empty(self, tmp_path):
        import audify.cli as cli

        assert cli._contains_audio_artifacts(tmp_path) is False


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

    def test_separator_prevents_overflow(self):
        synth = self._make_synth()
        # 10-char max, sentences of 4 each. With separator (space=1),
        # two sentences = 4+1+4 = 9 ≤ 10 → same batch.
        # Three sentences = 9+1+4 = 14 > 10 → new batch.
        batches = synth._batch_sentences(
            ["aaaa", "bbbb", "cccc"], max_length=10
        )
        assert batches == [["aaaa", "bbbb"], ["cccc"]]

    def test_exact_fit_with_separator(self):
        synth = self._make_synth()
        # 9 chars max. "aaaa" + " " + "bbbb" = 9 → fits
        batches = synth._batch_sentences(
            ["aaaa", "bbbb"], max_length=9
        )
        assert batches == [["aaaa", "bbbb"]]

    def test_separator_causes_split(self):
        synth = self._make_synth()
        # 8 chars max. "aaaa" + " " + "bbbb" = 9 > 8 → split
        batches = synth._batch_sentences(
            ["aaaa", "bbbb"], max_length=8
        )
        assert batches == [["aaaa"], ["bbbb"]]


# ---------------------------------------------------------------------------
# ebook reader: TOC match counter fix
# ---------------------------------------------------------------------------
class TestTocMatchCounter:
    def test_first_item_boundary_counted(self, caplog):
        """First spine item matching TOC increments matches_found.

        When the very first spine item is a TOC boundary, matches_found
        should still be incremented (even though current_group is empty).
        We need >= _MIN_TOC_MATCHES (3) boundaries that produce valid
        chapters to avoid the fallback path.
        """
        from ebooklib import ITEM_DOCUMENT

        from audify.readers.ebook import EpubReader

        reader = EpubReader.__new__(EpubReader)

        def _make_item(name):
            item = MagicMock()
            item.get_name.return_value = name
            item.get_type.return_value = ITEM_DOCUMENT
            body = (
                b"<html><body><p>"
                + (f"Content for {name}. " * 80).encode()
                + b"</p></body></html>"
            )
            item.get_body_content.return_value = body
            return item

        items = {f"id{i}": _make_item(f"ch{i}.xhtml") for i in range(1, 5)}
        reader.book = MagicMock()
        reader.book.spine = [(sid, "yes") for sid in items]
        reader.book.get_item_with_id = lambda sid: items[sid]

        # Mock TOC so _build_toc_item_name_set returns matching names
        toc_entries = []
        for i in range(1, 5):
            entry = MagicMock()
            entry.href = f"ch{i}.xhtml"
            toc_entries.append(entry)
        reader.book.toc = toc_entries

        chapters = reader._get_chapters_grouped_by_toc()

        # With 4 TOC boundaries all matching, we should get chapters
        # (the first boundary starts a group, subsequent ones close/open)
        assert len(chapters) >= 2

    def test_exception_narrowed_to_type_attribute_error(self):
        """_flatten_toc_hrefs catches TypeError and AttributeError."""
        from audify.readers.ebook import EpubReader

        reader = EpubReader.__new__(EpubReader)
        reader.book = MagicMock()

        # Create an object whose .href property raises AttributeError
        class BadEntry:
            @property
            def href(self):
                raise AttributeError("broken href")

        reader.book.toc = [BadEntry()]
        result = reader._flatten_toc_hrefs()
        assert result == []

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

    def test_ascii_bytes_equals_chars(self):
        """For pure ASCII text, bytes and chars modes produce identical batches."""
        synth = self._make_synth()
        sentences = ["Hello world", "Test sentence", "Another one"]
        chars_batches = synth._batch_sentences(sentences, 25, unit="chars")
        bytes_batches = synth._batch_sentences(sentences, 25, unit="bytes")
        assert chars_batches == bytes_batches

    def test_multibyte_text_respects_byte_limit(self):
        """Portuguese text with accents must not exceed byte limit."""
        synth = self._make_synth()
        # Each accented char is 2 bytes in UTF-8
        pt_sentence = "ração"  # 5 chars, 7 bytes (ç=2B, ã=2B)
        assert len(pt_sentence) == 5
        assert len(pt_sentence.encode("utf-8")) == 7

        # 7-byte limit: one sentence fits exactly
        batches = synth._batch_sentences([pt_sentence, pt_sentence], 7, unit="bytes")
        assert len(batches) == 2  # Can't fit two in one batch (7+1+7=15 > 7)

    def test_multibyte_fits_when_limit_is_chars(self):
        """Same text fits in fewer batches when measured by chars."""
        synth = self._make_synth()
        pt_sentence = "ração"  # 6 chars
        batches = synth._batch_sentences([pt_sentence, pt_sentence], 13, unit="chars")
        assert len(batches) == 1  # 6 + 1 + 6 = 13 ≤ 13

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

    def test_raises_on_high_failure_rate(self, tmp_path):
        """Should raise TTSSynthesisError when >5% of sentences fail."""
        from audify.text_to_speech import TTSSynthesisError

        synth = self._make_synth(tmp_path)
        mock_config = MagicMock()
        mock_config.provider_name = "test"
        mock_config.is_available.return_value = True
        mock_config.get_available_voices.return_value = []
        mock_config.voice = "v"
        mock_config.max_text_length = 5000
        mock_config.limit_unit = "chars"
        # All batches fail
        mock_config.synthesize.return_value = False

        with (
            patch.object(synth, "_get_tts_config", return_value=mock_config),
            pytest.raises(TTSSynthesisError, match="failure threshold"),
        ):
            synth._synthesize_with_provider(
                ["sentence one", "sentence two"],
                tmp_path / "out.wav",
            )

    def test_no_error_on_zero_failures(self, tmp_path):
        """Should succeed when all batches pass."""
        synth = self._make_synth(tmp_path)
        mock_config = MagicMock()
        mock_config.provider_name = "test"
        mock_config.is_available.return_value = True
        mock_config.get_available_voices.return_value = []
        mock_config.voice = "v"
        mock_config.max_text_length = 5000
        mock_config.limit_unit = "chars"

        def fake_synth(text, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"wav")
            return True

        mock_config.synthesize.side_effect = fake_synth

        with (
            patch.object(synth, "_get_tts_config", return_value=mock_config),
            patch("audify.text_to_speech.AudioProcessor.combine_wav_segments"),
        ):
            # Should not raise
            synth._synthesize_with_provider(
                ["hello world"], tmp_path / "out.wav"
            )


# ---------------------------------------------------------------------------
# LLM token size validation
# ---------------------------------------------------------------------------
class TestLLMTokenValidation:
    """Test that _split_text_into_chunks warns on oversized chunks."""

    def _make_creator(self):
        creator = AudiobookCreator.__new__(AudiobookCreator)
        return creator

    def test_no_warning_for_small_text(self):
        creator = self._make_creator()
        with patch("audify.audiobook_creator.logger") as mock_logger:
            chunks = creator._split_text_into_chunks("Short text " * 10)
        assert len(chunks) == 1
        warning_calls = [
            str(c) for c in mock_logger.warning.call_args_list
        ]
        assert not any("context window" in w for w in warning_calls)

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
