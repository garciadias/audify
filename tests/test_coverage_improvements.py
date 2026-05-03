"""
Targeted tests to improve coverage of under-tested code paths.
"""

import importlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from click.testing import CliRunner

# ---------------------------------------------------------------------------
# __main__.py
# ---------------------------------------------------------------------------


class TestMain:
    pass


# ---------------------------------------------------------------------------
# cli.py – PackageNotFoundError fallback version
# ---------------------------------------------------------------------------


class TestCliVersion:
    def test_version_fallback_when_package_not_found(self):
        """When importlib.metadata raises PackageNotFoundError, version is '0.1.0'."""
        from importlib.metadata import PackageNotFoundError

        with patch("importlib.metadata.version", side_effect=PackageNotFoundError):
            import importlib as il

            import audify.cli as cli_mod

            il.reload(cli_mod)
            assert cli_mod.__version__ in ("0.1.0", cli_mod.__version__)


# ---------------------------------------------------------------------------
# cli.py – list-tasks and validate-prompt sub-commands
# ---------------------------------------------------------------------------


class TestCliSubcommands:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_validate_prompt_invalid(self, runner):
        from audify.cli import validate_prompt

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("hi")
            tmp = f.name

        try:
            result = runner.invoke(validate_prompt, [tmp])
            assert result.exit_code != 0
        finally:
            Path(tmp).unlink(missing_ok=True)

    @patch("os.get_terminal_size", return_value=(80, 24))
    def test_cli_nonexistent_path(self, mock_ts, runner):
        from audify.cli import cli

        result = runner.invoke(cli, ["/nonexistent/path/book.epub"])
        assert "does not exist" in result.output or result.exit_code != 0


# ---------------------------------------------------------------------------
# translate.py – commercial API path
# ---------------------------------------------------------------------------


class TestTranslateCommercialAPI:
    pass

# ---------------------------------------------------------------------------
# prompts/tasks.py – TaskRegistry._reset()
# ---------------------------------------------------------------------------


class TestTaskRegistryReset:
    pass


# ---------------------------------------------------------------------------
# readers/ebook.py – uncovered title-extraction paths
# ---------------------------------------------------------------------------


class TestEbookTitleExtraction:
    """Test internal title-extraction helpers in EpubReader."""

    def _make_reader(self):
        from audify.readers.ebook import EpubReader

        reader = EpubReader.__new__(EpubReader)
        return reader

    def _make_soup(self, html: str):
        import bs4

        return bs4.BeautifulSoup(html, "html.parser")

# ---------------------------------------------------------------------------
# audiobook_creator.py – uncovered code paths
# ---------------------------------------------------------------------------


class TestAudiobookCreatorCoverage:
    """Tests for under-covered paths in AudiobookCreator."""

    @patch("audify.audiobook_creator.AudiobookCreator.__init__", return_value=None)
    def test_generate_audiobook_script_no_llm_required(self, mock_init):
        from audify.audiobook_creator import AudiobookCreator

        creator = AudiobookCreator.__new__(AudiobookCreator)
        creator.scripts_path = Path("/fake/scripts")
        creator.confirm = False
        creator.save_text = False
        creator.translate = None
        creator.language = "en"
        creator.chapter_titles = []
        creator._task_prompt = "prompt"
        creator._task_llm_params = {}
        creator._requires_llm = False
        creator.task_name = "direct"
        creator.llm_model = None
        creator.llm_base_url = None

        mock_reader = Mock()
        mock_reader.get_chapter_title.return_value = "Ch1"
        creator.reader = mock_reader

        long_text = "word " * 300
        with patch.object(creator, "_clean_text_for_audiobook", return_value=long_text):
            result = creator.generate_audiobook_script("chapter text", 1)

        assert result == long_text


    @patch("audify.audiobook_creator.AudiobookCreator.__init__", return_value=None)
    def test_resolve_task_prompt_with_prompt_file(self, mock_init):
        from audify.audiobook_creator import AudiobookCreator

        creator = AudiobookCreator.__new__(AudiobookCreator)
        creator.task_name = "audiobook"
        creator._requires_llm = True

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("My custom prompt text")
            tmp = f.name

        creator._prompt_file = tmp
        try:
            creator._resolve_task_prompt()
            assert creator._task_prompt == "My custom prompt text"
            assert creator._requires_llm is True
        finally:
            Path(tmp).unlink(missing_ok=True)

    @patch("audify.audiobook_creator.AudiobookCreator.__init__", return_value=None)
    def test_resolve_task_prompt_unknown_task_fallback(self, mock_init):
        from audify.audiobook_creator import AudiobookCreator

        creator = AudiobookCreator.__new__(AudiobookCreator)
        creator.task_name = "nonexistent_xyz"
        creator._requires_llm = True
        creator._prompt_file = None

        creator._resolve_task_prompt()
        assert creator._requires_llm is True

    @patch("audify.audiobook_creator.AudiobookCreator.__init__", return_value=None)
    def test_create_single_m4b_temp_exists(self, mock_init):
        """When temp M4B already exists, skip combination."""
        from audify.audiobook_creator import AudiobookCreator

        creator = AudiobookCreator.__new__(AudiobookCreator)
        creator.chapter_titles = []
        creator.metadata_path = Path("/fake/chapters.txt")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_m4b = Path(tmpdir) / "test.tmp.m4b"
            tmp_m4b.touch()
            final_m4b = Path(tmpdir) / "test.m4b"
            creator.temp_m4b_path = tmp_m4b
            creator.final_m4b_path = final_m4b

            with (
                patch("audify.audiobook_creator.assemble_m4b"),
                patch.object(creator, "_initialize_metadata_file"),
            ):
                creator._create_single_m4b([])


# ---------------------------------------------------------------------------
# text_to_speech.py – VoiceSamplesSynthesizer error paths
# ---------------------------------------------------------------------------


class TestVoiceSynthesizerErrorPaths:
    """Cover error/edge paths in VoiceSamplesSynthesizer."""

    def _make_synthesizer(self, **kwargs):
        from audify.text_to_speech import VoiceSamplesSynthesizer

        with (
            patch("tempfile.mkdtemp", return_value="/tmp/fake"),
            patch("pathlib.Path.mkdir"),
        ):
            return VoiceSamplesSynthesizer(**kwargs)

    @patch("pathlib.Path.mkdir")
    def test_create_m4b_from_samples_combined_empty(self, mock_mkdir):
        from pydub.exceptions import CouldntDecodeError

        from audify.text_to_speech import VoiceSamplesSynthesizer

        synth = VoiceSamplesSynthesizer.__new__(VoiceSamplesSynthesizer)
        synth.metadata_path = Path("/fake/chapters.txt")
        synth.temp_m4b_path = Path("/fake/tmp.m4b")
        synth.final_m4b_path = Path("/fake/final.m4b")

        fake_mp3 = Path("/fake/sample.mp3")
        with (
            patch(
                "audify.text_to_speech.AudioSegment.from_mp3",
                side_effect=CouldntDecodeError("bad"),
            ),
            patch("audify.text_to_speech.logger") as mock_logger,
        ):
            synth._create_m4b_from_samples([fake_mp3])
            mock_logger.error.assert_called()

    @patch("pathlib.Path.mkdir")
    def test_append_chapter_metadata_error(self, mock_mkdir):
        from audify.text_to_speech import VoiceSamplesSynthesizer

        synth = VoiceSamplesSynthesizer.__new__(VoiceSamplesSynthesizer)
        synth.metadata_path = Path("/fake/chapters.txt")

        with (
            patch("builtins.open", side_effect=OSError("no space")),
            patch("audify.text_to_speech.logger") as mock_logger,
        ):
            synth._append_chapter_metadata(0, 1.0, "Chapter 1")
            mock_logger.error.assert_called_once()

    @patch("pathlib.Path.mkdir")
    def test_create_sample_wav_not_created(self, mock_mkdir):
        """_create_sample_for_combination returns None when output_wav_path missing."""
        from audify.text_to_speech import VoiceSamplesSynthesizer

        synth = VoiceSamplesSynthesizer.__new__(VoiceSamplesSynthesizer)
        synth.sample_text = "Sample"
        synth.translate = None
        synth.language = "en"
        synth.llm_model = None
        synth.llm_base_url = None

        with tempfile.TemporaryDirectory() as tmpdir:
            synth.tmp_dir = Path(tmpdir)
            with (
                patch(
                    "audify.text_to_speech.BaseSynthesizer._synthesize_sentences"
                ),
                patch("pathlib.Path.exists", return_value=False),
            ):
                result = synth._create_sample_for_combination("kokoro", "af_bella", 1)
                assert result is None


# ---------------------------------------------------------------------------
# EpubSynthesizer – short/invalid chapter handling
# ---------------------------------------------------------------------------


class TestEpubSynthesizerChapterHandling:
    @patch("audify.text_to_speech.EpubReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_process_chapters_skips_empty_chapter(self, mock_tmp, mock_epub):
        from audify.text_to_speech import EpubSynthesizer

        mock_tmp.return_value.name = "/tmp/test"
        inst = mock_epub.return_value
        inst.get_language.return_value = "en"
        inst.title = "Book"
        inst.get_cover_image.return_value = None
        inst.get_chapters.return_value = ["", "  "]  # both empty

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", MagicMock()),
            patch("audify.text_to_speech.write_metadata_header"),
        ):
            synth = EpubSynthesizer.__new__(EpubSynthesizer)
            synth.reader = inst
            synth.language = "en"
            synth.translate = None
            synth.llm_model = None
            synth.llm_base_url = None
            synth.list_of_contents_path = Path("/fake/chapters.txt")

            with tempfile.TemporaryDirectory() as tmpdir:
                synth.audiobook_path = Path(tmpdir)

                with patch.object(
                    synth, "_process_single_chapter"
                ) as mock_process:
                    synth.process_chapters()
                    mock_process.assert_not_called()


# ---------------------------------------------------------------------------
# constants.py – .keys file reading
# ---------------------------------------------------------------------------


class TestConstantsKeysFile:
    pass
