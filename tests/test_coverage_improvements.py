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
    def test_main_module_importable(self):
        mod = importlib.import_module("audify.__main__")
        assert hasattr(mod, "cli")


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

    def test_list_tasks_command(self, runner):
        from audify.cli import list_tasks

        result = runner.invoke(list_tasks)
        assert result.exit_code == 0
        assert "audiobook" in result.output
        assert "podcast" in result.output
        assert "direct" in result.output

    def test_validate_prompt_valid(self, runner):
        from audify.cli import validate_prompt

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("A sufficiently long prompt for validation testing purposes here.")
            tmp = f.name

        try:
            result = runner.invoke(validate_prompt, [tmp])
            assert result.exit_code == 0
            assert "valid" in result.output.lower()
        finally:
            Path(tmp).unlink(missing_ok=True)

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
    def test_get_translation_config_commercial(self):
        from audify.translate import _get_translation_config
        from audify.utils.api_config import CommercialAPIConfig

        config = _get_translation_config(model="api:deepseek/deepseek-chat")
        assert isinstance(config, CommercialAPIConfig)
        assert config.model == "deepseek/deepseek-chat"

    def test_translate_sentence_commercial_api(self):
        from audify.translate import translate_sentence
        from audify.utils.api_config import CommercialAPIConfig

        mock_config = Mock(spec=CommercialAPIConfig)
        mock_config.model = "some/model"
        mock_config.generate.return_value = "Bonjour le monde"

        with patch(
            "audify.translate._get_translation_config", return_value=mock_config
        ):
            result = translate_sentence(
                "Hello world",
                model="api:some/model",
                src_lang="en",
                tgt_lang="fr",
            )
        assert result == "Bonjour le monde"
        mock_config.generate.assert_called_once()


# ---------------------------------------------------------------------------
# prompts/tasks.py – TaskRegistry._reset()
# ---------------------------------------------------------------------------


class TestTaskRegistryReset:
    def test_reset_clears_tasks(self):
        from audify.prompts.tasks import TaskConfig, TaskRegistry

        original = TaskRegistry.get_all()
        try:
            TaskRegistry._reset()
            assert TaskRegistry.get_all() == {}

            TaskRegistry.register(TaskConfig(name="test_task", prompt="test"))
            assert "test_task" in TaskRegistry.get_all()
        finally:
            TaskRegistry._reset()
            for name, config in original.items():
                TaskRegistry.register(config)


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

    def test_extract_from_title_attributes_no_match(self):
        reader = self._make_reader()
        soup = self._make_soup("<div><p>No title here</p></div>")
        result = reader._extract_from_title_attributes(soup)
        assert result == ""

    def test_extract_from_emphasis_tags_no_match(self):
        reader = self._make_reader()
        soup = self._make_soup(
            "<body><div><p>A long sentence that ends with a period.</p></div></body>"
        )
        result = reader._extract_from_emphasis_tags(soup)
        assert result == ""

    def test_is_leaf_paragraph_non_tag(self):
        import bs4

        from audify.readers.ebook import EpubReader

        soup = bs4.BeautifulSoup("plain text", "html.parser")
        text_node = list(soup.children)[0]
        assert EpubReader._is_leaf_paragraph(text_node) is True

    def test_extract_short_paragraph_title_no_match(self):
        reader = self._make_reader()
        html = "<body><p>This is a long sentence that ends with a period.</p></body>"
        soup = self._make_soup(html)
        result = reader._extract_short_paragraph_title(soup)
        assert result == ""


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
    def test_generate_audiobook_script_short_text(self, mock_init):
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
        creator._requires_llm = True
        creator.llm_model = None
        creator.llm_base_url = None

        mock_reader = Mock()
        mock_reader.get_chapter_title.return_value = "Ch1"
        creator.reader = mock_reader

        short_text = "too short"
        with patch.object(
            creator, "_clean_text_for_audiobook", return_value=short_text
        ):
            result = creator.generate_audiobook_script("chapter text", 1)

        assert result == short_text

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

    def test_audiobook_epub_creator_raises_without_epub_reader(self):
        from audify.audiobook_creator import AudiobookEpubCreator

        with (
            patch("audify.audiobook_creator.EpubReader") as mock_epub,
            patch(
                "audify.audiobook_creator.BaseSynthesizer.__init__",
                return_value=None,
            ),
            patch("audify.audiobook_creator.LLMClient"),
            patch("audify.audiobook_creator.AudiobookCreator._setup_paths"),
            patch("audify.audiobook_creator.AudiobookCreator._resolve_task_prompt"),
            patch("pathlib.Path.mkdir"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            mock_epub_instance = Mock()
            mock_epub_instance.get_language.return_value = "en"
            mock_epub_instance.title = "Test"
            mock_epub_instance.get_chapters = None  # no get_chapters
            mock_epub_instance.get_cover_image.return_value = None
            del mock_epub_instance.get_chapters
            mock_epub.return_value = mock_epub_instance

            with pytest.raises(ValueError, match="requires an EPUB reader"):
                AudiobookEpubCreator(path="test.epub")


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
    def test_create_metadata_file_error_raises(self, mock_mkdir):
        from audify.text_to_speech import VoiceSamplesSynthesizer

        synth = VoiceSamplesSynthesizer.__new__(VoiceSamplesSynthesizer)
        synth.metadata_path = Path("/fake/chapters.txt")

        with patch("builtins.open", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                synth._create_metadata_file([])

    @patch("pathlib.Path.mkdir")
    def test_create_m4b_from_samples_empty(self, mock_mkdir):
        from audify.text_to_speech import VoiceSamplesSynthesizer

        synth = VoiceSamplesSynthesizer.__new__(VoiceSamplesSynthesizer)
        synth.metadata_path = Path("/fake/chapters.txt")
        synth.temp_m4b_path = Path("/fake/tmp.m4b")
        synth.final_m4b_path = Path("/fake/final.m4b")

        with patch("audify.text_to_speech.logger") as mock_logger:
            synth._create_m4b_from_samples([])
            mock_logger.error.assert_called_once()

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
    def test_keys_file_parsing(self):
        """Test that constants.py reads .keys file correctly."""
        keys_content = "# comment\n\nMY_TEST_KEY=my_value\nINVALID_NO_VALUE\n"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".keys", delete=False
        ) as f:
            f.write(keys_content)
            tmp = f.name

        try:
            result: dict[str, str] = {}
            with open(tmp, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip()
                        if key and value:
                            result[key] = value
            assert result == {"MY_TEST_KEY": "my_value"}
        finally:
            Path(tmp).unlink(missing_ok=True)
