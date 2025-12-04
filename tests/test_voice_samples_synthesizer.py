"""Tests for VoiceSamplesSynthesizer class."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest
import requests

from audify.text_to_speech import VoiceSamplesSynthesizer


class TestVoiceSamplesSynthesizer:
    """Tests for VoiceSamplesSynthesizer class."""

    @pytest.fixture
    def mock_temp_dir(self):
        """Create a mock temporary directory."""
        with patch("tempfile.mkdtemp") as mock_mkdtemp:
            mock_mkdtemp.return_value = "/tmp/test_voice_samples"
            yield mock_mkdtemp

    @pytest.fixture
    def mock_output_dir(self):
        """Mock the output directory creation."""
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            yield mock_mkdir

    @pytest.fixture
    def synthesizer(self, mock_temp_dir, mock_output_dir):
        """Create a VoiceSamplesSynthesizer instance with mocked dependencies."""
        with patch("pathlib.Path.exists", return_value=True):
            return VoiceSamplesSynthesizer(
                language="en",
                translate=None,
                max_samples=3,
            )

    def test_init_default_values(self, mock_temp_dir, mock_output_dir):
        """Test VoiceSamplesSynthesizer initialization with default values."""
        with patch("pathlib.Path.exists", return_value=True):
            synthesizer = VoiceSamplesSynthesizer()

            assert synthesizer.language == "en"
            assert synthesizer.translate is None
            assert synthesizer.max_samples is None
            assert "Bean on bread" in synthesizer.sample_text
            assert synthesizer.output_path.name == "voice_samples"

    def test_init_with_translation(self, mock_temp_dir, mock_output_dir):
        """Test VoiceSamplesSynthesizer initialization with translation."""
        # Check the actual implementation - translation
        # happens in init if translate is specified
        with patch("pathlib.Path.exists", return_value=True):
            synthesizer = VoiceSamplesSynthesizer(
                language="en",
                translate="es",
                sample_text="Hello world",
            )

            assert synthesizer.language == "en"
            assert synthesizer.translate == "es"
            # The actual implementation doesn't
            # translate in __init__ when translate != None
            # It sets language to translate value instead

    def test_init_custom_values(self, mock_temp_dir, mock_output_dir):
        """Test VoiceSamplesSynthesizer initialization with custom values."""
        with patch("pathlib.Path.exists", return_value=True):
            custom_text = "Custom sample text for testing"
            synthesizer = VoiceSamplesSynthesizer(
                language="fr",
                translate="en",
                sample_text=custom_text,
                max_samples=5,
            )

            assert synthesizer.language == "fr"
            assert synthesizer.translate == "en"
            assert synthesizer.max_samples == 5
            # Note: sample_text will be translated, so we test the original input

    @patch("requests.get")
    def test_get_available_models_and_voices_success(self, mock_get, synthesizer):
        """Test successful retrieval of models and voices."""
        # Mock models response
        mock_models_response = Mock()
        mock_models_response.raise_for_status.return_value = None
        mock_models_response.json.return_value = {
            "data": [
                {"id": "kokoro"},
                {"id": "tts-1"},
                {"name": "model_without_id"},  # Should be ignored
            ]
        }

        # Mock voices response
        mock_voices_response = Mock()
        mock_voices_response.raise_for_status.return_value = None
        mock_voices_response.json.return_value = {
            "voices": ["af_bella", "af_alloy", "en_voice"]
        }

        # Configure mock to return different responses for different URLs
        def side_effect(url, **kwargs):
            if "models" in url:
                return mock_models_response
            elif "voices" in url:
                return mock_voices_response
            else:
                raise ValueError(f"Unexpected URL: {url}")

        mock_get.side_effect = side_effect

        models, voices = synthesizer._get_available_models_and_voices()

        assert models == ["kokoro", "tts-1"]
        assert voices == ["af_alloy", "af_bella", "en_voice"]
        assert mock_get.call_count == 2

    @patch("requests.get")
    def test_get_available_models_and_voices_api_error(self, mock_get, synthesizer):
        """Test API error handling in model/voice retrieval."""
        mock_get.side_effect = requests.RequestException("API Error")

        models, voices = synthesizer._get_available_models_and_voices()

        assert models == []
        assert voices == []

    def test_create_metadata_file(self, synthesizer):
        """Test metadata file creation."""
        combinations = [("kokoro", "af_bella"), ("tts-1", "af_alloy")]

        with patch("builtins.open", mock_open()) as mock_file:
            synthesizer._create_metadata_file(combinations)

            mock_file.assert_called_once_with(synthesizer.metadata_path, "w")
            handle = mock_file()

            # Check that metadata was written
            written_calls = [call.args[0] for call in handle.write.call_args_list]
            written_content = "".join(written_calls)

            assert ";FFMETADATA1" in written_content
            assert "title=Voice Samples Collection" in written_content
            assert "artist=Audify TTS System" in written_content

    def test_append_chapter_metadata(self, synthesizer):
        """Test chapter metadata appending."""
        with patch("builtins.open", mock_open()) as mock_file:
            synthesizer._append_chapter_metadata(
                start_time_ms=1000,
                duration_s=5.5,
                title="Test Chapter"
            )

            mock_file.assert_called_once_with(synthesizer.metadata_path, "a")
            handle = mock_file()

            written_calls = [call.args[0] for call in handle.write.call_args_list]
            written_content = "".join(written_calls)

            assert "[CHAPTER]" in written_content
            assert "START=1000" in written_content
            assert "END=6500" in written_content  # 1000 + 5.5*1000
            assert "title=Test Chapter" in written_content

    @patch("audify.text_to_speech.BaseSynthesizer")
    @patch("audify.text_to_speech.break_text_into_sentences")
    def test_create_sample_for_combination_success(
        self, mock_break_text, mock_base_synthesizer, synthesizer
    ):
        """Test successful sample creation for a model-voice combination."""
        # Mock text breaking
        mock_break_text.return_value = ["Sentence 1.", "Sentence 2."]

        # Mock synthesizer
        mock_synth_instance = Mock()
        mock_base_synthesizer.return_value = mock_synth_instance

        # Mock file operations
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch.object(mock_synth_instance, "_synthesize_kokoro") as mock_synthesize,
            patch.object(mock_synth_instance, "_convert_to_mp3") as mock_convert,
        ):
            mock_convert.return_value = Path("/tmp/test_sample.mp3")

            result = synthesizer._create_sample_for_combination(
                model="kokoro",
                voice="af_bella",
                chapter_index=1
            )

            assert result == Path("/tmp/test_sample.mp3")
            mock_synthesize.assert_called_once()
            mock_convert.assert_called_once()

    @patch("audify.text_to_speech.BaseSynthesizer")
    def test_create_sample_for_combination_failure(
        self, mock_base_synthesizer, synthesizer
    ):
        """Test sample creation failure handling."""
        mock_base_synthesizer.side_effect = Exception("Synthesis failed")

        result = synthesizer._create_sample_for_combination(
            model="kokoro",
            voice="af_bella",
            chapter_index=1
        )

        assert result is None

    @patch("audify.text_to_speech.AudioSegment")
    @patch("subprocess.run")
    def test_create_m4b_from_samples_success(
        self, mock_subprocess, mock_audio_segment, synthesizer
    ):
        """Test successful M4B creation from sample files."""
        # Mock sample files
        sample_files = [
            Path("/tmp/sample_001_kokoro_af_bella.mp3"),
            Path("/tmp/sample_002_tts-1_af_alloy.mp3"),
        ]

        # Mock AudioSegment

        mock_audio_instance = MagicMock()
        mock_audio_instance.__len__ = MagicMock(return_value=5000)  # 5 seconds in ms
        mock_audio_segment.empty.return_value = MagicMock()
        mock_audio_segment.from_mp3.return_value = mock_audio_instance

        # Mock combined audio
        mock_combined = MagicMock()
        mock_combined.__len__ = MagicMock(return_value=10000)
        mock_audio_segment.empty.return_value = mock_combined
        mock_combined.__iadd__ = MagicMock(return_value=mock_combined)

        with (
            patch.object(synthesizer, "_append_chapter_metadata") as mock_append,
            patch.object(synthesizer, "_finalize_m4b") as mock_finalize,
        ):
            synthesizer._create_m4b_from_samples(sample_files)

            # Verify metadata appending was called for each sample
            assert mock_append.call_count == len(sample_files)

            # Verify finalization was called
            mock_finalize.assert_called_once()

    def test_create_m4b_from_samples_empty_list(self, synthesizer):
        """Test M4B creation with empty sample list."""
        synthesizer._create_m4b_from_samples([])
        # Should not raise exception, just log error

    @patch("subprocess.run")
    def test_finalize_m4b_success(self, mock_subprocess, synthesizer):
        """Test successful M4B finalization with FFmpeg."""
        # Mock successful subprocess
        mock_result = Mock()
        mock_result.stdout = "FFmpeg success"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        with patch("pathlib.Path.unlink") as mock_unlink:
            synthesizer._finalize_m4b()

            mock_subprocess.assert_called_once()
            mock_unlink.assert_called_once()

    @patch("subprocess.run")
    def test_finalize_m4b_ffmpeg_error(self, mock_subprocess, synthesizer):
        """Test M4B finalization with FFmpeg error."""
        import subprocess
        mock_subprocess.side_effect = subprocess.CalledProcessError(
            1, "ffmpeg", "FFmpeg failed", "Error output"
        )

        with pytest.raises(subprocess.CalledProcessError):
            synthesizer._finalize_m4b()

    @patch("subprocess.run")
    def test_finalize_m4b_ffmpeg_not_found(self, mock_subprocess, synthesizer):
        """Test M4B finalization when FFmpeg is not found."""
        mock_subprocess.side_effect = FileNotFoundError("FFmpeg not found")

        with pytest.raises(FileNotFoundError):
            synthesizer._finalize_m4b()

    @patch.object(VoiceSamplesSynthesizer, "_get_available_models_and_voices")
    @patch.object(VoiceSamplesSynthesizer, "_create_metadata_file")
    @patch.object(VoiceSamplesSynthesizer, "_create_sample_for_combination")
    @patch.object(VoiceSamplesSynthesizer, "_create_m4b_from_samples")
    @patch("shutil.rmtree")
    def test_synthesize_success(
        self,
        mock_rmtree,
        mock_create_m4b,
        mock_create_sample,
        mock_create_metadata,
        mock_get_models_voices,
        synthesizer,
    ):
        """Test successful full synthesis process."""
        # Mock API responses
        mock_get_models_voices.return_value = (
            ["kokoro", "tts-1"],
            ["af_bella", "af_alloy"]
        )

        # Mock sample creation (return valid paths)
        mock_sample_paths = [
            Path("/tmp/sample_001.mp3"),
            Path("/tmp/sample_002.mp3"),
            Path("/tmp/sample_003.mp3"),
        ]
        mock_create_sample.side_effect = mock_sample_paths

        # Mock file unlinking and path existence
        with (
            patch("pathlib.Path.unlink") as mock_unlink,
            patch("pathlib.Path.exists", return_value=True),  # tmp_dir exists
        ):
            result = synthesizer.synthesize()

            # Verify the process
            assert result == synthesizer.final_m4b_path
            mock_get_models_voices.assert_called_once()
            mock_create_metadata.assert_called_once()

            # Should create max_samples (3) combinations
            assert mock_create_sample.call_count == 3
            mock_create_m4b.assert_called_once()

            # Cleanup should be called
            assert mock_unlink.call_count == len(mock_sample_paths)
            assert mock_rmtree.call_count >= 1

    @patch.object(VoiceSamplesSynthesizer, "_get_available_models_and_voices")
    def test_synthesize_no_models_or_voices(self, mock_get_models_voices, synthesizer):
        """Test synthesis when no models or voices are available."""
        mock_get_models_voices.return_value = ([], [])

        result = synthesizer.synthesize()

        assert result == synthesizer.final_m4b_path
        mock_get_models_voices.assert_called_once()

    @patch.object(VoiceSamplesSynthesizer, "_get_available_models_and_voices")
    @patch.object(VoiceSamplesSynthesizer, "_create_metadata_file")
    @patch.object(VoiceSamplesSynthesizer, "_create_sample_for_combination")
    def test_synthesize_no_successful_samples(
        self,
        mock_create_sample,
        mock_create_metadata,
        mock_get_models_voices,
        synthesizer,
    ):
        """Test synthesis when no samples are created successfully."""
        mock_get_models_voices.return_value = (["kokoro"], ["af_bella"])
        mock_create_sample.return_value = None  # Simulate failure

        result = synthesizer.synthesize()

        assert result == synthesizer.final_m4b_path
        mock_create_sample.assert_called_once()

    def test_max_samples_limiting(self, mock_temp_dir, mock_output_dir):
        """Test that max_samples properly limits the number of combinations."""
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch.object(
                VoiceSamplesSynthesizer, "_get_available_models_and_voices"
            ) as mock_get,
            patch.object(
                VoiceSamplesSynthesizer, "_create_metadata_file"
            ),
            patch.object(
                VoiceSamplesSynthesizer, "_create_sample_for_combination"
            ) as mock_sample,
            patch.object(
                VoiceSamplesSynthesizer, "_create_m4b_from_samples"
            ),
        ):
            # Setup with many models and voices
            mock_get.return_value = (
                ["model1", "model2", "model3"],
                ["voice1", "voice2", "voice3", "voice4"]
            )
            mock_sample.return_value = Path("/tmp/sample.mp3")

            synthesizer = VoiceSamplesSynthesizer(max_samples=2)
            synthesizer.synthesize()

            # Should only create 2 samples despite 3x4=12 possible combinations
            assert mock_sample.call_count == 2

    def test_cleanup_on_deletion(self, mock_temp_dir, mock_output_dir):
        """Test that temporary directory is cleaned up on object deletion."""
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("shutil.rmtree") as mock_rmtree,
        ):
            synthesizer = VoiceSamplesSynthesizer()
            synthesizer.__del__()

            mock_rmtree.assert_called_once_with(synthesizer.tmp_dir)

    def test_cleanup_on_deletion_missing_attribute(
            self, mock_temp_dir, mock_output_dir
        ):
        """Test deletion cleanup when tmp_dir attribute is missing."""
        with patch("pathlib.Path.exists", return_value=True):
            synthesizer = VoiceSamplesSynthesizer()
            delattr(synthesizer, 'tmp_dir')

            # Should not raise exception
            synthesizer.__del__()

    @patch("audify.text_to_speech.AudioSegment.from_mp3")
    def test_create_m4b_from_samples_audio_processing_error(
        self, mock_from_mp3, synthesizer
    ):
        """Test M4B creation when audio processing fails for some files."""
        sample_files = [
            Path("/tmp/sample_001_kokoro_af_bella.mp3"),
            Path("/tmp/sample_002_tts-1_af_alloy.mp3"),
        ]

        # First file succeeds, second fails
        from unittest.mock import MagicMock
        mock_audio = MagicMock()
        mock_audio.__len__ = MagicMock(return_value=5000)
        mock_from_mp3.side_effect = [mock_audio, Exception("Audio processing failed")]

        # Mock combined audio
        with (
            patch("audify.text_to_speech.AudioSegment.empty") as mock_empty,
            patch.object(synthesizer, "_append_chapter_metadata"),
            patch.object(synthesizer, "_finalize_m4b"),
        ):
            mock_combined = MagicMock()
            mock_combined.__iadd__ = MagicMock(return_value=mock_combined)
            mock_combined.__len__ = MagicMock(return_value=5000)
            mock_empty.return_value = mock_combined

            # Should not raise exception, just continue processing
            synthesizer._create_m4b_from_samples(sample_files)

            # Should still call finalize despite one file failing
            synthesizer._finalize_m4b.assert_called_once()
