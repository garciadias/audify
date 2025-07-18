"""
Job processor for handling file synthesis with the enhanced synthesizers.
"""
import logging
import tempfile
from pathlib import Path
from typing import Optional

from audify.enhanced_synthesizers import EnhancedEpubSynthesizer, EnhancedPdfSynthesizer
from audify.job_manager import Job
from audify.utils import get_file_extension

logger = logging.getLogger(__name__)


def process_file_job(
    job: Job,
    progress_callback: Optional[callable] = None,
    cancellation_check: Optional[callable] = None
) -> Path:
    """
    Process a file synthesis job.
    
    Args:
        job: The job containing file and synthesis parameters
        progress_callback: Function to call for progress updates
        cancellation_check: Function to check if job should be cancelled
        
    Returns:
        Path to the generated audio file
        
    Raises:
        InterruptedError: If the job was cancelled
        ValueError: If the file format is not supported
        Exception: If synthesis fails
    """
    file_path = job.file_path
    file_extension = get_file_extension(str(file_path))

    logger.info(f"Processing {file_extension} file: {job.file_name}")

    # Create temporary directory for this job
    with tempfile.TemporaryDirectory(prefix=f"audify_job_{job.id}_") as temp_dir:
        # Copy file to temp directory to avoid conflicts
        temp_file_path = Path(temp_dir) / file_path.name
        import shutil
        shutil.copy2(file_path, temp_file_path)

        synthesizer = None

        try:
            if file_extension == ".epub":
                synthesizer = EnhancedEpubSynthesizer(
                    path=str(temp_file_path),
                    language=job.language,
                    model_name=job.model if job.engine == "tts_models" else None,
                    translate=job.translate_language,
                    save_text=job.save_text,
                    engine=job.engine,
                    confirm=False,  # No confirmation needed in automated processing
                    progress_callback=progress_callback,
                    cancellation_check=cancellation_check,
                )
            elif file_extension == ".pdf":
                synthesizer = EnhancedPdfSynthesizer(
                    path=str(temp_file_path),
                    language=job.language,
                    model_name=job.model if job.engine == "tts_models" else None,
                    translate=job.translate_language,
                    save_text=job.save_text,
                    engine=job.engine,
                    progress_callback=progress_callback,
                    cancellation_check=cancellation_check,
                )
            else:
                raise ValueError(f"Unsupported file format: {file_extension}. "
                               "Please upload a .pdf or .epub file.")

            # Perform synthesis
            output_path = synthesizer.synthesize()

            # Verify output exists
            if not output_path.exists():
                raise RuntimeError(f"Synthesis completed but output file not found: {output_path}")

            logger.info(f"Job {job.id} completed successfully. Output: {output_path}")
            return output_path

        except InterruptedError:
            logger.info(f"Job {job.id} was cancelled during synthesis")
            raise
        except Exception as e:
            logger.error(f"Job {job.id} failed during synthesis: {e}", exc_info=True)
            raise
        finally:
            # Cleanup synthesizer resources if needed
            if synthesizer and hasattr(synthesizer, 'tmp_dir_context'):
                try:
                    synthesizer.tmp_dir_context.cleanup()
                except Exception as cleanup_error:
                    logger.warning(f"Error during synthesizer cleanup: {cleanup_error}")
