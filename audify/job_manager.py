import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

from sqlmodel import Session

from audify.database import engine
from audify.models import Job, JobStatus
from audify.text_to_speech import EpubSynthesizer, PdfSynthesizer

logger = logging.getLogger(__name__)


class JobManager:
    """Manages background job processing."""

    def __init__(self):
        self.running_jobs: Dict[int, threading.Thread] = {}
        self.cancelled_jobs: set = set()

    def start_job(self, job_id: int, file_path: Path) -> None:
        """Start processing a job in a background thread."""
        if job_id in self.running_jobs:
            logger.warning(f"Job {job_id} is already running")
            return

        thread = threading.Thread(target=self._process_job, args=(job_id, file_path))
        thread.daemon = True
        self.running_jobs[job_id] = thread
        thread.start()

    def cancel_job(self, job_id: int) -> bool:
        """Cancel a running job."""
        if job_id in self.running_jobs:
            self.cancelled_jobs.add(job_id)
            # Update job status in database
            with Session(engine) as session:
                job = session.get(Job, job_id)
                if job:
                    job.status = JobStatus.CANCELLED
                    job.updated_at = datetime.utcnow()
                    session.add(job)
                    session.commit()
            return True
        return False

    def is_job_running(self, job_id: int) -> bool:
        """Check if a job is currently running."""
        return job_id in self.running_jobs and self.running_jobs[job_id].is_alive()

    def _process_job(self, job_id: int, file_path: Path) -> None:
        """Process a job (runs in background thread)."""
        try:
            with Session(engine) as session:
                job = session.get(Job, job_id)
                if not job:
                    logger.error(f"Job {job_id} not found")
                    return

                # Check if job was cancelled before starting
                if job_id in self.cancelled_jobs:
                    logger.info(f"Job {job_id} was cancelled before processing")
                    self.cancelled_jobs.discard(job_id)
                    return

                # Update job status to running
                job.status = JobStatus.RUNNING
                job.started_at = datetime.utcnow()
                job.updated_at = datetime.utcnow()
                job.progress = 0.0
                session.add(job)
                session.commit()

                logger.info(f"Starting job {job_id}: {job.filename}")

                # Create appropriate synthesizer
                synthesizer = None
                if job.file_extension == ".epub":
                    synthesizer = EpubSynthesizer(
                        str(file_path),
                        language=job.language,
                        model_name=job.model_name if job.engine == "tts_models" else None,
                        translate=job.translate_language,
                        save_text=job.save_text,
                        engine=job.engine,
                        confirm=False,
                    )
                elif job.file_extension == ".pdf":
                    synthesizer = PdfSynthesizer(
                        str(file_path),
                        language=job.language,
                        model_name=job.model_name if job.engine == "tts_models" else None,
                        translate=job.translate_language,
                        save_text=job.save_text,
                        engine=job.engine,
                    )

                if not synthesizer:
                    raise ValueError(f"Unsupported file format: {job.file_extension}")

                # Create progress tracking wrapper
                original_synthesize = synthesizer.synthesize

                def progress_wrapper():
                    # Simple progress simulation - in a real implementation,
                    # you'd want to modify the synthesizer to report actual progress
                    for i in range(1, 11):
                        if job_id in self.cancelled_jobs:
                            raise InterruptedError("Job was cancelled")
                        time.sleep(0.1)  # Simulate work
                        with Session(engine) as progress_session:
                            current_job = progress_session.get(Job, job_id)
                            if current_job:
                                current_job.progress = i * 10.0
                                current_job.updated_at = datetime.utcnow()
                                progress_session.add(current_job)
                                progress_session.commit()
                    return original_synthesize()

                # Run synthesis
                output_path = progress_wrapper()

                # Check one more time if cancelled
                if job_id in self.cancelled_jobs:
                    logger.info(f"Job {job_id} was cancelled during processing")
                    self.cancelled_jobs.discard(job_id)
                    return

                # Update job as completed
                job.status = JobStatus.COMPLETED
                job.progress = 100.0
                job.output_path = str(output_path)
                job.completed_at = datetime.utcnow()
                job.updated_at = datetime.utcnow()
                session.add(job)
                session.commit()

                logger.info(f"Job {job_id} completed successfully")

        except InterruptedError:
            logger.info(f"Job {job_id} was cancelled")
            self.cancelled_jobs.discard(job_id)
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            # Update job as failed
            with Session(engine) as session:
                job = session.get(Job, job_id)
                if job:
                    job.status = JobStatus.FAILED
                    job.error_message = str(e)
                    job.updated_at = datetime.utcnow()
                    session.add(job)
                    session.commit()
        finally:
            # Clean up
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
            self.cancelled_jobs.discard(job_id)


# Global job manager instance
job_manager = JobManager()
