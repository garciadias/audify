"""
Job management system for handling multiple file processing with progress monitoring
and cancellation capabilities.
"""
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Status enum for jobs."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobProgress:
    """Progress information for a job."""
    current_step: int = 0
    total_steps: int = 100
    current_operation: str = "Initializing..."
    percentage: float = 0.0
    estimated_time_remaining: Optional[float] = None

    def update(self, current_step: int = None, total_steps: int = None,
               current_operation: str = None, percentage: float = None):
        """Update progress information."""
        if current_step is not None:
            self.current_step = current_step
        if total_steps is not None:
            self.total_steps = total_steps
        if current_operation is not None:
            self.current_operation = current_operation
        if percentage is not None:
            self.percentage = percentage
        elif self.total_steps > 0:
            self.percentage = min((self.current_step / self.total_steps) * 100, 100.0)


@dataclass
class Job:
    """Represents a processing job."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_path: Path = None
    file_name: str = ""
    status: JobStatus = JobStatus.PENDING
    progress: JobProgress = field(default_factory=JobProgress)
    result: Optional[Path] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    cancelled: threading.Event = field(default_factory=threading.Event)

    # Job parameters
    language: str = "en"
    translate_language: Optional[str] = None
    engine: str = "kokoro"
    model: Optional[str] = None
    save_text: bool = False

    @property
    def duration(self) -> Optional[float]:
        """Get job duration in seconds."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return time.time() - self.started_at
        return None

    def cancel(self):
        """Cancel the job."""
        self.cancelled.set()
        if self.status in [JobStatus.RUNNING, JobStatus.PENDING]:
            self.status = JobStatus.CANCELLED


class JobManager:
    """Manages multiple processing jobs with progress tracking and cancellation."""

    def __init__(self, max_concurrent_jobs: int = 2):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.jobs: Dict[str, Job] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_jobs)
        self._lock = threading.Lock()

    def create_job(self, file_path: Path, file_name: str, **kwargs) -> str:
        """Create a new job and return its ID."""
        job = Job(
            file_path=file_path,
            file_name=file_name,
            language=kwargs.get('language', 'en'),
            translate_language=kwargs.get('translate_language'),
            engine=kwargs.get('engine', 'kokoro'),
            model=kwargs.get('model'),
            save_text=kwargs.get('save_text', False)
        )

        with self._lock:
            self.jobs[job.id] = job

        logger.info(f"Created job {job.id} for file {file_name}")
        return job.id

    def submit_job(self, job_id: str, processor_func: Callable,
                   *args, **kwargs) -> None:
        """Submit a job for processing."""
        with self._lock:
            if job_id not in self.jobs:
                raise ValueError(f"Job {job_id} not found")

            job = self.jobs[job_id]
            if job.status != JobStatus.PENDING:
                raise ValueError(f"Job {job_id} is not in pending status")

        # Submit job to executor
        future = self.executor.submit(
            self._execute_job, job_id, processor_func, *args, **kwargs
        )
        return future

    def _execute_job(self, job_id: str, processor_func: Callable,
                     *args, **kwargs) -> None:
        """Execute a job with error handling and progress tracking."""
        job = self.jobs[job_id]

        try:
            job.status = JobStatus.RUNNING
            job.started_at = time.time()

            logger.info(f"Starting job {job_id} for file {job.file_name}")

            # Add progress callback to kwargs if not present
            if 'progress_callback' not in kwargs:
                def progress_callback(**progress_kwargs):
                    return self._update_job_progress(job_id, **progress_kwargs)
                kwargs['progress_callback'] = progress_callback

            # Add cancellation check to kwargs
            if 'cancellation_check' not in kwargs:
                kwargs['cancellation_check'] = lambda: job.cancelled.is_set()

            result = processor_func(job, *args, **kwargs)

            if job.cancelled.is_set():
                job.status = JobStatus.CANCELLED
                logger.info(f"Job {job_id} was cancelled")
            else:
                job.status = JobStatus.COMPLETED
                job.result = result
                job.completed_at = time.time()
                logger.info(f"Job {job_id} completed successfully in {job.duration:.2f}s")

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = time.time()
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)

    def _update_job_progress(self, job_id: str, **kwargs) -> None:
        """Update job progress."""
        if job_id in self.jobs:
            self.jobs[job_id].progress.update(**kwargs)

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        return self.jobs.get(job_id)

    def get_all_jobs(self) -> List[Job]:
        """Get all jobs."""
        return list(self.jobs.values())

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job.cancel()
            logger.info(f"Job {job_id} cancellation requested")
            return True
        return False

    def cancel_all_jobs(self) -> int:
        """Cancel all running jobs."""
        cancelled_count = 0
        for job in self.jobs.values():
            if job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
                job.cancel()
                cancelled_count += 1

        logger.info(f"Cancelled {cancelled_count} jobs")
        return cancelled_count

    def cleanup_completed_jobs(self, max_age_seconds: float = 3600) -> int:
        """Remove completed jobs older than max_age_seconds."""
        current_time = time.time()
        to_remove = []

        for job_id, job in self.jobs.items():
            if (job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED] and
                job.completed_at and (current_time - job.completed_at) > max_age_seconds):
                to_remove.append(job_id)

        with self._lock:
            for job_id in to_remove:
                del self.jobs[job_id]

        logger.info(f"Cleaned up {len(to_remove)} old jobs")
        return len(to_remove)

    def get_job_stats(self) -> Dict[str, int]:
        """Get statistics about jobs."""
        stats = {status.value: 0 for status in JobStatus}
        for job in self.jobs.values():
            stats[job.status.value] += 1
        return stats

    def shutdown(self) -> None:
        """Shutdown the job manager."""
        logger.info("Shutting down job manager...")
        self.cancel_all_jobs()
        self.executor.shutdown(wait=True)
