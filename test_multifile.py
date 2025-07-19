"""
Test script to demonstrate the multi-file processing functionality.
"""
import tempfile
from pathlib import Path

from audify.job_manager import JobManager


def create_sample_files():
    """Create sample test files for demonstration."""
    temp_dir = Path(tempfile.mkdtemp(prefix="audify_test_"))

    # Create a simple text file (we'll treat it as a PDF for testing)
    sample_text = """
    Chapter 1: Introduction

    This is a sample text for testing the multi-file processing functionality
    of the Audify application. This text will be converted to speech using
    the text-to-speech engine.

    Chapter 2: Features

    The enhanced Audify application now supports:
    - Multiple file processing
    - Real-time progress monitoring
    - Job cancellation capabilities
    - Batch operations with different settings

    This concludes our sample text for testing purposes.
    """

    sample_file = temp_dir / "sample_test.txt"
    with open(sample_file, "w") as f:
        f.write(sample_text)

    return temp_dir, [sample_file]


def test_job_manager():
    """Test the job management system."""
    print("🧪 Testing Job Manager Functionality...")

    # Create job manager
    job_manager = JobManager(max_concurrent_jobs=2)

    # Create sample files
    temp_dir, sample_files = create_sample_files()

    print(f"📁 Created {len(sample_files)} sample files in {temp_dir}")

    # Create jobs
    job_ids = []
    for i, file_path in enumerate(sample_files):
        job_id = job_manager.create_job(
            file_path=file_path,
            file_name=f"test_file_{i + 1}.txt",
            language="en",
            engine="kokoro"
        )
        job_ids.append(job_id)
        print(f"✅ Created job {job_id} for {file_path.name}")

    # Display job statistics
    stats = job_manager.get_job_stats()
    print(f"📊 Job Statistics: {stats}")

    # Get job details
    for job_id in job_ids:
        job = job_manager.get_job(job_id)
        print(f"🔍 Job {job_id}: Status={job.status.value}, File={job.file_name}")

    # Test cancellation
    if job_ids:
        first_job_id = job_ids[0]
        job_manager.cancel_job(first_job_id)
        cancelled_job = job_manager.get_job(first_job_id)
        print(f"❌ Cancelled job {first_job_id}: Status={cancelled_job.status.value}")

    # Cleanup
    job_manager.shutdown()

    # Clean up temp files
    for file_path in sample_files:
        file_path.unlink()
    temp_dir.rmdir()

    print("✨ Job Manager test completed successfully!")


if __name__ == "__main__":
    test_job_manager()
