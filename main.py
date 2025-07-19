import tempfile
from pathlib import Path
from typing import List

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session, select

from audify.constants import DEFAULT_LANGUAGE_LIST
from audify.database import create_db_and_tables, get_session
from audify.job_manager import job_manager
from audify.models import Job, JobCreate, JobRead, JobStatus
from audify.utils import get_file_extension

app = FastAPI(
    title="Audify API",
    description="Text-to-Speech conversion service with job management",
    version="1.0.0",
)

# Create database tables on startup
create_db_and_tables()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def read_root():
    """Serve the main page."""
    return FileResponse("static/index.html")


@app.post("/jobs/", response_model=JobRead, status_code=status.HTTP_201_CREATED)
async def create_job(
    file: UploadFile = File(...),
    language: str = Form(...),
    translate_language: str = Form(None),
    engine: str = Form("kokoro"),
    model_name: str = Form(None),
    save_text: bool = Form(False),
    session: Session = Depends(get_session),
):
    """Create a new text-to-speech job."""

    # Validate file type
    if not file.filename or not file.filename.lower().endswith(('.pdf', '.epub')):
        raise HTTPException(
            status_code=400,
            detail="File must be a PDF or EPUB file"
        )

    # Validate language
    if language not in DEFAULT_LANGUAGE_LIST:
        raise HTTPException(
            status_code=400,
            detail=f"Language must be one of: {DEFAULT_LANGUAGE_LIST}"
        )

    if translate_language and translate_language not in DEFAULT_LANGUAGE_LIST:
        raise HTTPException(
            status_code=400,
            detail=f"Translation language must be one of: {DEFAULT_LANGUAGE_LIST}"
        )

    # Save uploaded file temporarily
    temp_dir = Path(tempfile.mkdtemp(prefix="audify_upload_"))
    file_path = temp_dir / file.filename

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Get file extension
    file_extension = get_file_extension(str(file_path))

    # Create job in database
    job_data = JobCreate(
        filename=file.filename,
        file_extension=file_extension,
        language=language,
        translate_language=translate_language if translate_language else None,
        engine=engine,
        model_name=model_name if model_name else None,
        save_text=save_text,
    )

    job = Job.model_validate(job_data)
    session.add(job)
    session.commit()
    session.refresh(job)

    # Start job processing in background
    job_manager.start_job(job.id, file_path)

    return job


@app.get("/jobs/", response_model=List[JobRead])
def get_jobs(
    status_filter: JobStatus = None,
    session: Session = Depends(get_session),
):
    """Get all jobs, optionally filtered by status."""
    query = select(Job)
    if status_filter:
        query = query.where(Job.status == status_filter)

    jobs = session.exec(query).all()
    return jobs


@app.get("/jobs/{job_id}", response_model=JobRead)
def get_job(job_id: int, session: Session = Depends(get_session)):
    """Get a specific job by ID."""
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: int, session: Session = Depends(get_session)):
    """Cancel a running job."""
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status not in [JobStatus.PENDING, JobStatus.RUNNING]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {job.status}"
        )

    success = job_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Failed to cancel job"
        )

    return {"message": "Job cancelled successfully"}


@app.get("/jobs/{job_id}/download")
def download_result(job_id: int, session: Session = Depends(get_session)):
    """Download the result file for a completed job."""
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed. Current status: {job.status}"
        )

    if not job.output_path or not Path(job.output_path).exists():
        raise HTTPException(
            status_code=404,
            detail="Output file not found"
        )

    return FileResponse(
        path=job.output_path,
        filename=f"{Path(job.filename).stem}_audiobook.mp3",
        media_type="audio/mpeg",
    )


@app.delete("/jobs/{job_id}")
def delete_job(job_id: int, session: Session = Depends(get_session)):
    """Delete a job and its associated files."""
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Cancel if running
    if job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
        job_manager.cancel_job(job_id)

    # Delete output file if it exists
    if job.output_path and Path(job.output_path).exists():
        try:
            Path(job.output_path).unlink()
        except Exception:
            pass  # Ignore file deletion errors

    # Delete from database
    session.delete(job)
    session.commit()

    return {"message": "Job deleted successfully"}


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/languages")
def get_supported_languages():
    """Get list of supported languages."""
    return {"languages": DEFAULT_LANGUAGE_LIST}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
