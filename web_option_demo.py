#!/usr/bin/env python3
"""
FastAPI Alternative Demo for Audify Multi-File Processing

This demonstrates how the same functionality could be implemented
with FastAPI to solve the Streamlit limitations:
- Persistent jobs across browser refreshes
- Real-time progress updates via WebSockets
- Proper job cancellation
- Background task management
"""

import json
import tempfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# Use our existing job management system
from audify.job_manager import JobManager
from audify.job_processor import process_file_job

app = FastAPI(title="Audify Multi-File Processor")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global job manager (persists across requests)
job_manager = JobManager(max_concurrent_jobs=2)

# WebSocket connections for real-time updates
active_connections: List[WebSocket] = []


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast_job_update(self, job_id: str):
        """Broadcast job updates to all connected clients."""
        job = job_manager.get_job(job_id)
        if job:
            message = {
                "type": "job_update",
                "job": {
                    "id": job.id,
                    "file_name": job.file_name,
                    "status": job.status.value,
                    "progress": job.progress.dict() if job.progress else None,
                    "error": job.error,
                    "created_at": job.created_at.isoformat(),
                }
            }

            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception as e:
                    print(f"Error sending WebSocket message: {e}")
                    disconnected.append(connection)

            # Remove disconnected clients
            for conn in disconnected:
                self.active_connections.remove(conn)


manager = ConnectionManager()


@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    """Serve the main application page."""
    return open("static/index.html").read()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time job updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep the connection alive
            await websocket.receive_text()
    except Exception as e:
        manager.disconnect(websocket)
        print(f"WebSocket error: {e}")


@app.get("/jobs")
async def get_jobs():
    """Get all current jobs."""
    jobs = job_manager.get_all_jobs()
    return [
        {
            "id": job.id,
            "file_name": job.file_name,
            "status": job.status.value,
            "progress": job.progress.dict() if job.progress else None,
            "error": job.error,
            "created_at": job.created_at.isoformat(),
        }
        for job in jobs
    ]


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), language: str = "en"):
    """Upload and start processing a file."""

    # Save uploaded file temporarily
    temp_dir = Path(tempfile.gettempdir()) / "audify_uploads"
    temp_dir.mkdir(exist_ok=True)

    file_path = temp_dir / file.filename

    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    # Create and start job
    job_id = job_manager.create_job(
        file_path=file_path,
        file_name=file.filename,
        language=language
    )

    # Start processing with progress callback
    async def progress_callback(job_id: str):
        await manager.broadcast_job_update(job_id)

    job_manager.start_job(
        job_id,
        process_file_job,
        progress_callback=progress_callback
    )

    return {"job_id": job_id, "message": f"Started processing {file.filename}"}


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a specific job."""
    success = job_manager.cancel_job(job_id)
    if success:
        await manager.broadcast_job_update(job_id)
        return {"message": f"Job {job_id} cancelled"}
    else:
        raise HTTPException(status_code=404, detail="Job not found")


@app.post("/jobs/cancel-all")
async def cancel_all_jobs():
    """Cancel all running/pending jobs."""
    cancelled_count = job_manager.cancel_all_jobs()

    # Broadcast updates for all jobs
    for job in job_manager.get_all_jobs():
        await manager.broadcast_job_update(job.id)

    return {"message": f"Cancelled {cancelled_count} jobs"}


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI server...")
    print("📱 Open http://localhost:8000 in your browser")
    print("✨ Features:")
    print("  - Jobs persist across browser refreshes")
    print("  - Real-time progress updates")
    print("  - Proper job cancellation")
    print("  - Multi-file upload")

    uvicorn.run(app, host="0.0.0.0", port=8000)
