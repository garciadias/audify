#!/usr/bin/env python3
"""
Flask + HTMX Alternative for Audify Multi-File Processing

This shows a simpler alternative using Flask + HTMX that solves 
the Streamlit limitations without requiring a complex frontend framework.

Key advantages:
- Jobs persist across browser refreshes (stored server-side)
- Real-time progress updates via Server-Sent Events
- Proper job cancellation
- Simple HTML/JavaScript frontend
- Easy to understand and maintain
"""

import json
import tempfile
from pathlib import Path

from flask import Flask, render_template_string, request

# Use our existing job management system
from audify.job_manager import JobManager
from audify.job_processor import process_file_job

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Global job manager (persists across requests)
job_manager = JobManager(max_concurrent_jobs=2)

# Store for Server-Sent Events clients
sse_clients = []


def send_sse_update(job_id: str):
    """Send Server-Sent Events update to all connected clients."""
    job = job_manager.get_job(job_id)
    if job:
        data = {
            "id": job.id,
            "file_name": job.file_name,
            "status": job.status.value,
            "progress": job.progress.dict() if job.progress else None,
            "error": job.error,
            "created_at": job.created_at.isoformat(),
        }

        # Send to all connected SSE clients
        message = f"data: {json.dumps(data)}\n\n"
        for client_queue in sse_clients:
            try:
                client_queue.put(message)
            except:
                pass


@app.route('/')
def index():
    """Main application page."""
    return render_template_string("static/index.html")


@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file upload and start processing."""
    files = request.files.getlist('files')
    language = request.form.get('language', 'en')

    if not files:
        return "No files uploaded", 400

    job_cards = []

    for file in files:
        if file.filename == '':
            continue

        # Save uploaded file temporarily
        temp_dir = Path(tempfile.gettempdir()) / "audify_uploads"
        temp_dir.mkdir(exist_ok=True)

        file_path = temp_dir / file.filename
        file.save(str(file_path))

        # Create and start job
        job_id = job_manager.create_job(
            file_path=file_path,
            file_name=file.filename,
            language=language
        )

        # Start processing with progress callback
        def progress_callback(job_id=job_id):
            send_sse_update(job_id)

        job_manager.start_job(
            job_id,
            process_file_job,
            progress_callback=progress_callback
        )

        # Return the job card HTML for immediate display
        job = job_manager.get_job(job_id)
        job_cards.append(render_job_card(job))

    return ''.join(job_cards)


@app.route('/jobs')
def get_jobs():
    """Get all jobs as HTML."""
    jobs = job_manager.get_all_jobs()

    if not jobs:
        return '''
        <div style="text-align: center; color: #888; padding: 20px;">
            No jobs yet. Upload some files to get started!
        </div>'''

    return ''.join([render_job_card(job) for job in jobs])


@app.route('/stats')
def get_stats():
    """Get job statistics."""
    stats = job_manager.get_job_stats()
    return f"""
    <div class="stats">
        <div class="stat-item">
            <div class="stat-number">{stats.get('total', 0)}</div>
            <div class="stat-label">Total Jobs</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">{stats.get('running', 0)}</div>
            <div class="stat-label">Running</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">{stats.get('completed', 0)}</div>
            <div class="stat-label">Completed</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">{stats.get('failed', 0)}</div>
            <div class="stat-label">Failed</div>
        </div>
    </div>
    """


@app.route('/cancel/<job_id>', methods=['POST'])
def cancel_job(job_id):
    """Cancel a specific job."""
    success = job_manager.cancel_job(job_id)
    if success:
        job = job_manager.get_job(job_id)
        return render_job_card(job)
    return "Job not found", 404


@app.route('/cancel-all', methods=['POST'])
def cancel_all():
    """Cancel all jobs."""
    job_manager.cancel_all_jobs()
    return get_jobs()  # Return updated job list


def render_job_card(job):
    """Render a single job card as HTML."""
    if not job:
        return ""

    progress = job.progress or type('obj', (object,), {
        'percentage': 0,
        'current_step': 'Waiting...',
        'total_steps': 0
    })()

    progress_percent = getattr(progress, 'percentage', 0)
    current_step = getattr(progress, 'current_step', 'Waiting...')

    # Format creation time
    time_str = job.created_at.strftime("%H:%M:%S")

    cancel_button = ""
    if job.status.value in ['PENDING', 'RUNNING']:
        cancel_button = f'''
            <button class="btn-danger btn-small"
                    hx-post="/cancel/{job.id}"
                    hx-target="#job-{job.id}"
                    hx-swap="outerHTML">
                ❌ Cancel
            </button>
        '''

    error_display = ""
    if job.error:
        error_display = f'''
        <div style="color: #f44336; margin-top: 10px;">
            ❌ Error: {job.error}
        </div>'''

    return f'''
    <div id="job-{job.id}" class="job-card">
        <div class="job-header">
            <div class="job-info">
                <h4>📄 {job.file_name}</h4>
                    <span class="job-status status-{job.status.value}">
                        {job.status.value}
                    </span>
                <span style="margin-left: 10px; color: #ccc;">⏰ {time_str}</span>
            </div>
            <div>
                {cancel_button}
            </div>
        </div>

        <div class="progress-container">
            <div style="font-size: 0.9em; color: #ccc; margin-bottom: 5px;">
                {current_step}
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {progress_percent}%"></div>
            </div>
            <div style="font-size: 0.85em; color: #ccc; margin-top: 5px;">
                {progress_percent:.1f}% complete
            </div>
        </div>

        {error_display}
    </div>
    '''


if __name__ == "__main__":
    print("🚀 Starting Flask + HTMX server...")
    print("📱 Open http://localhost:5000 in your browser")
    print("✨ Features:")
    print("  - Jobs persist across browser refreshes")
    print("  - Real-time progress updates (auto-refresh)")
    print("  - Proper job cancellation")
    print("  - Multi-file upload")
    print("  - No complex frontend framework needed!")

    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
