import tempfile
import time
from pathlib import Path

import streamlit as st

from audify.constants import DEFAULT_LANGUAGE_LIST
from audify.job_manager import JobManager, JobStatus
from audify.job_processor import process_file_job

# Initialize session state for job manager
if 'job_manager' not in st.session_state:
    st.session_state.job_manager = JobManager(max_concurrent_jobs=2)

if 'uploaded_files_data' not in st.session_state:
    st.session_state.uploaded_files_data = {}

st.set_page_config(page_title="Audify", page_icon="🎙️", layout="wide")

st.title("Audify: Multi-File Text to Speech")

# --- Sidebar for options ---
st.sidebar.header("Configuration")

# File Upload - Multiple files
uploaded_files = st.sidebar.file_uploader(
    "Upload files (.pdf or .epub)",
    type=["pdf", "epub"],
    accept_multiple_files=True,
    help="You can upload multiple files to process them in batch"
)

# Language Selection
language = st.sidebar.selectbox(
    "Language of the text",
    options=DEFAULT_LANGUAGE_LIST,
    index=DEFAULT_LANGUAGE_LIST.index("en"),
    help="Select the language of the input text.",
)

# Translation Selection
translate_option = st.sidebar.checkbox(
    "Translate the text?",
    help="Check this box to translate the text before synthesis.",
)
translate_language = None
if translate_option:
    translate_language = st.sidebar.selectbox(
        "Translate to",
        options=DEFAULT_LANGUAGE_LIST,
        index=DEFAULT_LANGUAGE_LIST.index("en"),
        help="Select the language to translate the text to.",
    )

# TTS Engine
engine = st.sidebar.selectbox(
    "TTS Engine",
    options=["kokoro", "tts_models"],
    index=0,
    help="Select the TTS engine to use.",
)

# if engine == "tts_models", allow for tts model selection
model = None
if engine == "tts_models":
    # User needs to agree with terms and conditions of the xtts models
    st.sidebar.markdown(
        "By using the `tts_models` engine, you agree to the terms and "
        "conditions of the XTTS models."
    )
    st.sidebar.markdown(
        "You can find the terms and conditions [here](https://coqui.ai/cpml)."
    )
    st.sidebar.markdown(
        "The `tts_models` engine uses pre-trained models from the TTS library. "
        "You can specify the model name in the input below."
    )

    # Model Selection - only show for tts_models engine
    if engine == "tts_models":
        model_options = [
            "tts_models/multilingual/multi-dataset/xtts_v2",
            "tts_models/en/ljspeech/tacotron2-DDC",
            "tts_models/en/ljspeech/glow-tts",
            "tts_models/en/ljspeech/speedy-speech",
        ]
        
        model = st.sidebar.selectbox(
            "XTTS Model",
            options=model_options,
            index=0,
            help="Select a pre-trained TTS model. XTTS_v2 is recommended.",
        )
    else:
        model = None  # Not used for Kokoro engine

# Other options
save_text = st.sidebar.checkbox(
    "Save extracted text", help="Save the extracted text to a file."
)

# Job control buttons in sidebar
st.sidebar.header("Job Control")
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("Pause All", help="Pause all running jobs"):
        cancelled_count = st.session_state.job_manager.cancel_all_jobs()
        st.sidebar.success(f"Paused {cancelled_count} jobs")

with col2:
    if st.button("Clear Completed", help="Clear completed jobs from the list"):
        cleaned_count = st.session_state.job_manager.cleanup_completed_jobs(max_age_seconds=0)
        st.sidebar.success(f"Cleared {cleaned_count} jobs")

# --- Main Area ---
st.header("File Processing")

# Handle uploaded files
if uploaded_files:
    # Store uploaded files in session state to persist across reruns
    for uploaded_file in uploaded_files:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if file_key not in st.session_state.uploaded_files_data:
            # Create temporary file
            temp_dir = Path(tempfile.mkdtemp(prefix="audify_upload_"))
            temp_file_path = temp_dir / uploaded_file.name

            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.session_state.uploaded_files_data[file_key] = {
                'name': uploaded_file.name,
                'path': temp_file_path,
                'size': uploaded_file.size
            }

    # Display uploaded files
    st.subheader(f"📁 Uploaded Files ({len(st.session_state.uploaded_files_data)})")

    files_to_process = []
    for file_key, file_data in st.session_state.uploaded_files_data.items():
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

        with col1:
            st.write(f"📄 {file_data['name']} ({file_data['size']} bytes)")

        with col2:
            if st.checkbox("Process", key=f"process_{file_key}"):
                files_to_process.append(file_data)

        with col3:
            # Check if there's already a job for this file
            existing_job = None
            for job in st.session_state.job_manager.get_all_jobs():
                if job.file_path == file_data['path']:
                    existing_job = job
                    break

            if existing_job:
                status_color = {
                    JobStatus.PENDING: "🟡",
                    JobStatus.RUNNING: "🔵",
                    JobStatus.COMPLETED: "🟢",
                    JobStatus.FAILED: "🔴",
                    JobStatus.CANCELLED: "⚪"
                }.get(existing_job.status, "❓")
                st.write(f"{status_color} {existing_job.status.value}")
            else:
                st.write("⚪ Not queued")

        with col4:
            if existing_job and existing_job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
                if st.button("Cancel", key=f"cancel_{file_key}"):
                    st.session_state.job_manager.cancel_job(existing_job.id)
                    st.rerun()

    # Process selected files button
    if files_to_process:
        if st.button(f"🚀 Start Processing ({len(files_to_process)} files)", type="primary"):
            for file_data in files_to_process:
                # Create job
                job_id = st.session_state.job_manager.create_job(
                    file_path=file_data['path'],
                    file_name=file_data['name'],
                    language=language,
                    translate_language=translate_language,
                    engine=engine,
                    model=model,
                    save_text=save_text
                )

                # Submit job for processing
                st.session_state.job_manager.submit_job(job_id, process_file_job)

            st.success(f"Started processing {len(files_to_process)} files!")
            st.rerun()

# --- Job Status Display ---
jobs = st.session_state.job_manager.get_all_jobs()

if jobs:
    st.header("📊 Job Status")

    # Job statistics
    stats = st.session_state.job_manager.get_job_stats()
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Pending", stats.get('pending', 0))
    with col2:
        st.metric("Running", stats.get('running', 0))
    with col3:
        st.metric("Completed", stats.get('completed', 0))
    with col4:
        st.metric("Failed", stats.get('failed', 0))
    with col5:
        st.metric("Cancelled", stats.get('cancelled', 0))

    # Detailed job list
    for job in sorted(jobs, key=lambda x: x.created_at, reverse=True):
        with st.expander(f"{job.file_name} - {job.status.value.upper()}",
                        expanded=(job.status == JobStatus.RUNNING)):

            col1, col2 = st.columns([2, 1])

            with col1:
                st.write(f"**File:** {job.file_name}")
                st.write(f"**Language:** {job.language}")
                if job.translate_language:
                    st.write(f"**Translate to:** {job.translate_language}")
                st.write(f"**Engine:** {job.engine}")

                # Progress bar for running jobs
                if job.status == JobStatus.RUNNING:
                    progress_text = job.progress.current_operation
                    progress_value = job.progress.percentage / 100.0

                    st.write(f"**Progress:** {progress_text}")
                    st.progress(progress_value)
                    st.write(f"{job.progress.percentage:.1f}% complete")

                # Show error for failed jobs
                elif job.status == JobStatus.FAILED and job.error:
                    st.error(f"**Error:** {job.error}")

                # Show result for completed jobs
                elif job.status == JobStatus.COMPLETED and job.result:
                    st.success("✅ **Synthesis completed successfully!**")

                    # Download button
                    if job.result.exists():
                        with open(job.result, "rb") as f:
                            st.download_button(
                                label="📥 Download Audiobook",
                                data=f.read(),
                                file_name=job.result.name,
                                mime="audio/mpeg",
                                key=f"download_{job.id}"
                            )
                    else:
                        st.warning("Output file no longer exists")

            with col2:
                # Job timing information
                if job.duration:
                    st.write(f"**Duration:** {job.duration:.1f}s")

                created_time = time.strftime("%H:%M:%S", time.localtime(job.created_at))
                st.write(f"**Created:** {created_time}")

                if job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
                    if st.button("Cancel Job", key=f"cancel_job_{job.id}"):
                        st.session_state.job_manager.cancel_job(job.id)
                        st.rerun()

# Auto-refresh for running jobs
if any(job.status == JobStatus.RUNNING for job in jobs):
    time.sleep(2)
    st.rerun()

# Information section
st.header("ℹ️ Information")
st.write("""
### How to use:
1. **Upload files**: Select one or more PDF or EPUB files using the file uploader in the sidebar
2. **Configure settings**: Choose language, translation options, and TTS engine
3. **Select files**: Check the files you want to process in the main area
4. **Start processing**: Click the "Start Processing" button to begin synthesis
5. **Monitor progress**: Watch the job status and progress bars
6. **Download results**: Once complete, download the generated audiobooks

### Features:
- **Batch processing**: Process multiple files simultaneously
- **Progress monitoring**: Real-time progress updates for each job
- **Job cancellation**: Cancel individual jobs or all jobs at any time
- **Job management**: View detailed status and manage completed jobs
- **Auto-refresh**: The interface automatically updates to show current status

### Supported formats:
- **PDF files**: Text-based PDFs will be processed
- **EPUB files**: Standard EPUB ebooks are supported

### TTS Engines:
- **Kokoro**: Fast, lightweight TTS engine (recommended)
- **TTS Models**: More advanced models with better quality (slower)
""")

# Cleanup session state on app restart
if st.button("🗑️ Clear All Data", help="Clear all uploaded files and jobs"):
    # Cleanup temporary files
    for file_data in st.session_state.uploaded_files_data.values():
        try:
            if file_data['path'].exists():
                file_data['path'].parent.rmdir()
        except:
            pass

    # Clear session state
    st.session_state.uploaded_files_data = {}
    st.session_state.job_manager.cancel_all_jobs()
    st.session_state.job_manager = JobManager(max_concurrent_jobs=2)
    st.success("All data cleared!")
    st.rerun()
