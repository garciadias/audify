#!/usr/bin/env python3
"""
Simple CLI tool to test the Audify API
"""
import argparse
import time
from pathlib import Path

import requests

API_BASE = "http://localhost:8083"


def create_job(file_path: str, language: str = "en", engine: str = "kokoro", translate_language: str = None):
    """Create a new TTS job"""
    files = {"file": open(file_path, "rb")}
    data = {
        "language": language,
        "engine": engine,
        "save_text": False
    }

    if translate_language:
        data["translate_language"] = translate_language

    response = requests.post(f"{API_BASE}/jobs/", files=files, data=data)
    files["file"].close()

    if response.status_code == 201:
        job = response.json()
        print(f"✅ Job created successfully! ID: {job['id']}")
        return job["id"]
    else:
        print(f"❌ Error creating job: {response.json()}")
        return None


def get_jobs():
    """Get all jobs"""
    response = requests.get(f"{API_BASE}/jobs/")
    if response.status_code == 200:
        jobs = response.json()
        print(f"📋 Found {len(jobs)} jobs:")
        for job in jobs:
            status_emoji = {
                "pending": "⏳",
                "running": "🔄",
                "completed": "✅",
                "failed": "❌",
                "cancelled": "🚫"
            }.get(job["status"], "❓")

            print(f"  {status_emoji} ID: {job['id']} | {job['filename']} | Status: {job['status']} | Progress: {job['progress']:.1f}%")
        return jobs
    else:
        print(f"❌ Error getting jobs: {response.json()}")
        return []


def get_job(job_id: int):
    """Get specific job details"""
    response = requests.get(f"{API_BASE}/jobs/{job_id}")
    if response.status_code == 200:
        job = response.json()
        print(f"📄 Job {job_id} details:")
        print(f"  File: {job['filename']}")
        print(f"  Status: {job['status']}")
        print(f"  Progress: {job['progress']:.1f}%")
        print(f"  Language: {job['language']}")
        print(f"  Engine: {job['engine']}")
        if job.get('error_message'):
            print(f"  Error: {job['error_message']}")
        return job
    else:
        print(f"❌ Error getting job {job_id}: {response.json()}")
        return None


def cancel_job(job_id: int):
    """Cancel a job"""
    response = requests.post(f"{API_BASE}/jobs/{job_id}/cancel")
    if response.status_code == 200:
        print(f"✅ Job {job_id} cancelled successfully")
    else:
        print(f"❌ Error cancelling job {job_id}: {response.json()}")


def download_job(job_id: int, output_path: str = None):
    """Download job result"""
    response = requests.get(f"{API_BASE}/jobs/{job_id}/download")
    if response.status_code == 200:
        if not output_path:
            # Try to get filename from headers
            content_disposition = response.headers.get('content-disposition', '')
            if 'filename=' in content_disposition:
                filename = content_disposition.split('filename=')[1].strip('"')
            else:
                filename = f"job_{job_id}_result.mp3"
            output_path = filename

        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"✅ Downloaded result to: {output_path}")
    else:
        print(f"❌ Error downloading job {job_id}: {response.json()}")


def wait_for_completion(job_id: int, timeout: int = 300):
    """Wait for job to complete"""
    print(f"⏳ Waiting for job {job_id} to complete...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        job = get_job(job_id)
        if not job:
            break

        if job["status"] in ["completed", "failed", "cancelled"]:
            if job["status"] == "completed":
                print(f"✅ Job {job_id} completed successfully!")
            else:
                print(f"❌ Job {job_id} finished with status: {job['status']}")
            return job

        print(f"🔄 Progress: {job['progress']:.1f}% | Status: {job['status']}")
        time.sleep(5)

    print(f"⏰ Timeout waiting for job {job_id}")
    return None


def main():
    parser = argparse.ArgumentParser(description="Audify API CLI Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create job command
    create_parser = subparsers.add_parser("create", help="Create a new TTS job")
    create_parser.add_argument("file", help="Path to PDF or EPUB file")
    create_parser.add_argument("--language", "-l", default="en", help="Language of the text (default: en)")
    create_parser.add_argument("--engine", "-e", default="kokoro", choices=["kokoro", "tts_models"], help="TTS engine (default: kokoro)")
    create_parser.add_argument("--translate", "-t", help="Translate to this language")
    create_parser.add_argument("--wait", "-w", action="store_true", help="Wait for job completion")
    create_parser.add_argument("--download", "-d", help="Download result to this path after completion")

    # List jobs command
    list_parser = subparsers.add_parser("list", help="List all jobs")

    # Get job command
    get_parser = subparsers.add_parser("get", help="Get job details")
    get_parser.add_argument("job_id", type=int, help="Job ID")

    # Cancel job command
    cancel_parser = subparsers.add_parser("cancel", help="Cancel a job")
    cancel_parser.add_argument("job_id", type=int, help="Job ID")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download job result")
    download_parser.add_argument("job_id", type=int, help="Job ID")
    download_parser.add_argument("--output", "-o", help="Output file path")

    # Wait command
    wait_parser = subparsers.add_parser("wait", help="Wait for job completion")
    wait_parser.add_argument("job_id", type=int, help="Job ID")
    wait_parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds (default: 300)")

    args = parser.parse_args()

    if args.command == "create":
        if not Path(args.file).exists():
            print(f"❌ File not found: {args.file}")
            return

        job_id = create_job(args.file, args.language, args.engine, args.translate)
        if job_id and args.wait:
            job = wait_for_completion(job_id)
            if job and job["status"] == "completed" and args.download:
                download_job(job_id, args.download)

    elif args.command == "list":
        get_jobs()

    elif args.command == "get":
        get_job(args.job_id)

    elif args.command == "cancel":
        cancel_job(args.job_id)

    elif args.command == "download":
        download_job(args.job_id, args.output)

    elif args.command == "wait":
        wait_for_completion(args.job_id, args.timeout)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
