#!/usr/bin/env python3
"""
Diagnostic script to check TTS provider health and configuration.

This helps troubleshoot issues when audio files aren't being created.
"""

import os
import shutil
import sys
from pathlib import Path

# Add parent directory to path to import audify modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

from audify.utils.api_config import get_tts_config
from audify.utils.constants import AVAILABLE_TTS_PROVIDERS, KOKORO_API_BASE_URL


def check_provider_health(provider: str, language: str = "en") -> bool:
    """Check if a TTS provider is healthy and configured."""
    print(f"\n{'=' * 60}")
    print(f"Checking TTS Provider: {provider.upper()}")
    print(f"{'=' * 60}")

    try:
        config = get_tts_config(provider=provider, language=language)
        print("✓ Provider config created successfully")
        print(f"  Provider: {config.provider_name}")
        print(f"  Language: {config.language}")
        if hasattr(config, "base_url"):
            print(f"  API URL: {config.base_url}")

        # Check availability
        if config.is_available():
            print("✓ Provider is AVAILABLE and responding")

            # Get voices
            try:
                voices = config.get_available_voices()
                if voices:
                    print(f"✓ Available voices: {', '.join(voices[:5])}")
                    if len(voices) > 5:
                        print(f"  ... and {len(voices) - 5} more")
                else:
                    print("⚠ No voices information available")
            except Exception as e:
                print(f"⚠ Could not fetch voices: {e}")

            return True
        else:
            print("✗ Provider is NOT AVAILABLE")
            if hasattr(config, "base_url"):
                print(f"  Tried connecting to: {config.base_url}")
            return False

    except Exception as e:
        print(f"✗ Error checking provider: {e}")
        return False


def check_docker_services() -> None:
    """Check if Docker services are running."""
    print(f"\n{'=' * 60}")
    print("Checking Docker Services")
    print(f"{'=' * 60}")

    # Try to detect if Docker containers are running
    import subprocess

    try:
        docker_bin = shutil.which("docker")
        if docker_bin is None:
            print("✗ Docker is not installed or not in PATH")
            return

        result = subprocess.run(
            [docker_bin, "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            containers = result.stdout.strip().split("\n")
            print("✓ Docker is running")
            print(f"  Active containers: {len(containers)}")

            kokoro_running = any("kokoro" in c.lower() for c in containers)

            if kokoro_running:
                print("  ✓ kokoro container is running")
            else:
                print("  ✗ kokoro container is NOT running")
        else:
            print("⚠ Could not list Docker containers")
    except FileNotFoundError:
        print("✗ Docker is not installed or not in PATH")
    except subprocess.TimeoutExpired:
        print("⚠ Docker command timed out")
    except Exception as e:
        print(f"⚠ Error checking Docker: {e}")


def check_api_endpoints() -> None:
    """Check if API endpoints are responding."""
    print(f"\n{'=' * 60}")
    print("Checking API Endpoints")
    print(f"{'=' * 60}")

    endpoints = {
        "Kokoro TTS": f"{KOKORO_API_BASE_URL}/health",
    }

    for name, url in endpoints.items():
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                print(f"✓ {name}: {url} - responding")
            else:
                print(f"⚠ {name}: {url} - status {response.status_code}")
        except requests.exceptions.Timeout:
            print(f"✗ {name}: {url} - timeout")
        except requests.exceptions.ConnectionError:
            print(f"✗ {name}: {url} - connection refused")
        except Exception as e:
            print(f"✗ {name}: {url} - error: {e}")


def check_environment_variables() -> None:
    """Check for relevant environment variables."""
    print(f"\n{'=' * 60}")
    print("Environment Variables")
    print(f"{'=' * 60}")

    relevant_vars = [
        "TTS_PROVIDER",
        "KOKORO_API_URL",
        "OLLAMA_API_URL",
        "OPENAI_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ]

    set_vars = []
    for var in relevant_vars:
        value = os.environ.get(var)
        if value:
            # Mask sensitive values
            if "KEY" in var or "CREDENTIALS" in var:
                display_value = f"***{value[-4:]}"
            else:
                display_value = value
            print(f"✓ {var}: {display_value}")
            set_vars.append(var)

    if not set_vars:
        print("⚠ No TTS-related environment variables set")

    default_provider = os.environ.get("TTS_PROVIDER", "kokoro")
    print(f"\n  Default TTS Provider: {default_provider}")


def main():
    """Run all health checks."""
    print("\n" + "=" * 60)
    print("Audify TTS Provider Health Check")
    print("=" * 60)

    # Check environment
    check_environment_variables()
    check_docker_services()
    check_api_endpoints()

    # Check each provider
    results = {}
    for provider in AVAILABLE_TTS_PROVIDERS:
        results[provider] = check_provider_health(provider)

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")

    available_providers = [p for p, available in results.items() if available]
    unavailable_providers = [p for p, available in results.items() if not available]

    if available_providers:
        print(f"✓ Available providers: {', '.join(available_providers)}")
    else:
        print("✗ No TTS providers available")

    if unavailable_providers:
        print(f"✗ Unavailable providers: {', '.join(unavailable_providers)}")

    print(f"\n{'=' * 60}")

    if not available_providers:
        print(
            "⚠ No TTS providers are available. "
            "Please configure one before running audify."
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
