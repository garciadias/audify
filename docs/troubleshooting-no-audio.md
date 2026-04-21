# Troubleshooting: No Audio Files Created

This guide helps resolve issues when Audify processes files but doesn't create audio files in the output directory.

## Common Symptom

```
🐧 ❯ ls data/output/vertical_federated_learning_concepts_advances_and_challenges/ -lr
drwxr-xr-x rd24 bioeng_linux 10 B Sat Apr 18 10:58:15 2026  scripts
drwxr-xr-x rd24 bioeng_linux 10 B Sat Apr 18 16:14:09 2026  episodes
```

Episodes folder exists but contains no MP3 files.

---

## Quick Diagnosis

Run the health check diagnostic:

```bash
uv run python scripts/check_tts_health.py
```

This will show:

- Which TTS providers are configured
- Which services are running
- Which services are reachable
- Detailed connectivity status

If health checks are flaky during heavy synthesis, tune preflight behavior:

```bash
# More resilient Qwen health checks
export QWEN_HEALTH_TIMEOUT=8
export QWEN_HEALTH_RETRIES=3

# Optional: continue even if startup preflight fails
export AUDIFY_STRICT_TTS_PREFLIGHT=0

# Optional: skip preflight entirely (advanced)
export AUDIFY_SKIP_TTS_PREFLIGHT=1
```

---

## Root Causes & Solutions

### 1. TTS Provider Not Available

**Symptom:**

```
✗ Provider is NOT AVAILABLE
✗ TTS provider 'qwen' is not available. Please check your configuration and credentials.
```

**Solution:** Ensure the TTS service is running.

#### For Qwen TTS

```bash
# Start the Qwen TTS container with GPU support
docker compose --profile qwen -f docker-compose.yml up -d qwen-tts

# Check if container is running
docker compose ps qwen-tts

# View container logs
docker compose logs -f qwen-tts

# Test health endpoint from inside container (should work)
docker exec $(docker ps -q -f "name=qwen") curl -s http://localhost:8890/health | python3 -m json.tool
```

#### For Kokoro TTS

```bash
# Start the Kokoro TTS container
docker compose --profile kokoro -f docker-compose.yml up -d kokoro

# Check if container is running
docker compose ps kokoro
```

---

### 2. Docker Networking Issue (Most Common)

**Symptom:**

- Container is running ✓
- But requests from host machine timeout or hang
- Diagnostic shows: `✗ Qwen TTS: http://localhost:8890/health - timeout`
- But `docker exec ... curl` works fine

**Root Cause:**
When running Audify on the host machine (via `uv run audify ...`), it needs to connect to Docker services via port mappings. In some Docker configurations (especially with certain network drivers or WSL on Windows), the port forwarding doesn't work properly, causing hangs.

**Solutions:**

#### Option 1: Run Audify Inside Docker Container (Recommended)

Use the Dockerfile.api to run Audify in a Docker container where it can connect to other services:

```bash
# Build and run API container
docker compose --profile qwen -f docker-compose.yml up -d api

# The API is then available at http://localhost:8000
# Or run commands inside the container:
docker exec audify-api uv run audify /path/to/file.pdf --tts-provider qwen
```

Path handling in container runs:

- Host-like input paths are auto-resolved to mounted `/app/data/...` paths.
- Inputs outside `/app/data` are copied to `/app/data/input`.
- Outputs outside `/app/data` are copied to `/app/data/output` so they appear
   on the host volume automatically.

#### Option 2: Use Docker Host Network Mode

Edit your docker-compose file to enable host networking for better host-to-container connectivity:

```yaml
qwen-tts:
  # ... other config ...
  network_mode: "host"  # Add this line
```

Then restart:

```bash
docker compose --profile qwen -f docker-compose.yml up -d qwen-tts
```

#### Option 3: Use Docker DNS Resolution

Some systems work better with Docker DNS. Update your docker-compose:

```yaml
services:
  qwen-tts:
    # ... config ...
```

If Audify runs inside the Docker Compose network, use the service name:

```bash
docker exec audify-api env QWEN_API_URL=http://qwen-tts:8890 \
   uv run audify /app/data/input.pdf --tts-provider qwen
```

If Audify runs on the host machine, use localhost with the mapped port:

```bash
export QWEN_API_URL=http://localhost:8890
uv run audify input.pdf --tts-provider qwen
```

---

### 3. TTS Service Still Loading Model

**Symptom:**

```
✗ Provider is NOT AVAILABLE - model_loaded: false
```

Qwen TTS takes time to load the model (2-3 minutes) on first run.

**Solution:** Wait for the container to finish loading:

```bash
# Watch the logs
docker compose logs -f qwen-tts

# Wait until you see: "✓ Model loaded successfully"
# Then try again
```

---

### 4. Port Already in Use

**Symptom:**

```
docker: Error response from daemon: bind: address already in use
```

**Solution:**

```bash
# Find what's using port 8890
lsof -i :8890

# Either stop that process or use a different port in docker-compose.yml
```

---

### 5. CUDA/GPU Not Available

**Symptom:**

```
Could not load model on CUDA device
```

**Solution:**

```bash
# Check NVIDIA Docker support
docker run --rm --gpus all nvidia/cuda:12.0-runtime nvidia-smi

# If this doesn't work, install nvidia-docker:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Then update docker-compose to use:
runtime: nvidia
```

Or fall back to CPU (slower):

```bash
QWEN_TTS_DEVICE=cpu docker compose --profile qwen -f docker-compose.yml up -d qwen-tts
```

---

## Detailed Debugging

### Step-by-Step Verification

1. **Check Docker is working:**

   ```bash
   docker ps
   docker compose ps
   ```

2. **Check TTS container is running:**

   ```bash
   docker ps -f "name=qwen"
   ```

3. **Check port mapping:**

   ```bash
   docker port <container_id> 8890
   # Should show: 0.0.0.0:8890 -> 8890/tcp
   ```

4. **Test API from inside container:**

   ```bash
   docker exec <container_id> curl -s http://localhost:8890/health
   # Should return JSON with "model_loaded": true
   ```

5. **Test API from host machine:**

   ```bash
   # Try different addresses
   curl http://localhost:8890/health
   curl http://127.0.0.1:8890/health
   curl http://docker.host.internal:8890/health  # macOS/WSL
   ```

6. **Check container logs for errors:**

   ```bash
   docker logs -f <container_id>
   
   # Look for:
   # - Model loading errors
   # - API startup issues
   # - Memory/GPU errors
   ```

---

## Log Files

Check application logs for detailed error messages:

```bash
# Main Audify logs
cat audify.log

# Docker container logs
docker compose logs qwen-tts
docker compose logs api

# System logs (macOS)
log stream --predicate 'process == "Docker"'

# System logs (Linux)
journalctl -u docker
```

---

## Testing TTS Synthesis Directly

If the service is running but synthesis isn't working, test it directly:

```bash
# Using curl
curl -X POST http://localhost:8890/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world, this is a test.",
    "language": "Auto",
    "speaker": "Vivian",
    "instruct": null
  }' \
  -o test.wav

# Play the file
ffplay test.wav

# Or check the file
file test.wav
```

---

## Environment Variables

Make sure these are set correctly:

```bash
# For Qwen TTS
export TTS_PROVIDER=qwen
export QWEN_API_URL=http://localhost:8890

# For Kokoro TTS  
export TTS_PROVIDER=kokoro
export KOKORO_API_URL=http://localhost:8887/v1

# For Ollama (if using LLM)
export OLLAMA_API_URL=http://localhost:11434
```

---

## Getting Help

If issues persist:

1. **Run full diagnostic:**

   ```bash
   uv run python scripts/check_tts_health.py 2>&1 | tee diagnostic_output.txt
   ```

2. **Collect logs:**

   ```bash
   docker compose logs qwen-tts > qwen_logs.txt
   cat audify.log > audify_logs.txt
   ```

3. **Check system resources:**

   ```bash
   docker stats
   nvidia-smi
   ```

4. **Verify network connectivity:**

   ```bash
   docker exec audify-api ping qwen-tts
   docker network inspect audify_default  # or your network name
   ```

---

## Quick Reference: Working Setup

### Host machine with Docker services

```bash
# 1. Start Qwen TTS
docker compose --profile qwen -f docker-compose.yml up -d qwen-tts

# 2. Wait for model to load (check logs)
docker compose logs -f qwen-tts

# 3. Verify it's working
uv run python scripts/check_tts_health.py

# 4. Run Audify
export QWEN_API_URL=http://localhost:8890
uv run audify input.pdf --task audiobook --language en --tts-provider qwen
```

### Inside Docker container

```bash
# 1. Start all services
docker compose --profile qwen -f docker-compose.yml up -d

# 2. Run Audify inside container
docker exec audify-api uv run audify /app/data/input.pdf --tts-provider qwen

# 3. Check output
docker exec audify-api ls -la /app/data/output/
```

---

## Still Having Issues?

1. Check the [Docker troubleshooting guide](https://docs.docker.com/config/containers/container-networking/)
2. Review [NVIDIA Docker setup](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html)
3. Review the main [README](https://github.com/garciadias/audify#readme)
4. Check [issue tracker](https://github.com/you/audify/issues) for similar problems
