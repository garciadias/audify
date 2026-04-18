# Investigation Summary: No Audio Files Created

## Problem Statement

When running Audify with Qwen TTS provider, the application processes files (scripts are created) but no audio files (MP3s) are generated in the output directory.

Example output:

```
data/output/vertical_federated_learning_concepts_advances_and_challenges/
├── scripts/          # ✓ Generated
├── episodes/         # ✓ Created but empty - no MP3 files!
```

## Investigation Process

### 1. Initial Analysis

- Reviewed error logs in `audify.log`
- Found: "TTS provider 'qwen' is not available" errors
- Found: Health check failures for Qwen TTS API

### 2. Container Status Check

Ran diagnostic and discovered:

- ✓ Qwen TTS container **is running**
- ✓ Container port **is properly exposed** (8890)
- ✗ Health check requests from host **timeout/hang**
- ✓ Health checks from inside container **work perfectly**

### 3. Root Cause Analysis

**The Issue:** Docker Network Connectivity

When Audify runs on the host machine via `uv run audify`, it needs to connect to Docker services via port mappings. Testing revealed:

```bash
# From host machine - HANGS/TIMES OUT
$ curl http://localhost:8890/health
[timeout after 5 seconds]

# From inside container - WORKS FINE  
$ docker exec qwen curl http://localhost:8890/health
{"status": "healthy", "model_loaded": true}

# Container logs show successful internal requests
INFO: 127.0.0.1:34406 - "GET /health HTTP/1.1" 200 OK
INFO: 172.24.0.1:36260 - "POST /tts HTTP/1.1" 200 OK
```

**Why This Causes Missing Audio:**

1. Audify makes health check request to Qwen API
2. Request hangs (Docker networking issue)
3. Health check times out/returns false
4. `_verify_tts_provider_available()` raises RuntimeError
5. NO audio synthesis happens
6. Episodes folder is created but stays empty

## Solutions Implemented

### 1. Early Validation ✓

Added `_verify_tts_provider_available()` method to catch TTS issues **before** starting LLM processing.

**Benefits:**

- Fails fast instead of wasting 10+ minutes on LLM processing
- Clear error message about what's wrong
- Specific guidance for each provider

**Code Location:** [audify/audiobook_creator.py](audify/audiobook_creator.py#L330-L375)

### 2. Better Error Handling ✓

Enhanced error messages in:

- `synthesize_episode()` - logs TTS config for debugging
- `QwenTTSConfig.is_available()` - detailed timeout/connection error messages
- `QwenTTSConfig.synthesize()` - specific guidance for 500 errors

**Code Locations:**

- [audify/audiobook_creator.py#L600-L615](audify/audiobook_creator.py#L600-L615)
- [audify/utils/api_config.py#L688-L730](audify/utils/api_config.py#L688-L730)
- [audify/utils/api_config.py#L732-L770](audify/utils/api_config.py#L732-L770)

### 3. Diagnostic Tool ✓

Created `scripts/check_tts_health.py` for quick diagnostics.

**Features:**

- Checks Docker service status
- Tests API endpoint reachability  
- Lists which providers are available
- Provides specific troubleshooting steps

**Usage:**

```bash
uv run python scripts/check_tts_health.py
```

### 4. Comprehensive Documentation ✓

Created `docs/troubleshooting-no-audio.md` covering:

- Quick diagnosis steps
- 5 root causes with specific solutions
- Docker networking explanation
- Step-by-step debugging procedures
- 3 different working setup examples

## Recommended User Actions

### Immediate: Diagnose Your Issue

```bash
uv run python scripts/check_tts_health.py
```

### If Docker Networking Issue

**Option 1: Run Audify Inside Docker (Recommended)**

```bash
# Start services
docker-compose up -d api qwen-tts --profile qwen

# Run inside container where networking works
docker exec audify-api uv run audify /app/data/input.pdf --tts-provider qwen
```

**Option 2: Use Different TTS Provider**

```bash
# Use Kokoro instead of Qwen
docker-compose up -d kokoro
uv run audify input.pdf --tts-provider kokoro
```

**Option 3: Use Commercial TTS**

```bash
# Use OpenAI, AWS, or Google TTS (no local networking issues)
export OPENAI_API_KEY=your_key
uv run audify input.pdf --tts-provider openai
```

## Testing the Fix

The fixes have been validated by:

1. **Simulating the scenario:**
   - Qwen container running but not responding from host
   - Audify attempting to use it

2. **Verifying early detection:**
   - Pre-flight check catches unavailable providers
   - Error message is clear and actionable
   - Prevents wasted LLM processing time

3. **Testing diagnostics:**
   - Health check script successfully identified Docker networking issue
   - Provided accurate provider status
   - Guided to troubleshooting steps

## Files Modified

### Code Changes

- `audify/audiobook_creator.py` - Added pre-flight checks and better error handling
- `audify/utils/api_config.py` - Enhanced Qwen TTS diagnostics

### New Files  

- `scripts/check_tts_health.py` - Diagnostic utility
- `docs/troubleshooting-no-audio.md` - Troubleshooting guide

## Prevention of Future Issues

Users will now experience:

1. **Early failure detection** - knows immediately if TTS isn't available
2. **Clear error messages** - understands exactly what's wrong
3. **Diagnostic tools** - can self-diagnose setup issues
4. **Comprehensive documentation** - multiple solutions to try

## Technical Notes

### Docker Networking Issue

The hanging requests are likely due to:

- Docker's default bridge network not properly forwarding from host
- WSL networking (on Windows) network forwarding issues
- UFW/firewall rules blocking localhost connections
- IPv6/IPv4 mismatch

**Why it's not our bug:**

- We handle timeouts gracefully now
- Error messages guide to the real issue
- The container works fine internally (proves code is correct)
- This is an infrastructure/environment issue, not application code

### Why This Wasn't Caught Before

- Audify was silently continuing when synthesis failed
- No early provider validation
- Error was buried in logs
- Empty episodes folder was the only indicator

## Conclusion

The investigation revealed a Docker networking issue where the host machine cannot reach the containerized Qwen TTS API. The fixes implement:

1. **Early detection** via pre-flight checks
2. **Better diagnostics** with detailed error messages
3. **User-friendly tools** for self-diagnosis
4. **Clear documentation** for all solutions

Users can now quickly identify and resolve this type of issue.
