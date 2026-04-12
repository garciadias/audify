# REST API

Audify exposes a FastAPI HTTP server for programmatic access.

## Starting the API

```bash
# Development mode (auto-reload)
audify api
# or: task api

# Via Docker (starts with Kokoro and Ollama)
docker compose up -d
```

The API runs on `http://localhost:8000` by default.

Interactive Swagger docs: `http://localhost:8000/docs`

## Endpoints

| Method | Path           | Description                      |
|--------|----------------|----------------------------------|
| GET    | `/health`      | Health check                     |
| GET    | `/providers`   | List available TTS providers     |
| GET    | `/voices`      | List voices (query: `provider`, `language`) |
| POST   | `/synthesize`  | Convert EPUB/PDF to MP3          |
| POST   | `/audiobook`   | Convert EPUB/PDF to M4B audiobook|

## Examples

### Synthesize an EPUB to MP3

```bash
curl -X POST http://localhost:8000/synthesize \
  -F "file=@book.epub" \
  -F "voice=af_bella" \
  -F "language=en" \
  --output book.mp3
```

### Create an M4B audiobook

```bash
curl -X POST http://localhost:8000/audiobook \
  -F "file=@book.epub" \
  -F "voice=af_bella" \
  -F "language=en" \
  --output book.m4b
```

### List available providers

```bash
curl http://localhost:8000/providers
```

### List voices for a provider

```bash
curl "http://localhost:8000/voices?provider=kokoro&language=en"
```
