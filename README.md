# 🎙️ Audio Transcription & Diarization API

Production-ready FastAPI application for audio transcription with speaker diarization.

## ✨ Features

- 🎯 **Speaker Diarization** - Identify who speaks when using Pyannote
- 📝 **Speech-to-Text** - High-quality transcription with OpenAI Whisper
- 💡 **AI Summaries** - Optional summarization using Claude AI
- 🚀 **GPU Accelerated** - Optimized for NVIDIA Tesla V100
- 📊 **Detailed Statistics** - Word count, speaking time, speaker breakdown
- 🔒 **Production Ready** - Proper error handling, logging, validation

## 📋 Requirements

### Hardware

- NVIDIA GPU (Tesla V100 or similar) recommended
- 16GB+ RAM
- 50GB+ disk space

### Software

- Ubuntu 20.04/22.04
- Python 3.10
- CUDA 12.1
- FFmpeg

## 🚀 Quick Start

### 1. Clone & Deploy

```bash
cd /home/admin-lapas
git clone <your-repo> audio-api
cd audio-api

# Run deployment script
chmod +x deploy.sh
./deploy.sh
```

### 2. Configure API Keys

Edit `.env` file:

```bash
nano .env
```

Required:

```env
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxx
```

Optional (for AI summaries):

```env
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx
```

Get tokens:

- HuggingFace: https://huggingface.co/settings/tokens
- Anthropic: https://console.anthropic.com/

**Important:** Accept Pyannote terms at:
https://huggingface.co/pyannote/speaker-diarization-3.1

### 3. Start the API

```bash
# Start service
sudo systemctl start audio-api

# Enable auto-start on boot
sudo systemctl enable audio-api

# Check status
sudo systemctl status audio-api
```

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/health

# API documentation
http://YOUR_SERVER_IP:8000/docs
```

## 📡 API Usage

### Endpoint

```
POST /api/v1/transcribe
```

### Parameters

- `file` (required) - Audio file (WAV, MP3, M4A, FLAC, OGG)
- `num_speakers` (optional) - Expected number of speakers (auto-detect if not provided)
- `include_summary` (optional) - Generate AI summary (requires Anthropic API key)

### Example: cURL

```bash
curl -X POST "http://localhost:8000/api/v1/transcribe" \
  -F "file=@meeting.wav" \
  -F "num_speakers=2" \
  -F "include_summary=true"
```

### Example: Python

```python
import requests

url = "http://localhost:8000/api/v1/transcribe"

with open("meeting.wav", "rb") as f:
    files = {"file": f}
    data = {
        "num_speakers": 2,
        "include_summary": True
    }
    response = requests.post(url, files=files, data=data)

result = response.json()
print(result["data"]["transcript"])
```

### Example: JavaScript

```javascript
const formData = new FormData();
formData.append("file", audioFile);
formData.append("num_speakers", 2);
formData.append("include_summary", true);

const response = await fetch("http://localhost:8000/api/v1/transcribe", {
  method: "POST",
  body: formData,
});

const result = await response.json();
console.log(result.data.transcript);
```

## 📊 Response Format

```json
{
  "success": true,
  "message": "Audio processed successfully",
  "data": {
    "audio_info": {
      "duration_sec": 120.5,
      "duration_min": 2.0,
      "sample_rate": 16000,
      "channels": 1
    },
    "transcript": "SPEAKER_1 [00:00 - 00:05]: Hello...",
    "segments": [
      {
        "speaker": "SPEAKER_1",
        "start": 0.0,
        "end": 5.2,
        "duration": 5.2,
        "text": "Hello, how are you?"
      }
    ],
    "statistics": {
      "total_duration": 120.5,
      "total_words": 250,
      "num_speakers": 2,
      "speakers": {
        "SPEAKER_1": {
          "duration": 60.0,
          "turns": 5,
          "words": 120,
          "percentage": 49.8
        }
      }
    },
    "summary": "AI-generated summary..."
  },
  "timestamp": "2024-02-04T10:30:00"
}
```

## 🔧 Configuration

All settings in `.env` file:

```env
# Server
HOST=0.0.0.0
PORT=8000
WORKERS=1

# Whisper
WHISPER_MODEL=large-v3
WHISPER_LANGUAGE=id
WHISPER_DEVICE=cuda

# Pyannote
HUGGINGFACE_TOKEN=your_token

# Claude (optional)
ANTHROPIC_API_KEY=your_key
```

### Whisper Models

- `tiny` - Fastest, lowest quality
- `base` - Fast, decent quality
- `small` - Balanced
- `medium` - Good quality
- `large-v3` - Best quality (recommended)

## 📝 Logging

```bash
# View logs
sudo journalctl -u audio-api -f

# Log file
tail -f logs/api.log
```

## 🛠️ Management

```bash
# Start
sudo systemctl start audio-api

# Stop
sudo systemctl stop audio-api

# Restart
sudo systemctl restart audio-api

# Status
sudo systemctl status audio-api

# Enable auto-start
sudo systemctl enable audio-api

# Disable auto-start
sudo systemctl disable audio-api
```

## 🐛 Troubleshooting

### GPU Not Detected

```bash
# Check GPU
nvidia-smi

# Check CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
```

### HuggingFace Token Error

1. Get token: https://huggingface.co/settings/tokens
2. Accept terms: https://huggingface.co/pyannote/speaker-diarization-3.1
3. Update `.env` file
4. Restart service

### Out of Memory

Reduce Whisper model size in `.env`:

```env
WHISPER_MODEL=medium  # or small
```

## 📚 API Documentation

Interactive API docs available at:

- Swagger UI: `http://YOUR_SERVER:8000/docs`
- ReDoc: `http://YOUR_SERVER:8000/redoc`

## 🏗️ Project Structure

```
audio-api/
├── app.py                 # FastAPI application
├── src/
│   ├── __init__.py
│   ├── config.py         # Configuration
│   ├── models.py         # Pydantic models
│   ├── pipeline.py       # Main pipeline
│   ├── audio_utils.py    # Audio utilities
│   ├── diarizer.py       # Speaker diarization
│   ├── transcriber.py    # Whisper transcription
│   ├── formatter.py      # Transcript formatting
│   └── summarizer.py     # AI summarization
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables
├── deploy.sh             # Deployment script
└── logs/                 # Log files
```

## 📄 License

MIT License

## 🤝 Support

For issues or questions, please contact the development team.

---

Made with ❤️ for production deployment
