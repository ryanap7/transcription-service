# Audio Transcription System

A professional audio transcription system with speaker diarization and AI-powered summarization. Process audio files to get accurate transcripts with speaker identification and intelligent summaries.

## Features

- 🎙️ **Speaker Diarization**: Automatically identify and separate different speakers
- 📝 **Speech-to-Text**: High-quality transcription using OpenAI Whisper
- 🤖 **AI Summarization**: Intelligent summaries powered by Anthropic Claude
- 🌍 **Multi-language Support**: Support for multiple languages including Indonesian and English
- 🚫 **No Raw Audio Storage**: Process audio in-memory without saving raw files
- 📊 **Rich Statistics**: Detailed analytics on speakers, duration, and word counts
- 🔄 **Multiple Output Formats**: TXT, JSON, SRT, and VTT formats
- ⚡ **REST API**: Easy integration via HTTP API
- 🛡️ **Robust Error Handling**: Comprehensive error handling with standard international responses

## Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg (for audio processing)

#### Install FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html)

### Setup

1. Clone or download this project

2. Create virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

### Required Configuration

1. **HuggingFace Token** (Required for speaker diarization)
   - Get token from: https://huggingface.co/settings/tokens
   - Accept terms at: https://huggingface.co/pyannote/speaker-diarization-3.1

2. **Anthropic API Key** (Optional, for AI summaries)
   - Get from: https://console.anthropic.com/

### Environment Variables

```bash
# Required
HUGGINGFACE_TOKEN=your_huggingface_token_here

# Optional
ANTHROPIC_API_KEY=your_anthropic_api_key_here
WHISPER_MODEL=base  # Options: tiny, base, small, medium, large, large-v3
LANGUAGE=id  # Default language code
```

## Usage

### Command Line

#### Basic Usage

```bash
python pipeline.py audio_file.mp3
```

#### With Python Script

```python
from pipeline import AudioTranscriptionPipeline

# Initialize pipeline
pipeline = AudioTranscriptionPipeline(
    whisper_model='base',  # Model size
    language='id',         # Target language
    enable_summary=True    # Enable AI summary
)

# Process audio
result = pipeline.process(
    audio_input='recording.mp3',
    num_speakers=2,              # Optional: exact number of speakers
    include_summary=True,        # Generate AI summary
    output_formats=['txt', 'json', 'srt']  # Output formats
)

# Access results
print(f"Duration: {result['statistics']['total_duration']/60:.2f} minutes")
print(f"Speakers: {result['statistics']['num_speakers']}")
print(f"Summary: {result['summary']}")
```

### REST API

#### Start API Server

```bash
python api.py
```

The API will be available at `http://localhost:8000`

#### API Endpoints

**Health Check**
```bash
GET /health
```

**Transcribe Audio**
```bash
POST /transcribe

Form Data:
- file: Audio file (required)
- num_speakers: Number of speakers (optional)
- min_speakers: Minimum speakers (optional)
- max_speakers: Maximum speakers (optional)
- language: Language code (optional)
- include_summary: true/false (optional)
- output_formats: txt,json,srt,vtt (optional)
```

**Example with cURL:**
```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@recording.mp3" \
  -F "num_speakers=2" \
  -F "include_summary=true" \
  -F "output_formats=txt,json,srt"
```

**Example with Python:**
```python
import requests

url = 'http://localhost:8000/transcribe'
files = {'file': open('recording.mp3', 'rb')}
data = {
    'num_speakers': 2,
    'include_summary': 'true',
    'output_formats': 'txt,json,srt'
}

response = requests.post(url, files=files, data=data)
result = response.json()

print(result)
```

**Download Output File**
```bash
GET /download/<filename>
```

**Get Supported Formats**
```bash
GET /formats
```

## Output Formats

### TXT Format
Human-readable transcript with timestamps and speaker labels:
```
[00:00 - 00:05] SPEAKER_1: Hello, welcome to the meeting.
[00:05 - 00:10] SPEAKER_2: Thank you for having me.
```

### JSON Format
Structured data including full transcript, segments, statistics, and summary:
```json
{
  "audio_info": {
    "duration_minutes": 5.2,
    "sample_rate": 16000
  },
  "statistics": {
    "total_duration": 312.5,
    "total_words": 450,
    "num_speakers": 2
  },
  "segments": [...],
  "summary": "..."
}
```

### SRT Format
Subtitle format for video players:
```
1
00:00:00,000 --> 00:00:05,000
SPEAKER_1: Hello, welcome to the meeting.

2
00:00:05,000 --> 00:00:10,000
SPEAKER_2: Thank you for having me.
```

### VTT Format
WebVTT subtitle format for web browsers:
```
WEBVTT

00:00:00.000 --> 00:00:05.000
SPEAKER_1: Hello, welcome to the meeting.

00:00:05.000 --> 00:00:10.000
SPEAKER_2: Thank you for having me.
```

## Supported Audio Formats

- MP3
- WAV
- M4A
- FLAC
- OGG
- OPUS
- WebM

## Error Handling

The system provides comprehensive error handling with standard international error responses:

### Error Types

- `AudioFileError`: Issues with audio file loading
- `AudioFormatError`: Unsupported audio format
- `AudioSizeError`: File exceeds size limit
- `ModelLoadError`: Failed to load AI models
- `DiarizationError`: Speaker diarization failed
- `TranscriptionError`: Transcription failed
- `SummarizationError`: AI summarization failed
- `ConfigurationError`: Configuration issues

### Example Error Response (API)

```json
{
  "success": false,
  "error": "Unsupported format: .avi. Supported formats: .mp3, .wav, .m4a, .flac, .ogg, .opus, .webm",
  "error_type": "AudioFormatError"
}
```

## Performance Considerations

### Model Selection

| Model | Speed | Accuracy | RAM | Use Case |
|-------|-------|----------|-----|----------|
| tiny | Fastest | Low | ~1GB | Quick drafts |
| base | Fast | Good | ~1GB | General use |
| small | Medium | Better | ~2GB | Balanced |
| medium | Slow | High | ~5GB | High accuracy |
| large | Slowest | Highest | ~10GB | Best quality |

### Processing Time

Approximate processing time (relative to audio duration):
- **Tiny/Base**: 0.1-0.2x (e.g., 5 min audio = 0.5-1 min processing)
- **Small**: 0.3-0.5x
- **Medium**: 0.5-1x
- **Large**: 1-2x

### File Size Limits

- Default maximum: 500MB
- Configurable in `config.py`
- Recommended: Keep files under 100MB for optimal performance

## Architecture

```
┌─────────────────┐
│  Audio Input    │
│ (File/Stream)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Audio Processor │  ← Validate & Prepare (No Save)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Diarizer      │  ← Identify Speakers
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Transcriber    │  ← Speech-to-Text
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Summarizer     │  ← AI Summary (Optional)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Formatter     │  ← Generate Outputs
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Output Files   │
│ (TXT/JSON/SRT)  │
└─────────────────┘
```

## Project Structure

```
audio-transcription/
├── config.py              # Configuration settings
├── exceptions.py          # Custom exception classes
├── audio_processor.py     # Audio processing utilities
├── diarizer.py           # Speaker diarization
├── transcriber.py        # Speech transcription
├── summarizer.py         # AI summarization
├── formatter.py          # Output formatting
├── pipeline.py           # Main pipeline orchestration
├── api.py                # REST API server
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
└── README.md            # This file
```

## Troubleshooting

### Common Issues

**1. "HUGGINGFACE_TOKEN is required"**
- Solution: Set `HUGGINGFACE_TOKEN` in `.env` file
- Get token from https://huggingface.co/settings/tokens
- Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1

**2. "Failed to load Whisper model"**
- Solution: Check internet connection (first run downloads model)
- Ensure sufficient disk space (~1-10GB depending on model)
- Try smaller model: `WHISPER_MODEL=base`

**3. "FFmpeg not found"**
- Solution: Install FFmpeg (see Installation section)

**4. Out of memory errors**
- Solution: Use smaller Whisper model
- Reduce audio file size
- Close other applications

**5. Slow processing**
- Solution: Use smaller model (tiny/base)
- Enable GPU if available
- Process shorter audio segments

## Advanced Usage

### Batch Processing

```python
from pipeline import AudioTranscriptionPipeline

pipeline = AudioTranscriptionPipeline()

audio_files = ['file1.mp3', 'file2.mp3', 'file3.mp3']
results = pipeline.process_batch(
    audio_files,
    include_summary=True,
    output_formats=['txt', 'json']
)

for result in results:
    if result['success']:
        print(f"✓ {result['audio_info']}")
    else:
        print(f"✗ Error: {result['error']}")
```

### Custom Summary Types

```python
from summarizer import AISummarizer
from formatter import TranscriptFormatter

summarizer = AISummarizer()

# Meeting summary
summary = summarizer.create_meeting_summary(
    transcript=transcript_text,
    statistics=stats,
    meeting_type='standup',  # or 'interview', 'general'
    language='id'
)
```

### Stream Processing

```python
# Process audio from bytes or file-like object
with open('recording.mp3', 'rb') as f:
    result = pipeline.process(f.read())
```

## License

This project uses several open-source components:
- OpenAI Whisper (MIT License)
- pyannote.audio (MIT License)
- Anthropic Claude API (Commercial)

## Credits

- **Whisper**: OpenAI
- **Pyannote**: CNRS
- **Claude**: Anthropic

## Support

For issues, questions, or contributions, please refer to the project repository or contact the maintainers.

---

**Note**: This system processes audio without storing raw audio files, ensuring privacy and efficient disk usage. All processing is done in-memory with only the final transcripts and outputs saved.