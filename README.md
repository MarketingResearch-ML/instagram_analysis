# Instagram Audio Sentiment Analysis

## Setup

1. Install Python 3.10 or 3.12.

2. Install Python packages

    ```bash
    pip install -r requirements.txt

3. Install FFmpeg (System dependency)
   - Mac (brew):

    ```bash
    brew install ffmpeg
   ```

    - Windows:
    Download from https://ffmpeg.org/download.html

    Add ffmpeg/bin to system PATH.
4. Run
    ```bash
    python sound_analysis.py
    ```

## What this project does
- Extracts audio from Instagram videos
- Runs speech-to-text transcription using Whisper 
- Analyzes sentiment of spoken words
- Extracts background music mood
- Detects on-screen captions using OCR (EasyOCR)
- Saves all extracted features into a CSV file