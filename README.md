# Screen Chatbot

Talk to Gemini AI using your screen, camera, and voice.

## Quick Start

1. **Clone and install**:
   ```bash
   git clone https://github.com/jeff52415/screen-chatbot
   cd screen-chatbot
   
   # Choose one installation method:
   pip install -e .         # Development install
   # OR
   uv sync                  # Faster install with uv
   ```

2. **Set up your API key**:
   - Create a `.env` file with your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```
   - Get a key from: https://ai.google.dev/tutorials/setup
   - Tip: You can copy `env.txt` to `.env` and edit it as a template

3. **Run it**:
   ```bash
   # For voice conversations with audio responses:
   python src/live_stream_audio.py
   
   # For text conversations with more options:
   python src/stream_to_text.py
   ```

## Scripts

### live_stream_audio.py
- **Summary**: Creates a real-time voice conversation with Gemini. It continuously captures your screen or camera feed, records your voice through the microphone, and plays back Gemini's responses as audio. This creates a natural, conversational experience similar to talking with a person.
- **Input**: Your voice, screen or camera
- **Output**: Gemini's voice responses
- **Best for**: Natural conversations with voice
- **Note**: Use headphones to prevent feedback

### stream_to_text.py
- **Summary**: Provides a more flexible interaction with Gemini. It captures your screen or camera, can record audio from your microphone, and accepts text input. Gemini responds with text, which is displayed in the terminal. This script offers more control options during the session and automatically saves your entire conversation, including screenshots and audio recordings.
- **Input**: Your text, screen or camera, optional voice
- **Output**: Gemini's text responses
- **Best for**: Detailed interactions with more control
- **Note**: Saves conversation history and screenshots

## Features

- Capture your screen or camera
- Use your voice or type text
- Get text or voice responses
- Automatically save conversations


## Configuration

All settings can be configured in your `.env` file:

```
# Required
GEMINI_API_KEY=your_api_key_here

# Optional (defaults shown)
MODEL=gemini-2.0-flash-exp
RESPONSE_MODALITIES=TEXT  # or AUDIO
DEFAULT_QUERY="."
DEFAULT_VIDEO_MODE=screen  # none, screen, camera
DEFAULT_MONITOR=1
DEFAULT_AUDIO_ENABLED=true
AUDIO_DEVICE_INDEX=0
DEFAULT_AUDIO_CACHE_SECONDS=10
DEFAULT_DEBUG=false
```

## Need Help?
- Conversations are saved in the `exports/` folder if debug mode is activated.

