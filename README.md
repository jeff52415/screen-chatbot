# Screen Chatbot

A multimodal chatbot application that captures your screen, camera feed, and/or audio, allowing you to interact with Google's Gemini AI model through text or voice.

## Features

- **Multiple Input Modalities**:
  - Screen capture
  - Camera capture
  - Audio recording (microphone or system audio)
  - Text input
  
- **Multiple Output Modalities**:
  - Text responses
  - Audio responses (with `live_stream_audio.py`)
  
- **Data Export**:
  - Automatically saves conversations, screenshots, and audio recordings
  - Organizes data in conversation-specific folders

## Available Scripts

The project includes three main Python scripts, each with different capabilities:

### 1. `live_stream_audio.py`

Real-time bidirectional communication with Gemini AI:
- Captures your voice input
- Captures screenshots or camera feed
- Returns audio responses from Gemini
- Best used with headphones to prevent feedback

```bash
python src/live_stream_audio.py --mode [camera|screen|none]
```

### 2. `live_stream_text.py`

Multimodal interaction with text-based responses:
- Captures audio (from microphone or system audio)
- Captures screenshots or camera feed
- Triggered by text input
- Stores all input data in the exports folder
- Supports specific audio device selection

```bash
python src/live_stream_text.py --mode [camera|screen|none] --audio-device-index [index]
```

### 3. `stream_to_text.py`

Simplified interaction focused on visual input:
- Captures screenshots or camera feed
- Takes text input
- Returns text responses
- No audio processing

```bash
python src/stream_to_text.py --mode [camera|screen]
```

## Requirements

- Python 3.12 or higher 
- Google Gemini API key (get one from https://ai.google.dev/tutorials/setup)
- For audio capture: headphones is necessary 

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/screen-chatbot.git
   cd screen-chatbot
   ```

2. Install the required dependencies:

   Using pip (standard):
   ```bash
   pip install .
   ```
   
   Or using uv (faster alternative):
   ```bash
   # Install dependencies with uv
   uv sync
   ```
   
   Learn more about uv at: https://pypi.org/project/uv/

3. Create a `.env` file in the project root with your configuration (see `env.txt` for reference):
   ```
   GEMINI_API_KEY=your_api_key_here
   MODEL=gemini-2.0-flash-exp
   RESPONSE_MODALITIES=TEXT  # or AUDIO
   SYSTEM_PROMPT="You are a helpful assistant and answer in a friendly tone."
   DEFAULT_QUERY="."
   ```


## Data Export

`live_stream_text.py` automatically exports conversation data to the `exports/[conversation_name]` directory, including:
- Text queries
- Screenshots
- Audio recordings
- Metadata

## Interaction

1. Start the application with your preferred script
2. Type your question or message and press Enter
3. The application will capture the specified inputs and send them to Gemini
4. View or listen to Gemini's response
5. Type 'q' to quit the application


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

