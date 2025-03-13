# Screen Chatbot

A desktop application that captures your screen or camera feed and allows you to interact with Google's Gemini multimodal AI model through text.

![Screen Chatbot Screenshot](https://i.imgur.com/placeholder.png)

## Features

- Real-time screen or camera capture
- Text-based interaction with Gemini AI
- Audio input support
- Multiple capture modes:
  - Screen capture
  - Camera capture
  - Text-only mode

## Requirements

- Python 3.11 or higher (3.7+ with taskgroup/exceptiongroup packages)
- Google Gemini API key (get one from https://ai.google.dev/tutorials/setup)
- Headphones recommended (to prevent audio feedback)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/screen-chatbot.git
   cd screen-chatbot
   ```

2. Install the required dependencies:
   ```bash
   pip install google-genai opencv-python pyaudio pillow mss python-dotenv
   ```

3. Create a `.env` file in the project root with your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   MODEL=gemini-2.0-flash
   ```

## Usage

Run the application with one of the following commands:

### Screen Capture Mode (Default)
```bash
python screen_chatbot/multi_to_text.py
```
or
```bash
python screen_chatbot/multi_to_text.py --mode screen
```

### Camera Capture Mode
```bash
python screen_chatbot/multi_to_text.py --mode camera
```

### Text-Only Mode
```bash
python screen_chatbot/multi_to_text.py --mode none
```

### Selecting a Different Monitor
If you have multiple monitors, you can specify which one to capture:
```bash
python screen_chatbot/multi_to_text.py --monitor 1
```
Where 0 is the primary monitor, 1 is secondary, etc.

## Interaction

1. After starting the application, you'll see a "You: " prompt
2. Type your question or message and press Enter
3. The application will:
   - Capture your screen or camera (depending on mode)
   - Send your text input and the captured image to Gemini
   - Display Gemini's response as text
4. Continue the conversation by typing at the "You: " prompt
5. Type 'q' to quit the application

**Important:** Use headphones to prevent audio feedback, as the application uses your system's default audio input and output.

## API Reference

This application uses the Google Generative AI Python SDK to interact with the Gemini API. For more information, see:
- [Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
- [Python SDK Reference](https://ai.google.dev/api/python/google/generativeai)
- [Multimodal Capabilities](https://ai.google.dev/gemini-api/docs/multimodal)

## Troubleshooting

- If you encounter an error related to the API key, make sure you've entered a valid Gemini API key in the .env file.
- For Python versions below 3.11, ensure you have the taskgroup and exceptiongroup packages installed.
- If you have issues with screen capture, try specifying a different monitor with the --monitor flag.
- For camera capture issues, ensure your webcam is properly connected and accessible.

## License

Apache License 2.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.