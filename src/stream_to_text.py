#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import asyncio
import collections
import datetime
import io
import json
import os
import random
import string
import time
import traceback
import wave
from typing import Any, Dict, List, Optional, Tuple

import aiofiles
import cv2
import mss
import PIL.Image
import pyaudio
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import Field
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()


# Configuration model using Pydantic
class Config(BaseSettings):
    """Application configuration settings loaded from environment variables."""

    video_mode: str = Field(default="screen")
    monitor: int = Field(default=1)
    default_query: str = Field(default=".")
    model: str = Field(default="gemini-2.0-flash-exp")
    streaming: bool = Field(default=True)
    audio_enabled: bool = Field(default=True)
    audio_cache_seconds: int = Field(default=10)
    audio_device_index: int = Field(default=0)
    debug: bool = Field(default=False)

    class Config:
        # This tells Pydantic to read from environment variables
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Allow extra fields
        extra = "ignore"
        # Use environment variable names with specific prefixes
        env_prefix = "DEFAULT_"
        # These fields have different environment variable names
        env_mapping = {"model": "MODEL", "audio_device_index": "AUDIO_DEVICE_INDEX"}


# Create config instance
config = Config()


# Audio configuration constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000  # Gemini expects 16kHz audio
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

# Initialize PyAudio
pya = pyaudio.PyAudio()


class StreamToTextChatbot:
    """
    A chatbot that captures screenshots and sends them with text to the Gemini API.

    This class provides a simpler interface compared to the AudioLoop class,
    focusing only on screenshot capture and text interaction.
    """

    def __init__(
        self,
        video_mode: str = config.video_mode,
        monitor_index: int = config.monitor,
        default_query: str = config.default_query,
        model: str = config.model,
        streaming: bool = config.streaming,
        audio_enabled: bool = config.audio_enabled,
        audio_device_index: Optional[int] = config.audio_device_index,
        audio_cache_seconds: int = config.audio_cache_seconds,
        conversation_name: Optional[str] = None,
        debug: bool = config.debug,
    ) -> None:
        """
        Initialize the StreamToTextChatbot with specified settings.

        Args:
            video_mode: Mode for video capture ("screen" or "camera" or "none")
            monitor_index: Index of the monitor to capture (0-based)
            default_query: Default query to use when no text is provided
            model: Gemini model to use for generating responses
            streaming: Whether to use streaming mode for responses
            audio_enabled: Whether to capture and send audio
            audio_device_index: Index of the audio device to use
            audio_cache_seconds: Number of seconds of audio to cache
            conversation_name: Name for the conversation subfolder (random if None)
            debug: Whether to print debug information
        """
        self.video_mode = video_mode
        self.monitor_index = monitor_index
        self.default_query = default_query
        self.model = model
        self.streaming = streaming
        self.audio_enabled = audio_enabled
        self.audio_device_index = audio_device_index
        self.audio_cache_seconds = audio_cache_seconds
        self.debug = debug
        # Create export directory with subfolder
        self.export_dir = "exports"
        os.makedirs(self.export_dir, exist_ok=True)

        # Generate conversation name if not provided
        if conversation_name is None:
            # Generate a random name with timestamp and random characters
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            random_chars = "".join(
                random.choices(string.ascii_lowercase + string.digits, k=6)
            )
            self.conversation_name = f"conversation_{timestamp}_{random_chars}"
        else:
            self.conversation_name = conversation_name

        # Create subfolder for this conversation
        self.conversation_dir = os.path.join(self.export_dir, self.conversation_name)
        os.makedirs(self.conversation_dir, exist_ok=True)
        print(f"Exporting conversation data to: {self.conversation_dir}")

        # Add export session ID (timestamp)
        self.session_id = int(time.time())
        self.export_count = 0

        # Initialize Gemini client
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        # Initialize camera if needed
        self.camera = None
        if self.video_mode == "camera":
            self.camera = cv2.VideoCapture(0)

        # Always initialize audio cache regardless of whether audio is enabled
        self.audio_cache = collections.deque(
            maxlen=int(SEND_SAMPLE_RATE * self.audio_cache_seconds / CHUNK_SIZE)
        )

        # Initialize audio stream if audio is enabled
        self.audio_stream = None
        if self.audio_enabled:
            self._setup_audio()

    def _setup_audio(self) -> None:
        """Set up audio capture with the specified device."""
        try:
            if self.debug:
                print("\nSetting up audio capture...")
                print(f"Audio enabled: {self.audio_enabled}")

                # List available devices for debugging
                print("Available audio devices:")
            for i in range(pya.get_device_count()):
                info = pya.get_device_info_by_index(i)
                device_type = []
                if info["maxInputChannels"] > 0:
                    device_type.append("INPUT")
                if info["maxOutputChannels"] > 0:
                    device_type.append("OUTPUT")
                device_type = "+".join(device_type)
                if self.debug:
                    print(f"  [{i}] {info['name']} ({device_type})")

            # If a specific device index was provided, use it
            if self.audio_device_index is not None:
                try:
                    device_info = pya.get_device_info_by_index(self.audio_device_index)
                    self.audio_source = device_info["name"]
                    print(
                        f"Using specified audio device: {self.audio_source} (index: {self.audio_device_index})"
                    )
                except Exception as e:
                    print(
                        f"\nError using specified device index {self.audio_device_index}: {e}"
                    )
                    print("Falling back to default device selection")
                    # fallback to default device selection
                    self.audio_device_index = 0
                    device_info = pya.get_device_info_by_index(self.audio_device_index)
                    self.audio_source = device_info["name"]
            else:
                # Use default input device
                device_info = pya.get_default_input_device_info()
                self.audio_source = device_info["name"]
                self.audio_device_index = device_info["index"]
                print(
                    f"Using default audio device: {self.audio_source} (index: {self.audio_device_index})"
                )

            # Check if the device has input channels
            if device_info["maxInputChannels"] <= 0:
                print(
                    "Warning: Selected device has no input channels. Audio capture may not work."
                )

            # Open the audio stream
            self.audio_stream = pya.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SEND_SAMPLE_RATE,
                input=True,
                input_device_index=self.audio_device_index,
                frames_per_buffer=CHUNK_SIZE,
            )

            if self.debug:
                print("Audio stream opened successfully!")
                print(f"Audio cache size: {self.audio_cache_seconds} seconds")

        except Exception as e:
            print(f"Error setting up audio: {e}")
            traceback.print_exc()
            print("Audio capture has been disabled due to errors.")
            self.audio_enabled = False
            self.audio_stream = None

    def __del__(self) -> None:
        """Clean up resources when the object is destroyed."""
        if self.camera is not None:
            self.camera.release()

        if self.audio_stream is not None:
            self.audio_stream.close()

    async def capture_screenshot(self) -> Tuple[str, bytes]:
        """
        Capture a screenshot or camera frame and save it.

        Returns:
            Tuple containing the filename and the image bytes
        """
        # Get current timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.jpg"

        image_bytes = None

        try:
            if self.video_mode == "camera":
                # Capture from camera
                if self.camera is None or not self.camera.isOpened():
                    self.camera = cv2.VideoCapture(0)

                ret, frame = self.camera.read()
                if not ret:
                    raise Exception("Failed to capture frame from camera")

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = PIL.Image.fromarray(frame_rgb)

                # Resize if needed
                img.thumbnail([1024, 1024])

                # Get image bytes for API
                image_io = io.BytesIO()
                img.save(image_io, format="jpeg")
                image_io.seek(0)
                image_bytes = image_io.read()

            elif self.video_mode == "screen":  # screen mode
                # Capture from screen
                with mss.mss() as sct:
                    monitor = sct.monitors[self.monitor_index]
                    sct_img = sct.grab(monitor)

                    # Convert to PIL Image
                    img = PIL.Image.frombytes("RGB", sct_img.size, sct_img.rgb)

                    # Resize if needed (Gemini has image size limits)
                    img.thumbnail([1024, 1024])

                    # Get image bytes for API
                    image_io = io.BytesIO()
                    img.save(image_io, format="jpeg")
                    image_io.seek(0)
                    image_bytes = image_io.read()
            else:
                print("No video mode selected, skipping image capture")
                return "", b""

            return filename, image_bytes

        except Exception as e:
            print(f"\nError capturing screenshot: {e}")
            return "", b""

    def capture_audio(self) -> List[Dict]:
        """
        Capture audio from the microphone and store it in the cache.

        Returns:
            List of audio chunks in the cache
        """
        if not self.audio_enabled or self.audio_stream is None:
            return []

        try:
            # Read audio data until we have filled the cache
            kwargs = {"exception_on_overflow": False} if __debug__ else {}

            # Read a chunk of audio
            data = self.audio_stream.read(CHUNK_SIZE, **kwargs)

            # Check if we got valid data
            if (
                data and len(data) == CHUNK_SIZE * 2
            ):  # 16-bit audio = 2 bytes per sample
                audio_chunk = {"data": data, "mime_type": "audio/pcm"}

                # Add to cache
                self.audio_cache.append(audio_chunk)

            else:
                print(
                    f"Warning: Received invalid audio data of length {len(data) if data else 0}"
                )

            # Return a copy of the cache
            return list(self.audio_cache)
        except Exception as e:
            print(f"Error capturing audio: {e}")
            traceback.print_exc()
            return []

    def create_wav_from_chunks(self, audio_chunks: List[Dict]) -> bytes:
        """
        Create a WAV file from audio chunks.

        Args:
            audio_chunks: List of audio chunks

        Returns:
            WAV file as bytes
        """
        if not audio_chunks:
            return b""

        # Create a WAV file in memory with the audio data
        audio_io = io.BytesIO()
        with wave.open(audio_io, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pya.get_sample_size(FORMAT))
            wf.setframerate(SEND_SAMPLE_RATE)

            # Write all audio chunks
            for chunk in audio_chunks:
                wf.writeframes(chunk["data"])

        # Reset the buffer position
        audio_io.seek(0)

        # Return the WAV file as bytes
        return audio_io.read()

    def prepare_contents(
        self, prompt: str, image_bytes: bytes, audio_bytes: bytes
    ) -> List[Any]:
        """
        Prepare the contents for the Gemini API in the correct order.

        Args:
            prompt: Text prompt
            image_bytes: Image bytes
            audio_bytes: Audio bytes

        Returns:
            List of contents for the Gemini API
        """
        contents = []

        # 1. First add image content (if available)
        if image_bytes:
            contents.append(
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            )

        # 2. Then add audio content (if available)
        if audio_bytes:
            contents.append(
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
            )

        # 3. Finally add text prompt
        contents.append(prompt)

        return contents

    async def send_to_gemini(
        self, prompt: str, image_bytes: bytes, audio_chunks: List[Dict] = None
    ) -> Tuple[str, Dict[str, int]]:
        """
        Send the prompt, image, and audio to the Gemini API.

        Args:
            prompt: Text prompt to send
            image_bytes: Image bytes to send
            audio_chunks: List of audio chunks to send

        Returns:
            Tuple containing response text and token usage dictionary
        """
        try:
            # Convert audio chunks to WAV file
            audio_bytes = b""
            if audio_chunks and len(audio_chunks) > 0:
                audio_duration = len(audio_chunks) * CHUNK_SIZE / SEND_SAMPLE_RATE
                if self.debug:
                    print(
                        f"Sending {audio_duration:.2f} seconds of audio ({len(audio_chunks)} chunks)"
                    )
                audio_bytes = self.create_wav_from_chunks(audio_chunks)
            else:
                if self.debug:
                    print("No audio chunks to send")

            # Prepare contents in the correct order
            contents = self.prepare_contents(prompt, image_bytes, audio_bytes)

            if not contents:
                return "Error: No content to send to Gemini API", {
                    "input_tokens": 0,
                    "output_tokens": 0,
                }

            if self.debug:
                print(f"Sending to Gemini API with {len(contents)} content parts")

            # Initialize token counters for this interaction
            token_usage = {"input_tokens": 0, "output_tokens": 0}

            if self.streaming:
                # Use streaming mode
                print("\nGemini: ", end="", flush=True)
                response_text = ""

                async for chunk in await self.client.aio.models.generate_content_stream(
                    model=self.model, contents=contents
                ):
                    if chunk.text:
                        print(chunk.text, end="", flush=True)
                        response_text += chunk.text

                    # Track token usage if available
                    if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                        if hasattr(chunk.usage_metadata, "prompt_token_count"):
                            token_usage["input_tokens"] = (
                                chunk.usage_metadata.prompt_token_count or 0
                            )
                        if hasattr(chunk.usage_metadata, "candidates_token_count"):
                            token_usage["output_tokens"] += (
                                chunk.usage_metadata.candidates_token_count or 0
                            )

                return response_text, token_usage
            else:
                # Use non-streaming mode
                response = self.client.models.generate_content(
                    model=self.model, contents=contents
                )

                # Track token usage if available
                print("response usage_metadata: ", response.usage_metadata)
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    if hasattr(response.usage_metadata, "prompt_token_count"):
                        token_usage["input_tokens"] = (
                            response.usage_metadata.prompt_token_count or 0
                        )
                    if hasattr(response.usage_metadata, "candidates_token_count"):
                        token_usage["output_tokens"] = (
                            response.usage_metadata.candidates_token_count or 0
                        )

                # Return the response text and token usage
                return response.text, token_usage
        except Exception as e:
            print(f"\nError sending to Gemini API: {e}")
            traceback.print_exc()
            return f"Error: {str(e)}", {"input_tokens": 0, "output_tokens": 0}

    async def load_token_usage(self) -> Dict[str, Any]:
        """
        Load token usage data from file if it exists.

        Returns:
            Dictionary containing token usage data
        """
        token_usage_file = os.path.join(self.conversation_dir, "token_usage.json")

        # Default structure if file doesn't exist
        default_data = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "interactions": [],
        }

        try:
            if os.path.exists(token_usage_file):
                async with aiofiles.open(token_usage_file, "r", encoding="utf-8") as f:
                    content = await f.read()
                    return json.loads(content)
            return default_data
        except Exception as e:
            print(f"\nError loading token usage data: {e}")
            traceback.print_exc()
            return default_data

    async def update_token_usage(self, token_usage: Dict[str, int]) -> Dict[str, Any]:
        """
        Update the token usage tracking for the conversation.

        Args:
            token_usage: Dictionary with input_tokens and output_tokens

        Returns:
            Updated token usage data
        """
        # Load current token usage data
        conversation_token_usage = await self.load_token_usage()

        # Update total counts
        conversation_token_usage["total_input_tokens"] += token_usage["input_tokens"]
        conversation_token_usage["total_output_tokens"] += token_usage["output_tokens"]

        # Add this interaction
        timestamp = datetime.datetime.now().isoformat()
        conversation_token_usage["interactions"].append(
            {
                "timestamp": timestamp,
                "input_tokens": token_usage["input_tokens"],
                "output_tokens": token_usage["output_tokens"],
            }
        )

        # Save the updated token usage to a file
        await self.save_token_usage(conversation_token_usage)

        return conversation_token_usage

    async def save_token_usage(self, token_usage_data: Dict[str, Any]) -> None:
        """
        Save the token usage data to a file.

        Args:
            token_usage_data: Token usage data to save
        """
        try:
            # Create the token usage file path
            token_usage_file = os.path.join(self.conversation_dir, "token_usage.json")

            # Save the token usage data
            async with aiofiles.open(token_usage_file, "w", encoding="utf-8") as f:
                await f.write(
                    json.dumps(token_usage_data, indent=2, ensure_ascii=False)
                )
        except Exception as e:
            print(f"\nError saving token usage data: {e}")
            traceback.print_exc()

    async def export_interaction_data(
        self,
        text: str,
        image_bytes: bytes = None,
        audio_chunks: List[Dict] = None,
        token_usage: Dict[str, int] = None,
        response_text: str = None,
    ) -> None:
        """
        Export the current interaction data (text, image, audio) to files.

        Args:
            text: The text query being sent
            image_bytes: Image bytes data
            audio_chunks: List of audio chunks
            token_usage: Dictionary with input_tokens and output_tokens
            response_text: The model's response text
        """
        try:
            # Create timestamp and increment counter for unique filenames
            self.export_count += 1
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{self.conversation_dir}/export_{self.session_id}_{self.export_count}_{timestamp}"

            # Export text
            if self.debug:
                async with aiofiles.open(
                    f"{base_filename}_text.txt", "w", encoding="utf-8"
                ) as f:
                    await f.write(text)

            # Export screenshot if available
            if image_bytes and self.debug:
                # Save the image bytes directly to a file
                async with aiofiles.open(f"{base_filename}_image.jpg", "wb") as f:
                    await f.write(image_bytes)

            # Export audio data
            if audio_chunks and self.debug:
                # For wave files, we need to use synchronous IO as wave module doesn't support async
                # Create a WAV file with the cached audio
                with wave.open(f"{base_filename}_audio.wav", "wb") as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(pya.get_sample_size(FORMAT))
                    wf.setframerate(SEND_SAMPLE_RATE)

                    # Combine all audio chunks
                    for audio_chunk in audio_chunks:
                        wf.writeframes(audio_chunk["data"])

            # Get audio source name, ensuring it's properly encoded
            audio_source = None
            if self.audio_enabled:
                if hasattr(self, "audio_source"):
                    # Ensure the audio source name is properly encoded
                    audio_source = str(self.audio_source)
                else:
                    audio_source = str(self.audio_device_index)

            # Create a metadata file with information about the export
            metadata = {
                "timestamp": timestamp,
                "session_id": self.session_id,
                "export_id": self.export_count,
                "text_query": text,
                "video_mode": self.video_mode,
                "audio_enabled": self.audio_enabled,
                "audio_source": audio_source,
                "audio_duration_seconds": (
                    len(audio_chunks) * CHUNK_SIZE / SEND_SAMPLE_RATE
                    if audio_chunks
                    else 0
                ),
                "conversation_name": self.conversation_name,
                "model": self.model,
            }

            # Add response text if available
            if response_text:
                metadata["response_text"] = response_text

            # Add token usage if available
            if token_usage:
                metadata["token_usage"] = {
                    "input_tokens": token_usage["input_tokens"],
                    "output_tokens": token_usage["output_tokens"],
                }

            if self.debug:
                async with aiofiles.open(
                    f"{base_filename}_metadata.json", "w", encoding="utf-8"
                ) as f:
                    await f.write(json.dumps(metadata, indent=2, ensure_ascii=False))

        except Exception as e:
            print(f"\nError exporting interaction data: {e}")
            traceback.print_exc()

    async def run(self) -> None:
        """
        Main loop that handles user input and sends requests to the Gemini API.
        """
        print("\nWelcome to Stream-to-Text Chatbot!")
        print(
            "Type your question and press Enter to capture a screenshot/audio and get a response."
        )

        # Print configuration for debugging
        print("Configuration:")
        print(f"  DEBUG: {self.debug}")
        print(f"  MONITOR INDEX: {self.monitor_index}")
        print(f"  DEFAULT QUERY: {self.default_query}")
        print(f"  MODEL: {self.model}")
        print(f"  STREAMING: {self.streaming}")
        print(f"  AUDIO ENABLED: {self.audio_enabled}")
        print(f"  AUDIO_CACHE_SECONDS: {self.audio_cache_seconds}")
        if self.video_mode in ["screen", "camera"]:
            print(f"  VIDEO_MODE: {self.video_mode} (enabled)")
        else:
            print(f"  VIDEO_MODE: {self.video_mode} (disabled)")

        print("Type 'q' to quit.")
        print("Type 'toggle' to toggle streaming mode.")
        print("Type 'audio' to toggle audio capture.")
        print("Type 'mode' to toggle video mode (none/screen/camera).")
        print("Type 'test_audio' to test audio capture and save a sample.")
        print("Type 'token_usage' to display current token usage statistics.")

        # Start continuously capturing audio if enabled
        audio_cache_active = True if self.audio_enabled else False

        # Start continuous audio recording in a separate thread if enabled
        if self.audio_enabled and audio_cache_active:
            self._start_continuous_audio_recording()

        while True:
            # Get user input
            print("\nYou: ", end="", flush=True)
            text = input()

            # Check if user wants to quit
            if text.lower() == "q":
                break

            # Check if user wants to toggle streaming mode
            if text.lower() == "toggle":
                self.streaming = not self.streaming
                print(f"Streaming mode {'enabled' if self.streaming else 'disabled'}.")
                continue

            # Check if user wants to toggle audio capture
            if text.lower() == "audio":
                if not self.audio_enabled:
                    self.audio_enabled = True
                    if self.audio_stream is None:
                        self._setup_audio()
                    audio_cache_active = True
                    self._start_continuous_audio_recording()
                    print("Audio capture enabled.")
                else:
                    audio_cache_active = not audio_cache_active
                    if audio_cache_active:
                        self._start_continuous_audio_recording()
                    print(
                        f"Audio capture {'enabled' if audio_cache_active else 'paused'}."
                    )
                continue

            # Check if user wants to toggle video mode
            if text.lower() == "mode":
                if self.video_mode == "none":
                    self.video_mode = "screen"
                elif self.video_mode == "screen":
                    self.video_mode = "camera"
                    # Initialize camera if needed
                    if self.camera is None:
                        self.camera = cv2.VideoCapture(0)
                else:
                    self.video_mode = "none"
                print(f"Video mode set to: {self.video_mode}")
                continue

            # Check if user wants to test audio
            if text.lower() == "test_audio":
                await self.test_audio_capture()
                continue

            # Check if user wants to see token usage
            if text.lower() == "token_usage":
                token_data = await self.load_token_usage()
                print("\nToken Usage Statistics:")
                print(f"Total Input Tokens: {token_data['total_input_tokens']}")
                print(f"Total Output Tokens: {token_data['total_output_tokens']}")
                print(f"Total Interactions: {len(token_data['interactions'])}")
                if token_data["interactions"]:
                    print("\nRecent interactions:")
                    for i in range(min(5, len(token_data["interactions"]))):
                        interaction = token_data["interactions"][-(i + 1)]
                        print(
                            f"  {interaction['timestamp']}: Input={interaction['input_tokens']}, Output={interaction['output_tokens']}"
                        )
                continue

            # Use default prompt if user didn't type anything
            if not text:
                text = self.default_query
                print(f"Using default query: {text}")

            # Capture screenshot if video mode is enabled
            image_bytes = b""
            if self.video_mode in ["screen", "camera"]:
                filename, image_bytes = await self.capture_screenshot()
                if not image_bytes:
                    print("Failed to capture screenshot, continuing without image.")

            # Get audio data if enabled
            audio_chunks = None
            if self.audio_enabled and audio_cache_active:
                # Get a snapshot of the current audio cache
                audio_chunks = list(self.audio_cache) if self.audio_cache else []

            # Send to Gemini API
            if self.debug:
                print("Sending to Gemini API...")

            # Start timing the response
            start_time = time.time()

            # Get response from Gemini
            response, token_usage = await self.send_to_gemini(
                text, image_bytes, audio_chunks
            )

            # Update token usage tracking
            _ = await self.update_token_usage(token_usage)

            # Update the metadata file with token usage and response
            await self.export_interaction_data(
                text, image_bytes, audio_chunks, token_usage, response
            )

            # Calculate and display response time
            elapsed_time = time.time() - start_time
            if self.debug:
                print(f"\nResponse time: {elapsed_time:.2f} seconds")

            # If not streaming, print the response
            if not self.streaming:
                print("\nGemini: " + response)

        # Clean up continuous audio recording thread if it exists
        if (
            hasattr(self, "audio_recording_thread")
            and self.audio_recording_thread is not None
        ):
            self.stop_audio_recording = True
            self.audio_recording_thread.join(timeout=1.0)

        print("\nExiting...")

    def _start_continuous_audio_recording(self) -> None:
        """
        Start continuous audio recording in a separate thread.
        This ensures audio is captured continuously without blocking the main thread.
        """
        import threading

        # Stop existing recording thread if it exists
        if (
            hasattr(self, "audio_recording_thread")
            and self.audio_recording_thread is not None
        ):
            self.stop_audio_recording = True
            self.audio_recording_thread.join(timeout=1.0)

        # Reset the flag
        self.stop_audio_recording = False

        # Define the recording function
        def record_audio_continuously():
            print("Starting continuous audio recording...")

            if self.audio_stream is None:
                print("Audio stream is not initialized. Setting up audio...")
                self._setup_audio()
                if self.audio_stream is None:
                    print(
                        "Failed to initialize audio stream. Audio recording disabled."
                    )
                    return

            # Clear the audio cache to start fresh
            self.audio_cache.clear()

            kwargs = {"exception_on_overflow": False} if __debug__ else {}

            try:
                while not getattr(self, "stop_audio_recording", False):
                    # Read a chunk of audio
                    data = self.audio_stream.read(CHUNK_SIZE, **kwargs)

                    # Check if we got valid data
                    if (
                        data and len(data) == CHUNK_SIZE * 2
                    ):  # 16-bit audio = 2 bytes per sample
                        audio_chunk = {"data": data, "mime_type": "audio/pcm"}

                        # Add to cache
                        self.audio_cache.append(audio_chunk)

                    # Small sleep to prevent CPU overload
                    time.sleep(0.001)
            except Exception as e:
                print(f"Error in continuous audio recording: {e}")
                traceback.print_exc()
            finally:
                print("Continuous audio recording stopped.")

        # Start the recording thread
        self.audio_recording_thread = threading.Thread(target=record_audio_continuously)
        self.audio_recording_thread.daemon = (
            True  # Make thread exit when main program exits
        )
        self.audio_recording_thread.start()

    async def test_audio_capture(self) -> None:
        """Test audio capture and save a sample."""
        if not self.audio_enabled:
            print(
                "Audio capture is not enabled. Use --audio flag when starting the program."
            )
            return

        if self.audio_stream is None:
            print(
                "Audio stream is not initialized. There might be an issue with your audio setup."
            )
            return

        print("\nTesting audio capture...")
        print("Recording 5 seconds of audio...")

        # Clear the audio cache
        if self.audio_cache:
            self.audio_cache.clear()

        # Record for 5 seconds
        start_time = time.time()
        while time.time() - start_time < 5:
            try:
                kwargs = {"exception_on_overflow": False} if __debug__ else {}
                data = self.audio_stream.read(CHUNK_SIZE, **kwargs)
                audio_chunk = {"data": data, "mime_type": "audio/pcm"}
                self.audio_cache.append(audio_chunk)
                # Show progress
                if int((time.time() - start_time) * 10) % 10 == 0:
                    print(".", end="", flush=True)
                await asyncio.sleep(
                    0.01
                )  # Small sleep to prevent CPU overload, using asyncio.sleep
            except Exception as e:
                print(f"\nError during audio capture: {e}")
                return

        print("\nRecording complete!")

        # Save the audio to a file
        if self.audio_cache:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audio_test_{timestamp}.wav"
            filepath = os.path.join(self.conversation_dir, filename)

            try:
                # Wave module doesn't support async IO, so we use synchronous IO here
                with wave.open(filepath, "wb") as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(pya.get_sample_size(FORMAT))
                    wf.setframerate(SEND_SAMPLE_RATE)

                    # Write all audio chunks
                    for chunk in self.audio_cache:
                        wf.writeframes(chunk["data"])

                print(f"Audio sample saved to: {filepath}")
                print(
                    f"Audio duration: {len(self.audio_cache) * CHUNK_SIZE / SEND_SAMPLE_RATE:.2f} seconds"
                )

                # Create metadata
                metadata = {
                    "timestamp": timestamp,
                    "audio_device": getattr(
                        self, "audio_source", str(self.audio_device_index)
                    ),
                    "sample_rate": SEND_SAMPLE_RATE,
                    "channels": CHANNELS,
                    "format": "PCM_16",
                    "duration_seconds": len(self.audio_cache)
                    * CHUNK_SIZE
                    / SEND_SAMPLE_RATE,
                    "chunks": len(self.audio_cache),
                    "chunk_size": CHUNK_SIZE,
                }

                # Save metadata
                metadata_path = os.path.join(
                    self.conversation_dir, f"audio_test_{timestamp}_metadata.json"
                )
                async with aiofiles.open(metadata_path, "w", encoding="utf-8") as f:
                    await f.write(json.dumps(metadata, indent=2))

                print("Audio test completed successfully!")
            except Exception as e:
                print(f"Error saving audio sample: {e}")
                traceback.print_exc()
        else:
            print("No audio data captured. Check your microphone and audio settings.")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Stream-to-Text Chatbot with Gemini API"
    )
    parser.add_argument(
        "--video-mode",
        type=str,
        default=config.video_mode,
        help="Source of visual input",
        choices=["camera", "screen", "none"],
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=config.monitor,
        help="Monitor index to capture (0 is primary, 1 is secondary, etc.)",
    )
    parser.add_argument(
        "--model", type=str, default=config.model, help="Gemini model to use"
    )
    parser.add_argument(
        "--streaming",
        type=bool,
        default=config.streaming,
        help="Enable streaming mode (true/false)",
    )
    parser.add_argument(
        "--audio-enabled",
        type=bool,
        default=config.audio_enabled,
        help="Enable audio capture (true/false)",
    )
    parser.add_argument(
        "--audio-device-index",
        type=int,
        default=config.audio_device_index,
        help="Specific audio device index to use (overrides automatic selection)",
    )
    parser.add_argument(
        "--audio-cache",
        type=int,
        default=config.audio_cache_seconds,
        help=f"Seconds of audio to cache before sending (default: {config.audio_cache_seconds})",
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="List all audio devices and exit"
    )
    parser.add_argument(
        "--conversation-name",
        type=str,
        default=None,
        help="Name for the conversation subfolder (random if not specified)",
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=config.debug,
        help="Enable debug mode (true/false)",
    )
    return parser.parse_args()


async def main():
    args = parse_arguments()

    # If --list-devices is specified, just list devices and exit
    if args.list_devices:
        print("Available audio devices:")
        for i in range(pya.get_device_count()):
            info = pya.get_device_info_by_index(i)
            device_type = []
            if info["maxInputChannels"] > 0:
                device_type.append("INPUT")
            if info["maxOutputChannels"] > 0:
                device_type.append("OUTPUT")
            device_type = "+".join(device_type)
            print(f"  [{i}] {info['name']} ({device_type})")
        return

    chatbot = StreamToTextChatbot(
        video_mode=args.video_mode,
        monitor_index=args.monitor,
        model=args.model,
        streaming=args.streaming,
        audio_enabled=args.audio_enabled,
        audio_device_index=args.audio_device_index,
        audio_cache_seconds=args.audio_cache,
        conversation_name=args.conversation_name,
        debug=args.debug,
    )

    try:
        await chatbot.run()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")


if __name__ == "__main__":
    asyncio.run(main())
