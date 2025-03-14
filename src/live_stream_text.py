# -*- coding: utf-8 -*-
"""
Multimodal Chatbot with Screen/Camera Capture and Audio Processing

This module implements a chatbot that can capture screen or camera input,
process audio, and interact with the Gemini API to provide responses.
"""

import argparse
import asyncio
import base64
import collections
import datetime
import io
import json
import os
import random
import string
import sys
import time
import traceback
import wave
from typing import Any, Deque, Dict, Optional

import cv2
import mss
import PIL.Image
import pyaudio
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Backport for Python < 3.11
if sys.version_info < (3, 11, 0):
    import exceptiongroup
    import taskgroup

    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

# Audio configuration constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

# Default settings
DEFAULT_MODE = None
DEFAULT_MONITOR = 1  # Default to the primary monitor
DEFAULT_QUERY = os.getenv("DEFAULT_QUERY", ".")


# Initialize Gemini client
client = genai.Client(
    http_options={"api_version": "v1alpha"}, api_key=os.getenv("GEMINI_API_KEY")
)


CONFIG = {
    "system_instruction": types.Content(
        parts=[types.Part(text=os.getenv("SYSTEM_PROMPT", ""))]
    ),
    "response_modalities": [os.getenv("RESPONSE_MODALITIES")],
}

# Initialize PyAudio
pya = pyaudio.PyAudio()


class MultimodalChatbot:
    """
    Main class for handling audio, video, and text interactions with Gemini API.

    This class manages the capture of screen/camera input, audio recording,
    and communication with the Gemini API for multimodal conversations.
    """

    def __init__(
        self,
        video_mode: str = DEFAULT_MODE,
        monitor_index: int = DEFAULT_MONITOR,
        default_query: str = DEFAULT_QUERY,
        conversation_name: Optional[str] = None,
        screenshot_dir: Optional[str] = None,
        audio_device_index: Optional[int] = None,
    ) -> None:
        """
        Initialize the MultimodalChatbot with specified settings.

        Args:
            video_mode: Mode for video capture ("screen", "camera", or "none")
            monitor_index: Index of the monitor to capture (0-based)
            default_query: Default query to use when no text is provided
            conversation_name: Name for the conversation subfolder (random if None)
        """
        self.video_mode = video_mode
        self.monitor_index = monitor_index
        self.default_query = default_query
        self.audio_device_index = audio_device_index

        self.audio_in_queue: Optional[asyncio.Queue] = None
        self.out_queue: Optional[asyncio.Queue] = None
        self.session: Optional[Any] = None  # Type will be AsyncGenerativeModel
        self.audio_stream: Optional[pyaudio.Stream] = None

        # Create screenshots directory if it doesn't exist
        self.screenshots_dir = screenshot_dir
        if self.screenshots_dir:
            os.makedirs(self.screenshots_dir, exist_ok=True)

        # Audio caching parameters
        self.audio_cache_seconds: int = 10  # Cache 10 seconds of audio by default
        self.audio_cache: Deque = collections.deque(
            maxlen=int(SEND_SAMPLE_RATE * self.audio_cache_seconds / CHUNK_SIZE)
        )
        self.send_audio_cache: bool = (
            True  # Flag to control whether to send cached audio
        )

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

    async def export_interaction_data(self, text: str) -> None:
        """
        Export the current interaction data (text, image, audio) to files.

        Args:
            text: The text query being sent
        """
        try:
            # Create timestamp and increment counter for unique filenames
            self.export_count += 1
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{self.conversation_dir}/export_{self.session_id}_{self.export_count}_{timestamp}"

            # Export text
            with open(f"{base_filename}_text.txt", "w", encoding="utf-8") as f:
                f.write(text)

            # Export screenshot if in screen or camera mode
            if self.video_mode in ["screen", "camera"]:
                # Use the most recent screenshot
                latest_screenshot = max(
                    [f for f in os.listdir(self.screenshots_dir) if f.endswith(".jpg")],
                    key=lambda x: os.path.getmtime(
                        os.path.join(self.screenshots_dir, x)
                    ),
                    default=None,
                )

                if latest_screenshot:
                    # Copy the screenshot to exports
                    import shutil

                    shutil.copy(
                        os.path.join(self.screenshots_dir, latest_screenshot),
                        f"{base_filename}_image.jpg",
                    )

            # Export audio data
            if self.audio_cache:
                # Create a WAV file with the cached audio
                with wave.open(f"{base_filename}_audio.wav", "wb") as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(pya.get_sample_size(FORMAT))
                    wf.setframerate(SEND_SAMPLE_RATE)

                    # Combine all audio chunks
                    for audio_chunk in self.audio_cache:
                        wf.writeframes(audio_chunk["data"])

            # Create a metadata file with information about the export
            metadata = {
                "timestamp": timestamp,
                "session_id": self.session_id,
                "export_id": self.export_count,
                "text_query": text,
                "video_mode": self.video_mode,
                "audio_source": getattr(self, "audio_source", self.audio_device_index),
                "audio_duration_seconds": (
                    len(self.audio_cache) * CHUNK_SIZE / SEND_SAMPLE_RATE
                    if self.audio_cache
                    else 0
                ),
                "conversation_name": self.conversation_name,
            }

            with open(f"{base_filename}_metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            print(f"\nExported interaction data to {base_filename}_*")

        except Exception as e:
            print(f"\nError exporting interaction data: {e}")
            traceback.print_exc()

    async def send_text(self) -> None:
        """
        Task that handles user text input and sends it to the Gemini API.

        This function captures user input from the terminal, saves a screenshot,
        and sends both the text and cached audio to the Gemini API.
        """
        # First prompt
        print("\nYou: ", end="", flush=True)

        while True:
            text = await asyncio.to_thread(
                input, ""
            )  # No prompt here since we print it in receive_audio
            if text.lower() == "q":
                break

            # Save screenshot before sending message
            if self.video_mode in ["screen", "camera"] and self.screenshots_dir:
                await self.save_screenshot()

            # Use default prompt if user didn't type anything
            if not text:
                text = self.default_query
                print(f"Using default query: {text}")

            # Export interaction data before sending
            await self.export_interaction_data(text)

            # Send cached audio to the model
            if self.send_audio_cache and self.session:
                cache_size = len(self.audio_cache)
                audio_duration = cache_size * CHUNK_SIZE / SEND_SAMPLE_RATE
                print(f"Sending {audio_duration:.2f} seconds of cached audio")

                # Send all cached audio chunks
                for audio_chunk in list(self.audio_cache):
                    await self.session.send(input=audio_chunk)

                # Clear the cache after sending
                self.audio_cache.clear()

            if self.session:
                await self.session.send(input=text, end_of_turn=True)

    async def save_screenshot(self) -> None:
        """
        Capture and save a screenshot with timestamp.

        The screenshot is saved to the screenshots directory with a timestamp
        in the filename.
        """
        try:
            # Get current timestamp for filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.screenshots_dir}/screenshot_{timestamp}.jpg"

            # Capture screenshot
            with mss.mss() as sct:
                monitor = sct.monitors[self.monitor_index]
                sct_img = sct.grab(monitor)

                # Convert to PIL Image and save
                img = PIL.Image.frombytes("RGB", sct_img.size, sct_img.rgb)
                img.save(filename)
                print(f"\nScreenshot saved: {filename}")
        except Exception as e:
            print(f"\nError saving screenshot: {e}")

    def _get_frame(self, cap: cv2.VideoCapture) -> Optional[Dict[str, str]]:
        """
        Capture a frame from the camera and convert it to the required format.

        Args:
            cap: OpenCV VideoCapture object

        Returns:
            Dictionary with mime_type and base64-encoded image data,
            or None if frame capture failed
        """
        # Read the frame
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            return None

        # Convert BGR to RGB color space
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self) -> None:
        """
        Task that captures frames from the camera and puts them in the output queue.
        """
        # This takes about a second, and will block the whole program
        # causing the audio pipeline to overflow if you don't to_thread it.
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)

        try:
            while True:
                frame = await asyncio.to_thread(self._get_frame, cap)
                if frame is None:
                    break

                await asyncio.sleep(1.0)

                if self.out_queue:
                    await self.out_queue.put(frame)
        finally:
            # Release the VideoCapture object
            cap.release()

    def _get_screen(self) -> Dict[str, str]:
        """
        Capture the screen and convert it to the required format.

        Returns:
            Dictionary with mime_type and base64-encoded image data
        """
        with mss.mss() as sct:
            # Use the specified monitor index
            monitor = sct.monitors[self.monitor_index]
            i = sct.grab(monitor)

            mime_type = "image/jpeg"
            image_bytes = mss.tools.to_png(i.rgb, i.size)
            img = PIL.Image.open(io.BytesIO(image_bytes))

            image_io = io.BytesIO()
            img.save(image_io, format="jpeg")
            image_io.seek(0)

            image_bytes = image_io.read()
            return {
                "mime_type": mime_type,
                "data": base64.b64encode(image_bytes).decode(),
            }

    async def get_screen(self) -> None:
        """
        Task that captures the screen and puts frames in the output queue.
        """
        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            if self.out_queue:
                await self.out_queue.put(frame)

    async def send_realtime(self) -> None:
        """
        Task that sends non-audio data from the output queue to the Gemini API.
        """
        if not self.out_queue:
            return

        while True:
            msg = await self.out_queue.get()
            # Only send non-audio data in realtime (images)
            if msg.get("mime_type") != "audio/pcm" and self.session:
                await self.session.send(input=msg)

    async def listen_audio(self) -> None:
        """
        Task that captures audio from the microphone or system audio and stores it in the cache.
        """

        # Check if a specific device index was provided
        device_index = getattr(self, "audio_device_index", None)

        if device_index is not None:
            # Use the specified device index
            try:
                device_info = pya.get_device_info_by_index(device_index)
            except Exception as e:
                print(f"\nError using specified device index {device_index}: {e}")
                print("Falling back to default device selection")
                # fallback to default device selection
                device_index = 0
                device_info = pya.get_device_info_by_index(device_index)

        self.audio_source = device_info["name"]
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK_SIZE,
        )

        kwargs = {"exception_on_overflow": False} if __debug__ else {}

        try:
            while True:
                data = await asyncio.to_thread(
                    self.audio_stream.read, CHUNK_SIZE, **kwargs
                )
                audio_chunk = {"data": data, "mime_type": "audio/pcm"}

                # Store in cache instead of sending directly
                self.audio_cache.append(audio_chunk)

                # Only put in queue if we're not caching
                if not self.send_audio_cache and self.out_queue:
                    await self.out_queue.put(audio_chunk)
        except Exception as e:
            print(f"Error in audio capture: {e}")
        finally:
            if self.audio_stream:
                self.audio_stream.close()

    async def receive_audio(self) -> None:
        """
        Task that receives and processes responses from the Gemini API.

        This function collects text responses from the API and prints them
        to the terminal.
        """
        if not self.session or not self.audio_in_queue:
            return

        while True:
            turn = self.session.receive()
            response_text = ""

            # Collect all text from this turn
            async for response in turn:
                if text := response.text:
                    response_text += text

            # Only print if we got some text
            if response_text:
                print("\nGemini: " + response_text)
                # Add the "You: " prompt after Gemini's response
                print("\nYou: ", end="", flush=True)

            # Empty the audio queue
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def run(self) -> None:
        """
        Main entry point that starts all the tasks and manages the session.
        """
        try:
            async with (
                client.aio.live.connect(
                    model=os.getenv("MODEL"), config=CONFIG
                ) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                # Initialize queues
                self.audio_in_queue = asyncio.Queue()
                # Queue to store frames from camera or screen (limited size to prevent memory issues)
                self.out_queue = asyncio.Queue(maxsize=5)

                # Start tasks
                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())

                # Start appropriate video capture task based on mode
                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    tg.create_task(self.get_screen())

                tg.create_task(self.receive_audio())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            print("\nExiting gracefully...")
        except Exception as e:
            print(f"\nError: {e}")
            traceback.print_exc()
        finally:
            # Clean up resources
            if self.audio_stream:
                self.audio_stream.close()


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Multimodal chatbot with screen/camera capture"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="Source of visual input",
        choices=["camera", "screen", "none"],
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=DEFAULT_MONITOR,
        help="Monitor index to capture (0 is primary, 1 is secondary, etc.)",
    )
    parser.add_argument(
        "--audio-cache",
        type=int,
        default=30,
        help="Seconds of audio to cache before sending (default: 10)",
    )
    parser.add_argument(
        "--audio-device-index",
        type=int,
        default=0,
        help="Specific audio device index to use (overrides automatic selection)",
    )
    parser.add_argument(
        "--conversation-name",
        type=str,
        default=None,
        help="Name for the conversation subfolder (random if not specified)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List all audio devices and exit",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # If --list-devices is specified, just list devices and exit
    if True:
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
        info = pya.get_device_info_by_index(args.audio_device_index)
        print(
            f"Using audio device index: {args.audio_device_index}, device name: {info['name']}"
        )

    main = MultimodalChatbot(
        video_mode=args.mode,
        monitor_index=args.monitor,
        conversation_name=args.conversation_name,
        audio_device_index=args.audio_device_index,
    )
    main.audio_cache_seconds = args.audio_cache
    main.audio_cache = collections.deque(
        maxlen=int(SEND_SAMPLE_RATE * main.audio_cache_seconds / CHUNK_SIZE)
    )

    try:
        asyncio.run(main.run())
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
