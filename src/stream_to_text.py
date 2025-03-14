#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import datetime
import io
import os
import time
from typing import Tuple

import cv2
import mss
import PIL.Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# Default settings
DEFAULT_MODE = "screen"
DEFAULT_MONITOR = 1  # Default to the primary monitor
DEFAULT_QUERY = os.getenv("DEFAULT_QUERY", ".")
DEFAULT_MODEL = os.getenv("MODEL", "gemini-2.0-flash-exp")
DEFAULT_STREAMING = True  # Default to streaming mode


class ScreenshotChatbot:
    """
    A chatbot that captures screenshots and sends them with text to the Gemini API.

    This class provides a simpler interface compared to the AudioLoop class,
    focusing only on screenshot capture and text interaction.
    """

    def __init__(
        self,
        video_mode: str = DEFAULT_MODE,
        monitor_index: int = DEFAULT_MONITOR,
        default_query: str = DEFAULT_QUERY,
        model: str = DEFAULT_MODEL,
        streaming: bool = DEFAULT_STREAMING,
    ) -> None:
        """
        Initialize the ScreenshotChatbot with specified settings.

        Args:
            video_mode: Mode for video capture ("screen" or "camera")
            monitor_index: Index of the monitor to capture (0-based)
            default_query: Default query to use when no text is provided
            model: Gemini model to use for generating responses
            streaming: Whether to use streaming mode for responses
        """
        self.video_mode = video_mode
        self.monitor_index = monitor_index
        self.default_query = default_query
        self.model = model
        self.streaming = streaming

        # Create screenshots directory if it doesn't exist
        self.screenshots_dir = "screenshots"
        os.makedirs(self.screenshots_dir, exist_ok=True)

        # Initialize Gemini client
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        # Initialize camera if needed
        self.camera = None
        if self.video_mode == "camera":
            self.camera = cv2.VideoCapture(0)

    def __del__(self) -> None:
        """Clean up resources when the object is destroyed."""
        if self.camera is not None:
            self.camera.release()

    def capture_screenshot(self) -> Tuple[str, bytes]:
        """
        Capture a screenshot or camera frame and save it.

        Returns:
            Tuple containing the filename and the image bytes
        """
        # Get current timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.screenshots_dir}/screenshot_{timestamp}.jpg"

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

                # Save the image
                img.save(filename)

                # Get image bytes for API
                image_io = io.BytesIO()
                img.save(image_io, format="jpeg")
                image_io.seek(0)
                image_bytes = image_io.read()

            else:  # screen mode
                # Capture from screen
                with mss.mss() as sct:
                    monitor = sct.monitors[self.monitor_index]
                    sct_img = sct.grab(monitor)

                    # Convert to PIL Image
                    img = PIL.Image.frombytes("RGB", sct_img.size, sct_img.rgb)

                    # Resize if needed (Gemini has image size limits)
                    img.thumbnail([1024, 1024])

                    # Save the image
                    img.save(filename)

                    # Get image bytes for API
                    image_io = io.BytesIO()
                    img.save(image_io, format="jpeg")
                    image_io.seek(0)
                    image_bytes = image_io.read()

            print(f"\nScreenshot saved: {filename}")
            return filename, image_bytes

        except Exception as e:
            print(f"\nError capturing screenshot: {e}")
            return "", b""

    def send_to_gemini(self, prompt: str, image_bytes: bytes) -> str:
        """
        Send the prompt and image to the Gemini API.

        Args:
            prompt: Text prompt to send
            image_bytes: Image bytes to send

        Returns:
            Response text from Gemini
        """
        try:
            # Prepare the contents
            contents = [
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            ]

            if self.streaming:
                # Use streaming mode
                print("\nGemini: ", end="", flush=True)
                response_text = ""

                # Stream the response
                response_stream = self.client.models.generate_content_stream(
                    model=self.model, contents=contents
                )

                for chunk in response_stream:
                    if chunk.text:
                        print(chunk.text, end="", flush=True)
                        response_text += chunk.text

                # Add a newline after streaming is complete
                print()
                return response_text
            else:
                # Use non-streaming mode
                response = self.client.models.generate_content(
                    model=self.model, contents=contents
                )

                # Return the response text
                return response.text
        except Exception as e:
            print(f"\nError sending to Gemini API: {e}")
            return f"Error: {str(e)}"

    def run(self) -> None:
        """
        Main loop that handles user input and sends requests to the Gemini API.
        """
        print("\nWelcome to Screenshot Chatbot!")
        print(
            "Type your question and press Enter to capture a screenshot and get a response."
        )
        print(f"Streaming mode is {'enabled' if self.streaming else 'disabled'}.")
        print("Type 'q' to quit.")
        print("Type 'toggle' to toggle streaming mode.")

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

            # Use default prompt if user didn't type anything
            if not text:
                text = self.default_query
                print(f"Using default query: {text}")

            # Capture screenshot
            filename, image_bytes = self.capture_screenshot()
            if not filename or not image_bytes:
                print("Failed to capture screenshot. Please try again.")
                continue

            # Send to Gemini API
            print("Sending to Gemini API...")

            # Start timing the response
            start_time = time.time()

            # Get response from Gemini
            response = self.send_to_gemini(text, image_bytes)

            # Calculate and display response time
            elapsed_time = time.time() - start_time
            print(f"\nResponse time: {elapsed_time:.2f} seconds")

            # If not streaming, print the response
            if not self.streaming:
                print("\nGemini: " + response)

        print("\nExiting...")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Screenshot-to-Text Chatbot with Gemini API"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="Source of visual input",
        choices=["camera", "screen"],
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=DEFAULT_MONITOR,
        help="Monitor index to capture (0 is primary, 1 is secondary, etc.)",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, help="Gemini model to use"
    )
    parser.add_argument(
        "--no-streaming", action="store_true", help="Disable streaming mode"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    chatbot = ScreenshotChatbot(
        video_mode=args.mode,
        monitor_index=args.monitor,
        model=args.model,
        streaming=not args.no_streaming,
    )

    try:
        chatbot.run()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")


if __name__ == "__main__":
    main()
