"""Video processing utilities for NRHA reining analysis."""
import cv2
import numpy as np
from typing import Tuple, Optional

class VideoProcessor:
    """Handles video processing and standardization."""

    def __init__(self, target_resolution: Tuple[int, int], target_fps: int):
        """Initialize the video processor.

        Args:
            target_resolution: Target resolution (width, height)
            target_fps: Target frames per second
        """
        self.target_resolution = target_resolution
        self.target_fps = target_fps

    def load_video(self, video_path: str) -> Tuple[cv2.VideoCapture, dict]:
        """Load a video file.

        Args:
            video_path: Path to the video file

        Returns:
            Tuple of (video capture object, video properties)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        properties = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }

        return cap, properties

    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize a frame to the target resolution.

        Args:
            frame: Input frame

        Returns:
            Resized frame
        """
        return cv2.resize(frame, self.target_resolution)

    def process_video(self, video_path: str) -> Optional[np.ndarray]:
        """Process a video file to extract frames.

        Args:
            video_path: Path to the video file

        Returns:
            Array of processed frames or None if processing fails
        """
        try:
            cap, properties = self.load_video(video_path)
            frames = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize frame
                frame = self.resize_frame(frame)
                frames.append(frame)

            cap.release()
            return np.array(frames)

        except Exception as e:
            print(f"Error processing video: {e}")
            return None 