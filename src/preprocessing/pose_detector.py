"""Pose detection utilities for horse and rider tracking."""
import mediapipe as mp
import numpy as np
from typing import Dict, List, Optional, Tuple
import cv2

class PoseDetector:
    """Handles pose detection for horses and riders."""

    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """Initialize the pose detector.

        Args:
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detect_pose(self, frame: np.ndarray) -> Optional[Dict[str, List[Tuple[float, float, float]]]]:
        """Detect poses in a frame.

        Args:
            frame: Input frame

        Returns:
            Dictionary of pose landmarks or None if detection fails
        """
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.pose.process(frame_rgb)
            
            if not results.pose_landmarks:
                return None

            # Extract landmarks
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append((
                    landmark.x,
                    landmark.y,
                    landmark.visibility
                ))

            return {
                "landmarks": landmarks,
                "frame_shape": frame.shape
            }

        except Exception as e:
            print(f"Error detecting pose: {e}")
            return None

    def process_video(self, frames: np.ndarray) -> List[Optional[Dict]]:
        """Process a sequence of frames for pose detection.

        Args:
            frames: Array of video frames

        Returns:
            List of pose detection results for each frame
        """
        results = []
        for frame in frames:
            result = self.detect_pose(frame)
            results.append(result)
        return results 