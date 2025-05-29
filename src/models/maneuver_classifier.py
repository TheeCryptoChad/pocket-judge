"""Maneuver classification model for NRHA reining analysis."""
import torch
import torch.nn as nn
from typing import List, Dict, Any

class ManeuverClassifier(nn.Module):
    """Neural network for classifying reining maneuvers."""

    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 128,
                 num_classes: int = 8,
                 num_layers: int = 2):
        """Initialize the maneuver classifier.

        Args:
            input_size: Size of input features (pose landmarks)
            hidden_size: Size of hidden layers
            num_classes: Number of maneuver classes
            num_layers: Number of LSTM layers
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use the last output of the sequence
        last_output = lstm_out[:, -1, :]
        
        # Pass through fully connected layers
        return self.fc(last_output)

    def predict_maneuver(self, 
                        pose_sequence: List[Dict[str, Any]],
                        confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """Predict the maneuver type from a sequence of pose detections.

        Args:
            pose_sequence: List of pose detection results
            confidence_threshold: Minimum confidence for prediction

        Returns:
            Dictionary containing predicted maneuver and confidence
        """
        self.eval()
        with torch.no_grad():
            # Convert pose sequence to tensor
            # This is a placeholder - actual implementation will depend on
            # how we process the pose landmarks
            x = torch.tensor([p["landmarks"] for p in pose_sequence if p is not None])
            
            # Add batch dimension if needed
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
            
            # Get predictions
            outputs = self(x)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get predicted class and confidence
            confidence, predicted = torch.max(probabilities, dim=1)
            
            if confidence.item() < confidence_threshold:
                return {
                    "maneuver": "unknown",
                    "confidence": confidence.item()
                }
            
            return {
                "maneuver": self._get_maneuver_name(predicted.item()),
                "confidence": confidence.item()
            }

    def _get_maneuver_name(self, class_idx: int) -> str:
        """Convert class index to maneuver name.

        Args:
            class_idx: Class index

        Returns:
            Maneuver name
        """
        maneuvers = [
            "circles",
            "spins",
            "sliding_stops",
            "rollbacks",
            "lead_changes",
            "backups",
            "run_downs",
            "pauses"
        ]
        return maneuvers[class_idx] 