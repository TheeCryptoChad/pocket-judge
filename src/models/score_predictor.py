"""Score prediction model for NRHA reining analysis."""
import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple

class ScorePredictor(nn.Module):
    """Neural network for predicting maneuver scores."""

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2):
        """Initialize the score predictor.

        Args:
            input_size: Size of input features (pose landmarks)
            hidden_size: Size of hidden layers
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
            nn.Linear(hidden_size // 2, 1)  # Single output for score
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use the last output of the sequence
        last_output = lstm_out[:, -1, :]
        
        # Pass through fully connected layers
        return self.fc(last_output)

    def predict_score(self,
                     pose_sequence: List[Dict[str, Any]],
                     maneuver_type: str) -> Dict[str, Any]:
        """Predict the score for a maneuver.

        Args:
            pose_sequence: List of pose detection results
            maneuver_type: Type of maneuver being scored

        Returns:
            Dictionary containing predicted score and confidence
        """
        self.eval()
        with torch.no_grad():
            # Convert pose sequence to tensor
            x = torch.tensor([p["landmarks"] for p in pose_sequence if p is not None])
            
            # Add batch dimension if needed
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
            
            # Get prediction
            score = self(x)
            
            # Clamp score to valid range (-1.5 to 1.5)
            score = torch.clamp(score, -1.5, 1.5)
            
            return {
                "score": score.item(),
                "maneuver_type": maneuver_type
            }

    def predict_penalties(self,
                         pose_sequence: List[Dict[str, Any]],
                         maneuver_type: str) -> List[Dict[str, Any]]:
        """Predict penalties for a maneuver.

        Args:
            pose_sequence: List of pose detection results
            maneuver_type: Type of maneuver being analyzed

        Returns:
            List of detected penalties
        """
        # This is a placeholder - actual implementation will depend on
        # specific penalty detection rules for each maneuver type
        return []

    def calculate_total_score(self,
                            maneuver_scores: List[Dict[str, Any]],
                            penalties: List[Dict[str, Any]]) -> float:
        """Calculate the total score for a run.

        Args:
            maneuver_scores: List of maneuver scores
            penalties: List of penalties

        Returns:
            Total score (0-100)
        """
        # Sum all maneuver scores
        total = sum(score["score"] for score in maneuver_scores)
        
        # Apply penalties
        for penalty in penalties:
            total += penalty["value"]
        
        # Convert to 0-100 scale
        # Assuming 7 maneuvers with max score of 1.5 each = 10.5 total
        # Scale to 0-100 range
        scaled_score = (total + 10.5) * (100 / 21)
        
        # Clamp to valid range
        return max(0, min(100, scaled_score)) 