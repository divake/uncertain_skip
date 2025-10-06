"""
DQN-based Model Selector for Adaptive Tracking
Plug-in replacement for rule-based model selection
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from pathlib import Path
import json

class DQN(nn.Module):
    """Simple DQN for model selection"""
    def __init__(self, state_dim=10, action_dim=5, hidden_dim=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class RLModelSelector:
    """
    RL-based model selector that can be used as drop-in replacement
    for the rule-based select_adaptive_model_bidirectional method
    """
    
    def __init__(self, 
                 model_names=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                 pretrained_path=None,
                 training_mode=False):
        """
        Initialize RL model selector
        
        Args:
            model_names: List of model names in order
            pretrained_path: Path to pretrained weights (if None, uses random init)
            training_mode: Whether to collect experiences for training
        """
        self.model_names = model_names
        self.action_dim = len(model_names)
        self.state_dim = 10
        self.training_mode = training_mode
        
        # Model parameters for reward calculation
        self.model_params = {
            'yolov8n': 3.2,
            'yolov8s': 11.2,
            'yolov8m': 25.9,
            'yolov8l': 43.7,
            'yolov8x': 68.2
        }
        
        # Initialize network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(self.state_dim, self.action_dim).to(self.device)
        
        # Load pretrained weights if available
        if pretrained_path and Path(pretrained_path).exists():
            self.load_weights(pretrained_path)
            print(f"Loaded RL model from {pretrained_path}")
        else:
            print("Using random initialization for RL model")
            # Initialize with slight bias towards middle models
            with torch.no_grad():
                # Bias the initial Q-values
                bias = torch.tensor([0.0, 0.5, 1.0, 0.5, 0.0]).to(self.device)
                self.q_network.fc3.bias.data = bias
        
        # Training components (only if in training mode)
        if training_mode:
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-4)
            self.experiences = []
            self.epsilon = 0.1  # Low epsilon for mostly exploitation
        else:
            self.epsilon = 0.0  # Pure exploitation when not training
        
        # State tracking
        self.previous_state = None
        self.previous_action = None
        self.frames_since_switch = 0
        
    def extract_state(self, 
                     confidence: float,
                     confidence_history: list,
                     current_model: str,
                     iou: float = 0.5,
                     bbox: np.ndarray = None,
                     frame_shape: tuple = (1080, 1920)) -> np.ndarray:
        """
        Extract state features from tracking information
        Compatible with existing tracker
        """
        # Get confidence statistics
        if len(confidence_history) > 1:
            conf_mean = np.mean(confidence_history[-5:])
            conf_var = np.var(confidence_history[-5:])
            conf_trend = (confidence_history[-1] - confidence_history[-2]) if len(confidence_history) > 1 else 0
        else:
            conf_mean = confidence
            conf_var = 0.0
            conf_trend = 0.0
        
        # Current model index
        current_model_idx = self.model_names.index(current_model) if current_model in self.model_names else 2
        
        # Object size (if bbox provided)
        if bbox is not None:
            x, y, w, h = bbox
            object_size = (w * h) / (frame_shape[0] * frame_shape[1])
            center_x = x + w/2
            center_y = y + h/2
            edge_dist = min(center_x, center_y, 
                          frame_shape[1] - center_x, 
                          frame_shape[0] - center_y) / max(frame_shape)
        else:
            object_size = 0.1  # Default medium size
            edge_dist = 0.5    # Default center
        
        # Uncertainty metric (combination of variance and low confidence)
        uncertainty = conf_var + (1 - confidence) * 0.3
        
        # Build state vector
        state = np.array([
            confidence,                       # Current confidence
            conf_mean,                        # Mean confidence
            conf_var,                         # Confidence variance (uncertainty)
            conf_trend,                       # Confidence trend
            iou,                             # Tracking continuity
            self.frames_since_switch / 100.0, # Normalized time since switch
            current_model_idx / 4.0,          # Normalized current model
            object_size,                      # Object size
            edge_dist,                        # Distance to edges
            uncertainty                       # Combined uncertainty
        ], dtype=np.float32)
        
        return state
    
    def select_model(self, state: np.ndarray) -> str:
        """
        Select model using DQN
        
        Args:
            state: State vector
            
        Returns:
            Model name to use
        """
        # Epsilon-greedy for exploration (only in training mode)
        if self.training_mode and random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            # Use Q-network for selection
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()
        
        # Update tracking
        selected_model = self.model_names[action]
        if self.previous_action != action:
            self.frames_since_switch = 0
        else:
            self.frames_since_switch += 1
        
        self.previous_action = action
        self.previous_state = state
        
        return selected_model
    
    def select_adaptive_model_rl(self, 
                                 confidence: float,
                                 confidence_history: list,
                                 current_model: str,
                                 cooldown_frames: int = 0,
                                 **kwargs) -> str:
        """
        Main interface method - compatible with existing tracker
        
        This method signature matches what the tracker expects
        """
        # Skip if in cooldown
        if cooldown_frames > 0:
            return current_model
        
        # Need at least 3 frames of history
        if len(confidence_history) < 3:
            return current_model
        
        # Extract state
        state = self.extract_state(
            confidence=confidence,
            confidence_history=confidence_history,
            current_model=current_model,
            **kwargs
        )
        
        # Select model using DQN
        selected_model = self.select_model(state)
        
        # Store experience if training
        if self.training_mode and self.previous_state is not None:
            # Calculate reward (simplified for now)
            reward = self.calculate_reward(confidence, selected_model)
            self.experiences.append({
                'state': self.previous_state,
                'action': self.previous_action,
                'reward': reward,
                'next_state': state,
                'done': False
            })
        
        return selected_model
    
    def calculate_reward(self, confidence: float, model_name: str) -> float:
        """
        Simple reward function
        """
        # Tracking quality
        tracking_reward = confidence
        
        # Computational cost
        computation_cost = self.model_params[model_name] / self.model_params['yolov8x']
        
        # Combine
        reward = tracking_reward - 0.3 * computation_cost
        
        # Bonus for stability
        if self.frames_since_switch > 5:
            reward += 0.1
            
        return reward
    
    def save_weights(self, path: str):
        """Save model weights"""
        torch.save({
            'model_state': self.q_network.state_dict(),
            'model_names': self.model_names
        }, path)
        print(f"Saved RL model to {path}")
    
    def load_weights(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['model_state'])
        if 'model_names' in checkpoint:
            self.model_names = checkpoint['model_names']
    
    def save_experiences(self, path: str):
        """Save collected experiences for offline training"""
        if self.training_mode and self.experiences:
            # Convert numpy arrays to lists for JSON serialization
            json_experiences = []
            for exp in self.experiences:
                json_exp = {
                    'state': exp['state'].tolist() if isinstance(exp['state'], np.ndarray) else exp['state'],
                    'action': int(exp['action']) if isinstance(exp['action'], np.integer) else exp['action'],
                    'reward': float(exp['reward']),
                    'next_state': exp['next_state'].tolist() if isinstance(exp['next_state'], np.ndarray) else exp['next_state'],
                    'done': exp['done']
                }
                json_experiences.append(json_exp)
            
            with open(path, 'w') as f:
                json.dump(json_experiences, f, indent=2)
            print(f"Saved {len(json_experiences)} experiences to {path}")