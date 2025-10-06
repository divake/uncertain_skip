"""
Reinforcement Learning based Adaptive Model Selection for Object Tracking
Uses Deep Q-Network (DQN) to learn optimal model switching policy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
from typing import List, Tuple, Optional
import yaml
from dataclasses import dataclass

# Experience replay buffer
Experience = namedtuple('Experience', 
    ['state', 'action', 'reward', 'next_state', 'done'])

@dataclass
class RLConfig:
    """Configuration for RL training"""
    # Model parameters
    state_dim: int = 10
    action_dim: int = 5  # 5 YOLOv8 models
    hidden_dim: int = 256
    
    # Training parameters
    learning_rate: float = 1e-4
    gamma: float = 0.99  # Discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Reward weights
    tracking_weight: float = 1.0
    computation_weight: float = 0.3
    stability_weight: float = 0.1
    
    # Buffer parameters
    buffer_size: int = 10000
    batch_size: int = 64
    update_frequency: int = 4
    target_update_frequency: int = 100

class DQNetwork(nn.Module):
    """Deep Q-Network for model selection"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(DQNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        q_values = self.fc4(x)
        return q_values

class ReplayBuffer:
    """Experience replay buffer for stable training"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class RLAdaptiveTracker:
    """
    RL-based adaptive model selector for object tracking
    Learns to select optimal YOLOv8 model based on tracking state
    """
    
    def __init__(self, config: RLConfig = None):
        self.config = config or RLConfig()
        
        # Model parameters (in millions)
        self.model_params = {
            'yolov8n': 3.2,
            'yolov8s': 11.2,
            'yolov8m': 25.9,
            'yolov8l': 43.7,
            'yolov8x': 68.2
        }
        self.model_names = list(self.model_params.keys())
        
        # Initialize networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNetwork(
            self.config.state_dim, 
            self.config.action_dim,
            self.config.hidden_dim
        ).to(self.device)
        
        self.target_network = DQNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim
        ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Training components
        self.optimizer = optim.Adam(self.q_network.parameters(), 
                                   lr=self.config.learning_rate)
        self.replay_buffer = ReplayBuffer(self.config.buffer_size)
        
        # Exploration parameters
        self.epsilon = self.config.epsilon_start
        self.steps = 0
        
        # Tracking state
        self.previous_action = None
        self.frames_since_switch = 0
        
    def extract_state(self, 
                     confidence: float,
                     confidence_history: List[float],
                     iou: float,
                     bbox: np.ndarray,
                     frame_shape: Tuple[int, int],
                     current_model_idx: int) -> np.ndarray:
        """
        Extract state features from tracking information
        
        Returns:
            state vector of shape (state_dim,)
        """
        # Basic features
        conf_variance = np.var(confidence_history[-5:]) if len(confidence_history) > 1 else 0
        conf_trend = np.polyfit(range(len(confidence_history[-5:])), 
                                confidence_history[-5:], 1)[0] if len(confidence_history) > 1 else 0
        
        # Object size ratio
        x, y, w, h = bbox
        object_size = (w * h) / (frame_shape[0] * frame_shape[1])
        
        # Distance to edges (normalized)
        center_x = x + w/2
        center_y = y + h/2
        edge_dist = min(center_x, center_y, 
                       frame_shape[1] - center_x, 
                       frame_shape[0] - center_y) / max(frame_shape)
        
        # Motion (simplified - would need previous bbox in real implementation)
        motion = 0.0  # Placeholder
        
        # Occlusion likelihood (simplified)
        occlusion_likelihood = 1.0 - confidence if confidence < 0.5 else 0.0
        
        state = np.array([
            confidence,
            conf_variance,
            conf_trend,
            iou,
            self.frames_since_switch / 100.0,  # Normalize
            current_model_idx / 4.0,  # Normalize to [0,1]
            object_size,
            motion,
            edge_dist,
            occlusion_likelihood
        ], dtype=np.float32)
        
        return state
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state vector
            training: Whether in training mode
            
        Returns:
            Action index (0-4 for different models)
        """
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return random.randint(0, self.config.action_dim - 1)
        
        # Exploitation: choose best action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def calculate_reward(self,
                        confidence: float,
                        iou: float,
                        model_idx: int,
                        tracked: bool,
                        switched: bool) -> float:
        """
        Calculate reward based on tracking performance and efficiency
        
        Args:
            confidence: Detection confidence
            iou: IoU with previous frame
            model_idx: Selected model index
            tracked: Whether object was successfully tracked
            switched: Whether model was switched
            
        Returns:
            Reward value
        """
        # Tracking quality reward
        if tracked:
            tracking_reward = confidence * iou
        else:
            tracking_reward = -5.0  # Heavy penalty for losing track
        
        # Computational cost penalty
        model_name = self.model_names[model_idx]
        computation_penalty = self.model_params[model_name] / self.model_params['yolov8x']
        
        # Stability bonus (avoid unnecessary switching)
        stability_bonus = 0.0
        if not switched:
            stability_bonus = 0.1
        elif self.frames_since_switch < 5:
            stability_bonus = -0.2  # Penalty for rapid switching
        
        # Combine rewards
        reward = (self.config.tracking_weight * tracking_reward - 
                 self.config.computation_weight * computation_penalty + 
                 self.config.stability_weight * stability_bonus)
        
        return reward
    
    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < self.config.batch_size:
            return
        
        # Sample batch
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        # Prepare batch tensors
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.FloatTensor([e.done for e in batch]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.config.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        if self.steps % self.config.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.config.epsilon_end, 
                          self.epsilon * self.config.epsilon_decay)
        
        self.steps += 1
        
        return loss.item()
    
    def update_tracking_state(self, action: int):
        """Update internal tracking state"""
        if self.previous_action != action:
            self.frames_since_switch = 0
        else:
            self.frames_since_switch += 1
        self.previous_action = action
    
    def save_model(self, path: str):
        """Save trained model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
    
    def load_model(self, path: str):
        """Load trained model"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']