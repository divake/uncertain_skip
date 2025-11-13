"""
Train DQN Policy for Adaptive Model Selection with Real Uncertainty

This script trains a DQN to learn optimal model selection using:
- 12D state space with real aleatoric (Mahalanobis) and epistemic (Triple-S)
- Orthogonality-aware reward function (r=0.0063)
- Experience replay from MOT17-04 tracking
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from collections import deque
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.core.rl_model_selector import DQN, RLModelSelector
from src.utils.mot_utils import load_mot_gt


class ExperienceReplay:
    """Experience replay buffer for DQN training"""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample random batch"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class DQNTrainer:
    """DQN trainer for adaptive model selection"""

    def __init__(self,
                 state_dim=12,
                 action_dim=5,
                 hidden_dim=128,
                 lr=1e-4,
                 gamma=0.99,
                 device='cuda'):
        """
        Initialize DQN trainer

        Args:
            state_dim: State space dimension (12 with uncertainty)
            action_dim: Number of models (5: n, s, m, l, x)
            hidden_dim: Hidden layer size
            lr: Learning rate
            gamma: Discount factor
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Q-networks
        self.q_network = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Hyperparameters
        self.gamma = gamma
        self.action_dim = action_dim

        # Training stats
        self.losses = []
        self.episode_rewards = []

    def train_step(self, batch, batch_size=64):
        """Single training step"""
        states, actions, rewards, next_states, dones = batch

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (Double DQN)
        with torch.no_grad():
            # Use online network to select actions
            next_actions = self.q_network(next_states).argmax(1)
            # Use target network to evaluate actions
            next_q_values = self.target_network(next_states)
            next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # TD target
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = self.loss_fn(q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_checkpoint(self, path, epoch, replay_buffer):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'losses': self.losses,
            'episode_rewards': self.episode_rewards
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.losses = checkpoint['losses']
        self.episode_rewards = checkpoint['episode_rewards']
        return checkpoint['epoch']


def load_uncertainty_data(uncertainty_json_path):
    """Load precomputed uncertainty from JSON"""
    with open(uncertainty_json_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded uncertainty data:")
    print(f"  Sequence: {data['sequence']}")
    print(f"  Total detections: {data['n_detections']}")
    print(f"  Number of frames: {data['n_frames']}")
    print(f"  Aleatoric: {data['statistics']['aleatoric']['mean']:.3f} ± {data['statistics']['aleatoric']['std']:.3f}")
    print(f"  Epistemic: {data['statistics']['epistemic']['mean']:.3f} ± {data['statistics']['epistemic']['std']:.3f}")
    print(f"  Orthogonality: r = {data['statistics']['orthogonality']:.4f}")

    return data


def generate_training_experiences(uncertainty_data,
                                  sequence_path,
                                  ground_truth_path,
                                  track_id=1,
                                  max_frames=1050):
    """
    Generate training experiences by simulating tracking with different models

    This creates a dataset of (state, action, reward, next_state) tuples
    by trying different model selections at each frame.
    """

    print("\n" + "="*80)
    print("GENERATING TRAINING EXPERIENCES")
    print("="*80)

    # Load ground truth
    ground_truth = load_mot_gt(ground_truth_path)
    ground_truth = np.array(ground_truth)  # Convert to numpy array
    gt_track = ground_truth[ground_truth[:, 1] == track_id]  # [frame, id, x, y, w, h]

    print(f"\nGround truth track {track_id}: {len(gt_track)} frames")

    experiences = []
    frames_processed = 0

    # Model names
    model_names = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
    model_params = {
        'yolov8n': 3.2,
        'yolov8s': 11.2,
        'yolov8m': 25.9,
        'yolov8l': 43.7,
        'yolov8x': 68.2
    }

    # Track state histories
    confidence_history = []
    aleatoric_history = []
    epistemic_history = []
    prev_bbox = None
    frames_since_switch = 0
    current_model_idx = 2  # Start with medium model

    # Process each frame
    for frame_idx in tqdm(range(max_frames), desc="Generating experiences"):
        frame_num = frame_idx + 1

        # Get ground truth for this frame
        gt_frame = gt_track[gt_track[:, 0] == frame_num]
        if len(gt_frame) == 0:
            continue

        gt_bbox = gt_frame[0, 2:6]  # [x, y, w, h]

        # Get uncertainty for this frame
        frame_key = str(frame_num)
        if frame_key not in uncertainty_data['frames']:
            continue

        frame_detections = uncertainty_data['frames'][frame_key]

        # Find best matching detection (highest IoU with GT)
        best_iou = 0.0
        best_detection = None

        for det in frame_detections:
            det_bbox = det['bbox']
            iou = calculate_iou(det_bbox, gt_bbox)
            if iou > best_iou:
                best_iou = iou
                best_detection = det

        if best_detection is None or best_iou < 0.3:
            # Lost track - reset
            confidence_history = []
            aleatoric_history = []
            epistemic_history = []
            prev_bbox = None
            continue

        # Extract state features
        confidence = best_detection['confidence']
        aleatoric = best_detection['aleatoric']
        epistemic = best_detection['epistemic']

        confidence_history.append(confidence)
        aleatoric_history.append(aleatoric)
        epistemic_history.append(epistemic)

        if len(confidence_history) < 3:
            prev_bbox = gt_bbox
            continue

        # Build state
        conf_mean = np.mean(confidence_history[-5:])
        conf_trend = confidence_history[-1] - confidence_history[-2] if len(confidence_history) > 1 else 0.0
        aleatoric_mean = np.mean(aleatoric_history[-5:])
        epistemic_mean = np.mean(epistemic_history[-5:])

        object_size = (gt_bbox[2] * gt_bbox[3]) / (1080 * 1920)
        center_x = gt_bbox[0] + gt_bbox[2] / 2
        center_y = gt_bbox[1] + gt_bbox[3] / 2
        edge_dist = min(center_x, center_y, 1920 - center_x, 1080 - center_y) / 1920

        state = np.array([
            confidence,
            conf_mean,
            conf_trend,
            best_iou,
            frames_since_switch / 100.0,
            current_model_idx / 4.0,
            object_size,
            edge_dist,
            aleatoric,
            aleatoric_mean,
            epistemic,
            epistemic_mean
        ], dtype=np.float32)

        # Try each action (model selection) and compute reward
        for action_idx, model_name in enumerate(model_names):
            # Compute reward for this action using NEW reward function
            model_cost = model_params[model_name] / model_params['yolov8x']

            # Base reward: quality is primary
            reward = 2.0 * best_iou

            # Cost penalty - reduced
            reward -= 0.1 * model_cost

            # CRITICAL: Epistemic-based model size matching
            if epistemic > 0.5:  # High epistemic
                required_model_idx = min(4, int(epistemic * 6))
                if action_idx < required_model_idx:
                    reward -= 1.5 * (required_model_idx - action_idx) / 4.0

            if epistemic > 0.6:  # Very high epistemic
                if action_idx >= 3:  # Using L or X
                    reward += 0.8 * epistemic

            if epistemic < 0.3:  # Low epistemic
                if action_idx > 2:  # Using more than M
                    reward -= 0.6 * (1.0 - epistemic)

            # Aleatoric penalty
            if aleatoric > 0.3:
                reward -= 0.3 * model_cost * aleatoric

            # Stability bonuses (reduced)
            if action_idx == current_model_idx and frames_since_switch > 3:
                reward += 0.05
            if action_idx != current_model_idx and frames_since_switch < 2:
                reward -= 0.05

            # Store experience (we'll compute next_state on next iteration)
            if prev_bbox is not None:
                experiences.append({
                    'state': state,
                    'action': action_idx,
                    'reward': reward,
                    'next_state': None,  # Will be filled in next iteration
                    'done': False
                })

        # Update tracking state
        if random.random() < 0.1:  # Occasionally switch models for exploration
            current_model_idx = random.randint(0, 4)
            frames_since_switch = 0
        else:
            frames_since_switch += 1

        prev_bbox = gt_bbox
        frames_processed += 1

    # Fill in next_states
    print(f"\nFilling in next_states for {len(experiences)} experiences...")
    for i in range(len(experiences) - 1):
        experiences[i]['next_state'] = experiences[i + 1]['state']
    # Mark last experience as done
    if experiences:
        experiences[-1]['done'] = True
        experiences[-1]['next_state'] = experiences[-1]['state']

    print(f"\n✓ Generated {len(experiences)} training experiences from {frames_processed} frames")

    return experiences


def calculate_iou(bbox1, bbox2):
    """Calculate IoU between two bboxes [x, y, w, h]"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Convert to [x1, y1, x2, y2]
    box1 = [x1, y1, x1 + w1, y1 + h1]
    box2 = [x2, y2, x2 + w2, y2 + h2]

    # Intersection
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def train(uncertainty_json_path,
          sequence_path,
          ground_truth_path,
          output_dir='results/rl_training',
          num_epochs=100,
          batch_size=64,
          target_update_freq=10,
          save_freq=10):
    """
    Main training loop

    Args:
        uncertainty_json_path: Path to precomputed uncertainty JSON
        sequence_path: Path to MOT17 sequence
        ground_truth_path: Path to ground truth file
        output_dir: Directory to save results
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        target_update_freq: How often to update target network
        save_freq: How often to save checkpoints
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load uncertainty data
    print("\n" + "="*80)
    print("LOADING UNCERTAINTY DATA")
    print("="*80)
    uncertainty_data = load_uncertainty_data(uncertainty_json_path)

    # Generate training experiences
    experiences = generate_training_experiences(
        uncertainty_data,
        sequence_path,
        ground_truth_path,
        track_id=1,
        max_frames=1050
    )

    # Create replay buffer
    replay_buffer = ExperienceReplay(capacity=20000)
    for exp in experiences:
        replay_buffer.push(
            exp['state'],
            exp['action'],
            exp['reward'],
            exp['next_state'],
            exp['done']
        )

    print(f"\nReplay buffer size: {len(replay_buffer)} experiences")

    # Initialize trainer
    trainer = DQNTrainer(
        state_dim=12,
        action_dim=5,
        hidden_dim=128,
        lr=1e-4,
        gamma=0.99,
        device='cuda'
    )

    # Training loop
    print("\n" + "="*80)
    print("TRAINING DQN")
    print("="*80)

    for epoch in range(num_epochs):
        epoch_losses = []

        # Train on multiple batches per epoch
        num_batches = len(replay_buffer) // batch_size
        for _ in range(num_batches):
            if len(replay_buffer) < batch_size:
                break

            batch = replay_buffer.sample(batch_size)
            loss = trainer.train_step(batch, batch_size)
            epoch_losses.append(loss)

        avg_loss = np.mean(epoch_losses)
        trainer.losses.append(avg_loss)

        # Update target network
        if (epoch + 1) % target_update_freq == 0:
            trainer.update_target_network()

        # Save checkpoint
        if (epoch + 1) % save_freq == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            trainer.save_checkpoint(checkpoint_path, epoch, replay_buffer)

        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

    # Save final model
    final_path = output_dir / "final_model.pt"
    trainer.save_checkpoint(final_path, num_epochs, replay_buffer)

    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(trainer.losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DQN Training Loss')
    plt.grid(True)
    plt.savefig(output_dir / 'training_loss.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved training loss plot to {output_dir / 'training_loss.png'}")

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Final model saved to: {final_path}")
    print(f"Training loss: {trainer.losses[-1]:.4f}")


if __name__ == "__main__":
    # Paths
    uncertainty_json = "data/mot17_04_uncertainty.json"
    sequence_path = "data/MOT17/train/MOT17-04-FRCNN"
    ground_truth_path = "data/MOT17/train/MOT17-04-FRCNN/gt/gt.txt"

    # Train
    train(
        uncertainty_json_path=uncertainty_json,
        sequence_path=sequence_path,
        ground_truth_path=ground_truth_path,
        output_dir='results/rl_training',
        num_epochs=100,
        batch_size=64,
        target_update_freq=10,
        save_freq=10
    )
