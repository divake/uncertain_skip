"""
Comprehensive RL Training for Adaptive Model Selection
Implements the full orthogonality-aware reward function from RL_POLICY_PAPER_CONTENT.md

Train on MOT17-04 with Track ID 1
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from pathlib import Path
import pickle

# ==================== CONFIGURATION ====================
UNCERTAINTY_FILE = "/ssd_4TB/divake/uncertain_skip/data/mot17_04_uncertainty.json"
GT_FILE = "/ssd_4TB/divake/uncertain_skip/data/MOT17/train/MOT17-04-FRCNN/gt/gt.txt"
OUTPUT_DIR = Path("/ssd_4TB/divake/uncertain_skip/results/rl_training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model definitions
MODEL_NAMES = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
MODEL_PARAMS = {
    'yolov8n': 3.2,
    'yolov8s': 11.2,
    'yolov8m': 25.9,
    'yolov8l': 43.7,
    'yolov8x': 68.2
}

# RL Hyperparameters
STATE_DIM = 12
ACTION_DIM = 5
HIDDEN_DIM = 256
BUFFER_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-4
EPSILON_START = 0.3
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 100
NUM_EPISODES = 200


# ==================== DQN ARCHITECTURE ====================
class DQN(nn.Module):
    """Deep Q-Network with 2 hidden layers"""
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ==================== EXPERIENCE REPLAY ====================
class ReplayBuffer:
    """Experience replay buffer for off-policy learning"""
    def __init__(self, capacity=BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


# ==================== ORTHOGONALITY-AWARE REWARD FUNCTION ====================
def compute_orthogonal_reward(state, action, next_state, iou, debug=False):
    """
    Implements the full reward function from RL_POLICY_PAPER_CONTENT.md

    Args:
        state: Current state [12-dim]
        action: Action index (0-4)
        next_state: Next state [12-dim]
        iou: Ground-truth IoU
        debug: If True, print reward components

    Returns:
        Total reward scalar
    """
    # Extract state components
    conf = state[0]
    conf_mean = state[1]
    conf_trend = state[2]
    iou_state = state[3]
    frames_since_switch = state[4]
    current_model = int(state[5])
    bbox_size = state[6]
    edge_dist = state[7]
    aleatoric = state[8]
    aleatoric_mean = state[9]
    epistemic = state[10]
    epistemic_mean = state[11]

    # ===== Component 1: Tracking Quality (Primary) =====
    R_track = 2.0 * iou

    # ===== Component 2: Computational Cost =====
    cost_normalized = MODEL_PARAMS[MODEL_NAMES[action]] / MODEL_PARAMS['yolov8x']
    R_cost = -0.1 * cost_normalized

    # ===== Component 3: Epistemic-Based Model Matching (CRITICAL) =====
    R_epistemic = 0.0

    # Map epistemic to required model capacity
    m_required = min(4, int(6 * epistemic))

    # High epistemic → need large model
    if epistemic > 0.5 and action < m_required:
        penalty = -1.5 * (m_required - action) / 4.0
        R_epistemic += penalty

    # Very high epistemic + using large model → bonus
    if epistemic > 0.6 and action >= 3:
        bonus = 0.8 * epistemic
        R_epistemic += bonus

    # Low epistemic + using large model → waste penalty
    if epistemic < 0.3 and action > 2:
        penalty = -0.6 * (1 - epistemic)
        R_epistemic += penalty

    # ===== Component 4: Aleatoric-Based Data Quality Adjustment =====
    R_aleatoric = 0.0
    if aleatoric > 0.3:
        # High aleatoric → don't waste expensive models on noisy data
        R_aleatoric = -0.3 * cost_normalized * aleatoric

    # ===== Component 5: Temporal Stability =====
    R_stability = 0.0
    if frames_since_switch > 3:
        R_stability = 0.05  # Bonus for stability
    elif frames_since_switch < 2:
        R_stability = -0.05  # Penalty for rapid switching

    # ===== Total Reward =====
    total_reward = R_track + R_cost + R_epistemic + R_aleatoric + R_stability

    if debug:
        print(f"\n=== Reward Breakdown ===")
        print(f"  IoU: {iou:.3f}")
        print(f"  Epistemic: {epistemic:.3f}, Aleatoric: {aleatoric:.3f}")
        print(f"  Action (model): {action} ({MODEL_NAMES[action]})")
        print(f"  R_track:     {R_track:+.4f}")
        print(f"  R_cost:      {R_cost:+.4f}")
        print(f"  R_epistemic: {R_epistemic:+.4f}")
        print(f"  R_aleatoric: {R_aleatoric:+.4f}")
        print(f"  R_stability: {R_stability:+.4f}")
        print(f"  TOTAL:       {total_reward:+.4f}")

    return total_reward


# ==================== TRACKING ENVIRONMENT ====================
class TrackingEnvironment:
    """Simulates tracking environment for MOT17-04 Track 1"""

    def __init__(self, uncertainty_file, gt_file, track_id=1):
        """Load uncertainty data and ground truth"""
        print(f"\n[ENV] Loading data for Track {track_id}...")

        # Load uncertainty data
        with open(uncertainty_file, 'r') as f:
            self.uncertainty_data = json.load(f)

        # Load ground truth
        gt_data = np.loadtxt(gt_file, delimiter=',')
        self.gt_tracks = {}
        for row in gt_data:
            frame_id, track_id_gt, x, y, w, h, conf, cls, vis = row
            frame_id = int(frame_id)
            track_id_gt = int(track_id_gt)
            if track_id_gt not in self.gt_tracks:
                self.gt_tracks[track_id_gt] = {}
            self.gt_tracks[track_id_gt][frame_id] = {
                'bbox': [x, y, w, h],
                'conf': conf,
                'vis': vis
            }

        self.track_id = track_id
        self.frames = sorted([int(f) for f in self.uncertainty_data['frames'].keys()])
        self.current_frame_idx = 0

        # State tracking
        self.conf_history = []
        self.aleatoric_history = []
        self.epistemic_history = []
        self.frames_since_switch = 0
        self.current_model = 2  # Start with medium model
        self.previous_bbox = None

        print(f"[ENV] Loaded {len(self.frames)} frames")
        print(f"[ENV] Track {track_id} has {len(self.gt_tracks[track_id])} detections")
        print(f"[ENV] Orthogonality r = {self.uncertainty_data['statistics']['orthogonality']:.4f}")

    def reset(self):
        """Reset environment to start"""
        self.current_frame_idx = 0
        self.conf_history = []
        self.aleatoric_history = []
        self.epistemic_history = []
        self.frames_since_switch = 0
        self.current_model = 2
        self.previous_bbox = None
        return self._get_state()

    def _get_detection_for_track(self, frame_id):
        """Get detection with uncertainty for specific track"""
        if frame_id not in self.gt_tracks[self.track_id]:
            return None

        frame_detections = self.uncertainty_data['frames'][str(frame_id)]
        gt_bbox = self.gt_tracks[self.track_id][frame_id]['bbox']

        # Find detection closest to ground truth
        best_detection = None
        best_iou = 0.0

        for det in frame_detections:
            det_bbox = det['bbox']
            iou = self._compute_iou(gt_bbox, det_bbox)
            if iou > best_iou:
                best_iou = iou
                best_detection = det

        if best_detection is None or best_iou < 0.3:
            return None

        return best_detection

    def _compute_iou(self, bbox1, bbox2):
        """Compute IoU between two bboxes [x, y, w, h]"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        x1_max = x1 + w1
        y1_max = y1 + h1
        x2_max = x2 + w2
        y2_max = y2 + h2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1_max, x2_max)
        yi2 = min(y1_max, y2_max)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union_area = w1 * h1 + w2 * h2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _get_state(self):
        """Extract 12-dim state vector"""
        if self.current_frame_idx >= len(self.frames):
            return None

        frame_id = self.frames[self.current_frame_idx]
        detection = self._get_detection_for_track(frame_id)

        if detection is None:
            # Skip frame without detection
            self.current_frame_idx += 1
            return self._get_state()

        # Extract features
        conf = detection['confidence']
        bbox = detection['bbox']
        aleatoric = detection['aleatoric']
        epistemic = detection['epistemic']
        iou = detection.get('iou', 0.5)

        # Update histories
        self.conf_history.append(conf)
        self.aleatoric_history.append(aleatoric)
        self.epistemic_history.append(epistemic)

        # Confidence statistics
        conf_mean = np.mean(self.conf_history[-5:])
        conf_trend = (self.conf_history[-1] - self.conf_history[-2]) if len(self.conf_history) > 1 else 0.0

        # Object size (normalized)
        bbox_size = (bbox[2] * bbox[3]) / (1920 * 1080)  # MOT17-04 resolution

        # Edge distance (normalized)
        center_x = bbox[0] + bbox[2] / 2
        center_y = bbox[1] + bbox[3] / 2
        edge_dist = min(center_x, center_y, 1920 - center_x, 1080 - center_y) / 1920

        # Uncertainty statistics
        aleatoric_mean = np.mean(self.aleatoric_history[-5:])
        epistemic_mean = np.mean(self.epistemic_history[-5:])

        state = np.array([
            conf,                                    # 0
            conf_mean,                               # 1
            conf_trend,                              # 2
            iou,                                     # 3
            self.frames_since_switch / 100.0,        # 4 (normalized)
            self.current_model / 4.0,                # 5 (normalized)
            bbox_size,                               # 6
            edge_dist,                               # 7
            aleatoric,                               # 8
            aleatoric_mean,                          # 9
            epistemic,                               # 10
            epistemic_mean                           # 11
        ], dtype=np.float32)

        return state, detection

    def step(self, action):
        """Take action and return (next_state, reward, done, info)"""
        # Get current state info
        current_state, detection = self._get_state()
        if current_state is None:
            return None, 0.0, True, {}

        # Update model selection
        if action != self.current_model:
            self.frames_since_switch = 0
            self.current_model = action
        else:
            self.frames_since_switch += 1

        # Move to next frame
        self.current_frame_idx += 1

        # Get next state
        if self.current_frame_idx >= len(self.frames):
            done = True
            next_state = current_state  # Terminal state
            iou = detection['iou']
        else:
            done = False
            next_state_data = self._get_state()
            if next_state_data is None:
                done = True
                next_state = current_state
                iou = detection['iou']
            else:
                next_state, next_detection = next_state_data
                # IoU between consecutive frames
                if self.previous_bbox is not None:
                    iou = self._compute_iou(self.previous_bbox, detection['bbox'])
                else:
                    iou = detection['iou']
                self.previous_bbox = detection['bbox']

        # Compute reward
        reward = compute_orthogonal_reward(current_state, action, next_state, iou)

        info = {
            'frame': self.frames[self.current_frame_idx - 1],
            'iou': iou,
            'epistemic': current_state[10],
            'aleatoric': current_state[8],
            'model': MODEL_NAMES[action]
        }

        return next_state, reward, done, info


# ==================== DQN AGENT ====================
class DQNAgent:
    """Double DQN agent with experience replay"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[AGENT] Using device: {self.device}")

        # Networks
        self.q_network = DQN().to(self.device)
        self.target_network = DQN().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LR)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)

        # Exploration
        self.epsilon = EPSILON_START
        self.steps = 0

    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randint(0, ACTION_DIM - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def train_step(self):
        """Single training step using experience replay"""
        if len(self.replay_buffer) < BATCH_SIZE:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Double DQN: Use online network to select action, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + GAMMA * next_q * (1 - dones)

        # Loss
        loss = nn.MSELoss()(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

        return loss.item()

    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())


# ==================== TRAINING LOOP ====================
def train():
    """Main training loop"""
    print("\n" + "="*70)
    print(" RL TRAINING: ORTHOGONALITY-AWARE ADAPTIVE MODEL SELECTION")
    print("="*70)

    # Initialize environment and agent
    env = TrackingEnvironment(UNCERTAINTY_FILE, GT_FILE, track_id=1)
    agent = DQNAgent()

    # Training logs
    episode_rewards = []
    episode_lengths = []
    losses = []

    # Dump directory for intermediate results
    dump_dir = OUTPUT_DIR / "dumps"
    dump_dir.mkdir(exist_ok=True)

    print(f"\n[TRAIN] Starting training for {NUM_EPISODES} episodes...")
    print(f"[TRAIN] Results will be saved to: {OUTPUT_DIR}")

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_info = []

        while True:
            # Select action
            action = agent.select_action(state, training=True)

            # Take step
            next_state, reward, done, info = env.step(action)

            if next_state is None:
                break

            # Store experience
            agent.replay_buffer.push(state, action, reward, next_state, float(done))

            # Train
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            # Update target network periodically
            agent.steps += 1
            if agent.steps % TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()

            # Track episode
            episode_reward += reward
            episode_length += 1
            episode_info.append({
                'frame': info['frame'],
                'model': info['model'],
                'reward': reward,
                'epistemic': info['epistemic'],
                'aleatoric': info['aleatoric'],
                'iou': info['iou']
            })

            state = next_state

            if done:
                break

        # Log episode
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_loss = np.mean(losses[-100:]) if len(losses) > 0 else 0.0
            print(f"Episode {episode+1:3d}/{NUM_EPISODES} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Avg(10): {avg_reward:7.2f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Frames: {episode_length}")

        # Save intermediate dumps every 50 episodes
        if (episode + 1) % 50 == 0:
            dump_file = dump_dir / f"episode_{episode+1:03d}.pkl"
            with open(dump_file, 'wb') as f:
                pickle.dump({
                    'episode': episode + 1,
                    'episode_info': episode_info,
                    'avg_reward': np.mean(episode_rewards[-10:]),
                    'epsilon': agent.epsilon
                }, f)
            print(f"[DUMP] Saved intermediate results to {dump_file}")

    # Save final model
    model_path = OUTPUT_DIR / "dqn_final.pt"
    torch.save({
        'q_network': agent.q_network.state_dict(),
        'target_network': agent.target_network.state_dict(),
        'optimizer': agent.optimizer.state_dict(),
        'episode': NUM_EPISODES,
        'epsilon': agent.epsilon
    }, model_path)
    print(f"\n[SAVE] Model saved to {model_path}")

    # Save training history
    history = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'losses': losses,
        'hyperparameters': {
            'state_dim': STATE_DIM,
            'action_dim': ACTION_DIM,
            'hidden_dim': HIDDEN_DIM,
            'buffer_size': BUFFER_SIZE,
            'batch_size': BATCH_SIZE,
            'gamma': GAMMA,
            'lr': LR,
            'num_episodes': NUM_EPISODES
        }
    }

    history_path = OUTPUT_DIR / "training_history.pkl"
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"[SAVE] Training history saved to {history_path}")

    print("\n" + "="*70)
    print(" TRAINING COMPLETE")
    print("="*70)
    print(f"Final avg reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
    print(f"Final epsilon: {agent.epsilon:.3f}")

    return agent, env, history


# ==================== MAIN ====================
if __name__ == "__main__":
    agent, env, history = train()
