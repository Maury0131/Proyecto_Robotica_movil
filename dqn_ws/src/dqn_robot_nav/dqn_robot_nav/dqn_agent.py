import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


# ==============================
# Q-Network
# ==============================
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )

    def forward(self, x):
        return self.net(x)


# ==============================
# DQN Agent
# ==============================
class DQNAgent:
    def _init_(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.995,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.99,
        memory_size: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 500
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        # 🔧 Epsilon schedule
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_count = 0

        # Experience replay
        self.memory = deque(maxlen=memory_size)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        self.update_target_network()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    # ==============================
    # Memory
    # ==============================
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # ==============================
    # Action selection
    # ==============================
    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)

        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_network(state_t)

        return int(torch.argmax(q_values).item())

    # ==============================
    # Learning step
    # ==============================
    def replay(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # Q(s, a)
        current_q = self.q_network(states).gather(1, actions).squeeze(1)

        # Target Q
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Target network update
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    # ==============================
    # 🔧 Epsilon decay (per episode)
    # ==============================
    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    # ==============================
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    # ==============================
    # Save / Load
    # ==============================
    def save(self, path: str):
        torch.save({
            "q_net": self.q_network.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.step_count
        }, path)

    def load(self, path: str, reset_epsilon: bool = False):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_net"])
        self.target_network.load_state_dict(checkpoint["q_net"])
        self.step_count = checkpoint["steps"]

        if reset_epsilon:
            self.epsilon = 1.0
        else:
            self.epsilon = checkpoint["epsilon"]
