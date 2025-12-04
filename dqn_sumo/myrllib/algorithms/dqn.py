from __future__ import annotations
import random
import math
from dataclasses import dataclass
from typing import Deque, Tuple, Optional
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def layer_init(layer: nn.Linear, std: float = 1.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: Tuple[int, int] = (128, 128)):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            layer_init(nn.Linear(input_dim, h1), std=math.sqrt(2)),
            nn.ReLU(),
            layer_init(nn.Linear(h1, h2), std=math.sqrt(2)),
            nn.ReLU(),
            layer_init(nn.Linear(h2, output_dim), std=0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DQNConfig:
    state_dim: int
    action_dim: int
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 100_000
    min_buffer_to_learn: int = 1_000
    tau: float = 1.0            # hard update when >=1.0
    target_update_every: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    grad_clip_norm: float = 10.0
    seed: int = 42
    hidden: Tuple[int, int] = (128, 128)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 42):
        self.capacity = capacity
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)
        random.seed(seed)
        np.random.seed(seed)

    def push(self, s, a, r, s2, d):
        self.buffer.append((np.array(s, dtype=np.float32),
                            int(a),
                            float(r),
                            np.array(s2, dtype=np.float32),
                            bool(d)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, cfg: DQNConfig):
        self.cfg = cfg
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        self.device = torch.device(cfg.device)
        self.q = MLP(cfg.state_dim, cfg.action_dim, cfg.hidden).to(self.device)
        self.q_target = MLP(cfg.state_dim, cfg.action_dim, cfg.hidden).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optimizer = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size, cfg.seed)

        # epsilon schedule
        self._eps = cfg.epsilon_start
        self._eps_step = (cfg.epsilon_start - cfg.epsilon_end) / max(1, cfg.epsilon_decay_steps)
        self.global_step = 0

    @property
    def epsilon(self) -> float:
        return self._eps

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        """epsilon-greedy policy"""
        self.global_step += 1
        if (not greedy) and random.random() < self._eps:
            action = random.randrange(self.cfg.action_dim)
        else:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.q(s)
                action = int(torch.argmax(q_values, dim=1).item())
        # decay eps
        if self._eps > self.cfg.epsilon_end:
            self._eps = max(self.cfg.epsilon_end, self._eps - self._eps_step)
        return action

    def remember(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)

    def _update_target(self):
        # hard update (copy) by default
        self.q_target.load_state_dict(self.q.state_dict())

    def replay(self) -> Optional[dict]:
        if len(self.buffer) < self.cfg.min_buffer_to_learn:
            return None

        s, a, r, s2, d = self.buffer.sample(self.cfg.batch_size)
        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(-1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(-1)
        s2 = torch.tensor(s2, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(-1)

        q_vals = self.q(s).gather(1, a)
        with torch.no_grad():
            next_q = self.q_target(s2).max(dim=1, keepdim=True)[0]
            target = r + (1.0 - d) * self.cfg.gamma * next_q

        loss = nn.functional.smooth_l1_loss(q_vals, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip_norm)
        self.optimizer.step()

        if self.global_step % self.cfg.target_update_every == 0:
            self._update_target()

        return {"loss": float(loss.item()),
                "epsilon": float(self._eps),
                "buffer": len(self.buffer)}

    def save(self, path: str):
        torch.save({
            "model": self.q.state_dict(),
            "target": self.q_target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "cfg": self.cfg.__dict__,
            "global_step": self.global_step,
            "epsilon": self._eps
        }, path)

    def load(self, path: str, map_location: Optional[str] = None):
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.q.load_state_dict(ckpt["model"])
        self.q_target.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = ckpt.get("global_step", 0)
        self._eps = ckpt.get("epsilon", self.cfg.epsilon_end)
