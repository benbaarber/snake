from collections import deque, namedtuple
from itertools import count
import math
import random
from typing import Any, Mapping
import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim
from numpy.typing import NDArray

from snake.game.field import Field
from snake.game.util import Dir

if torch.cuda.is_available():
  DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
  DEVICE = torch.device("mps")
else:
  DEVICE = torch.device("cpu")

# Replay Memory

Experience = namedtuple("Experience", ("state", "action", "next_state", "reward"))


class ReplayMemory:
  def __init__(self, capacity: int) -> None:
    self.memory = deque([], maxlen=capacity)

  def push(self, *args) -> None:
    """Save an experience"""
    self.memory.append(Experience(*args))

  def sample(self, batch_size: int):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)


class DQN(nn.Module):
  def __init__(self, field_size: int) -> None:
    super(DQN, self).__init__()

    self.fc1 = nn.Linear(field_size**2, 128).to(DEVICE)
    self.fc2 = nn.Linear(128, 128).to(DEVICE)
    self.fc3 = nn.Linear(128, 3).to(DEVICE)

  def forward(self, x: Tensor) -> Tensor:
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    q_values = self.fc3(x)

    return q_values

  # def __call__(self, *args: Any, **kwds: Any) -> Tensor:
  #   return super().__call__(*args, **kwds)


class Brain:
  BATCH_SIZE = 50  # number of experiences sampled from replay buffer
  GAMMA = 0.95  # discount factor
  EPS_START = 0.9  # starting value of epsilon
  EPS_END = 0.05  # ending value of epsilon
  EPS_DECAY = 1000  # rate of exponential decay of epsilon, higher value = slower decay
  TAU = 0.005  # update rate of the target network
  LR = 1e-4  # Learning rate of the AdamW optimizer

  def __init__(self, field_size: int) -> None:
    self.policy_net = DQN(field_size).to(DEVICE)
    self.target_net = DQN(field_size).to(DEVICE)
    self.target_net.load_state_dict(self.policy_net.state_dict())

    self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
    self.memory = ReplayMemory(250)
    self.steps_done = 0

  def act(self, state: Tensor) -> int:
    """Returns an action based on the current state. -1 = turn left, 0 = go straight, 1 = turn right"""
    eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
      -1.0 * self.steps_done / self.EPS_DECAY
    )
    self.steps_done += 1
    if random.random() > eps_threshold:
      with torch.no_grad():
        return torch.argmax(self.policy_net(state)).item()
    else:
      return random.randint(0, 2)

  def optim_step(self) -> None:
    if len(self.memory) < self.BATCH_SIZE:
      return

    experiences = self.memory.sample(self.BATCH_SIZE)
    batch = Experience(*zip(*experiences))

    non_final_mask = torch.tensor(
      tuple(map(lambda s: s is not None, batch.next_state)),
      device=DEVICE,
      dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    q_values = self.policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(self.BATCH_SIZE, device=DEVICE)
    with torch.no_grad():
      next_state_values[non_final_mask] = (
        self.target_net(non_final_next_states).max(1).values
      )

    expected_q_values = (next_state_values * self.GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(q_values, expected_q_values.unsqueeze(1))

    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad.clip_grad_value_(self.policy_net.parameters(), 100)
    self.optimizer.step()

  def play_game(self, field: Field, train=False) -> tuple[int, int]:
    """Train the DQN over a game of snake. Returns tuple (time survived, score)"""
    field.reset()
    state = torch.tensor(
      field.get_state(), dtype=torch.float32, device=DEVICE
    ).unsqueeze(0)
    for t in count():
      action = self.act(state)
      direction = Dir((field.snake.direction.value + action - 1) % 4)
      reward = field.step(direction)
      dead = reward < 0

      next_state = (
        torch.tensor(field.get_state(), dtype=torch.float32, device=DEVICE).unsqueeze(0)
        if not dead
        else None
      )
      if train:
        self.memory.push(
          state,
          torch.tensor([[action]], dtype=torch.long, device=DEVICE),
          next_state,
          torch.tensor(reward, dtype=torch.float32, device=DEVICE).unsqueeze(0),
        )
      state = next_state

      if train:
        self.optim_step()

        pnsd, tnsd = self.policy_net.state_dict(), self.target_net.state_dict()

        for key in pnsd:
          tnsd[key] = pnsd[key] * self.TAU + tnsd[key] * (1 - self.TAU)

        self.target_net.load_state_dict(tnsd)

      if dead:
        return t, field.score()

  def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
    self.policy_net.load_state_dict(state_dict)
    self.target_net.load_state_dict(state_dict)
