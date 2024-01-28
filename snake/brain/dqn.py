from collections import deque, namedtuple
from itertools import count
import math
import random
from typing import Any, Mapping, TypedDict
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
  def __init__(
    self,
    field_size: int,
    conv1_out=16,
    conv2_out=32,
    fc1_out=32,
    fc2_out=64,
    **_,
  ) -> None:
    super(DQN, self).__init__()

    D_IN = field_size**2
    D_OUT = 4
    K_SIZE = 3

    self.conv1 = nn.Conv2d(1, conv1_out, K_SIZE).to(DEVICE)
    self.conv2 = nn.Conv2d(conv1_out, conv2_out, K_SIZE).to(DEVICE)

    reduced_size = int(math.sqrt(D_IN)) - ((K_SIZE - 1) * 2)
    conv_output_size = conv2_out * (reduced_size**2)

    self.fc1 = nn.Linear(conv_output_size, fc1_out).to(DEVICE)
    self.fc2 = nn.Linear(fc1_out, fc2_out).to(DEVICE)
    self.fc3 = nn.Linear(fc2_out, D_OUT).to(DEVICE)

  def forward(self, x: Tensor) -> Tensor:
    if x.dim() < 4:
      x = x.unsqueeze(0)
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = x.view(x.size(0), -1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    q_values = self.fc3(x)

    return q_values

  # def __call__(self, *args: Any, **kwds: Any) -> Tensor:
  #   return super().__call__(*args, **kwds)


class BrainConfig(TypedDict):
  buffer_size: int  # size of replay buffer
  batch_size: int  # number of experiences sampled from replay buffer
  gamma: float  # discount factor
  eps_start: float  # starting value of epsilon
  eps_end: float  # ending value of epsilon
  eps_decay: float  # rate of exponential decay of epsilon, higher value = slower decay
  tau: float  # update rate of the target network
  lr: float  # learning rate of the AdamW optimizer

  conv1_out: int  # number of feature maps for first conv2d layer
  conv2_out: int  # number of feature maps for second conv2d layer
  fc1_out: int  # number of features for first fully connected layer
  fc2_out: int  # number of features for second fully connected layer


class Brain:
  BATCH_SIZE = 256  # number of experiences sampled from replay buffer
  GAMMA = 0.99  # discount factor
  EPS_START = 0.9  # starting value of epsilon
  EPS_END = 0.05  # ending value of epsilon
  EPS_DECAY = 1000  # rate of exponential decay of epsilon, higher value = slower decay
  TAU = 0.005  # update rate of the target network
  LR = 1e-4  # learning rate of the AdamW optimizer

  def __init__(
    self, field_size: int, config: BrainConfig, train=False, loaded=False
  ) -> None:
    self.train = train
    self.loaded = loaded
    self.hp = config
    self.policy_net = DQN(field_size, **config).to(DEVICE)
    self.target_net = DQN(field_size, **config).to(DEVICE)
    self.target_net.load_state_dict(self.policy_net.state_dict())

    self.optimizer = optim.AdamW(
      self.policy_net.parameters(), lr=self.hp["lr"], amsgrad=True
    )
    self.criterion = nn.SmoothL1Loss()
    self.memory = ReplayMemory(self.hp["buffer_size"])
    self.steps_done = 0

  def act(self, state: Tensor) -> int:
    """Returns an action based on the current state. -1 = turn left, 0 = go straight, 1 = turn right"""
    if self.train and not self.loaded:
      hp = self.hp
      eps_threshold = hp["eps_end"] + (hp["eps_start"] - hp["eps_end"]) * math.exp(
        -1.0 * self.steps_done / hp["eps_decay"]
      )
      self.steps_done += 1
      if random.random() < eps_threshold:
        return random.randint(0, 2)

    with torch.no_grad():
      return torch.argmax(self.policy_net(state)).item()

  def optim_step(self) -> float:
    """Perform optimization step, returns loss"""
    if len(self.memory) < self.hp["batch_size"]:
      return

    experiences = self.memory.sample(self.hp["batch_size"])
    batch = Experience(*zip(*experiences))

    non_final_mask = torch.tensor(
      tuple(map(lambda s: s is not None, batch.next_state)),
      device=DEVICE,
      dtype=torch.bool,
    )
    non_final_next_states = (
      torch.cat([s for s in batch.next_state if s is not None])
      .unsqueeze(0)
      .transpose(0, 1)
    )

    state_batch = torch.cat(batch.state).unsqueeze(0).transpose(0, 1)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    q_values = self.policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(self.hp["batch_size"], device=DEVICE)
    with torch.no_grad():
      next_state_values[non_final_mask] = (
        self.target_net(non_final_next_states).max(1).values
      )

    expected_q_values = (next_state_values * self.hp["gamma"]) + reward_batch

    loss = self.criterion(q_values, expected_q_values.unsqueeze(1))

    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad.clip_grad_value_(self.policy_net.parameters(), 100)
    self.optimizer.step()

    return loss.item()

  def play_game(self, field: Field) -> dict:
    """Train the DQN over a game of snake. Returns tuple (time survived, score)"""
    field.reset()
    state = torch.from_numpy(field.get_state()).to(DEVICE).unsqueeze(0)
    for t in count():
      action = self.act(state)
      reward = field.step(Dir(action))
      dead = reward == -1

      next_state = (
        torch.from_numpy(field.get_state()).to(DEVICE).unsqueeze(0)
        if not dead
        else None
      )
      if self.train:
        self.memory.push(
          state,
          torch.tensor([[action]], dtype=torch.long, device=DEVICE),
          next_state,
          torch.tensor(reward, dtype=torch.float32, device=DEVICE).unsqueeze(0),
        )
      state = next_state

      if self.train:
        self.optim_step()

        pnsd, tnsd = self.policy_net.state_dict(), self.target_net.state_dict()

        for key in pnsd:
          tnsd[key] = pnsd[key] * self.hp["tau"] + tnsd[key] * (1 - self.hp["tau"])

        self.target_net.load_state_dict(tnsd)

      if dead:
        return {
          "time": t,
          "score": field.score(),
        }

  def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
    self.policy_net.load_state_dict(state_dict)
    self.target_net.load_state_dict(state_dict)
