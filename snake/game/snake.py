import random
from snake.game.util import Dir
from numpy.typing import NDArray
import numpy as np


class Snake:
  body: list[NDArray]
  direction: Dir

  def __init__(self, field_size: int) -> None:
    head = (random.randrange(5, field_size - 5), random.randrange(4, field_size - 4))
    body = [head]
    self.direction = Dir(random.randrange(4))
    match self.direction:
      case Dir.UP:
        body += [
          (head[0], head[1] + 1),
          (head[0], head[1] + 2),
          (head[0], head[1] + 3),
        ]
      case Dir.RIGHT:
        body += [
          (head[0] - 1, head[1]),
          (head[0] - 2, head[1]),
          (head[0] - 3, head[1]),
        ]
      case Dir.DOWN:
        body += [
          (head[0], head[1] - 1),
          (head[0], head[1] - 2),
          (head[0], head[1] - 3),
        ]
      case Dir.LEFT:
        body += [
          (head[0] + 1, head[1]),
          (head[0] + 2, head[1]),
          (head[0] + 3, head[1]),
        ]
    self.body = [np.array(e) for e in body]

  def is_intersecting(self) -> bool:
    tuples = [tuple(e) for e in self.body]
    return len(set(tuples)) < len(tuples)
