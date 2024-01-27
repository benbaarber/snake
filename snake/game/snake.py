import random
from snake.game.util import Dir
from numpy.typing import NDArray
import numpy as np


class Snake:
  body: list[NDArray]
  direction: Dir

  def __init__(self, field_size: int) -> None:
    head = (random.randrange(2, field_size - 2), random.randrange(2, field_size - 2))
    self.direction = Dir(random.randrange(4))
    self.body = [np.array(head)]

  def is_intersecting(self) -> bool:
    tuples = [tuple(e) for e in self.body]
    return len(set(tuples)) < len(tuples)
