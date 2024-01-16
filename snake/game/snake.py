from snake.game.util import Dir
from numpy.typing import NDArray
import numpy as np


class Snake:
  body: list[NDArray]
  direction: Dir

  def __init__(self) -> None:
    self.body = [np.array(e) for e in [(10, 5), (9, 5), (8, 5), (7, 5)]]
    self.direction = Dir.RIGHT

  def is_intersecting(self) -> bool:
    tuples = [tuple(e) for e in self.body]
    return len(set(tuples)) < len(tuples)
