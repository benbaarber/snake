from enum import Enum


class Dir(Enum):
  UP = 0
  RIGHT = 1
  DOWN = 2
  LEFT = 3


class Snake:
  body: list[tuple[int, int]]
  direction: Dir

  def __init__(self) -> None:
    self.body = [(100, 50), (90, 50), (80, 50), (70, 50)]
    self.direction = Dir.RIGHT
