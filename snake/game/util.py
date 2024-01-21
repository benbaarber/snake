from enum import Enum
from operator import itemgetter


class Dir(Enum):
  UP = 0
  RIGHT = 1
  DOWN = 2
  LEFT = 3


def report(games: list[dict]) -> None:
  print("\nCheckpoint")
  print("Max time survived:", max(games, key="time")[0])
  print("High score:", max(games, key="score")[1])
