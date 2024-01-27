from enum import Enum
from operator import itemgetter


class Dir(Enum):
  UP = 0
  RIGHT = 1
  DOWN = 2
  LEFT = 3


def report(games: list[dict]) -> None:
  print("\nCheckpoint")
  print("Max time survived:", max(games, key=itemgetter("time"))["time"])
  print("High score:", max(games, key=itemgetter("score"))["score"])
