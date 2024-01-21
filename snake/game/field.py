from math import hypot, sqrt
import os
import time

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame as pg
import random
import numpy as np
from numpy.typing import NDArray

from snake.game.snake import Snake
from snake.game.util import Dir


BLACK = pg.Color(0, 0, 0)
WHITE = pg.Color(255, 255, 255)
RED = pg.Color(255, 0, 0)
GREEN = pg.Color(0, 255, 0)
BLUE = pg.Color(0, 0, 255)


class Field:
  window: pg.Surface
  fps: pg.time.Clock
  snake: Snake
  food: NDArray

  def __init__(self, size: int, speed: int, show_window=True) -> None:
    self.size = size
    self.speed = speed
    self.show_window = show_window
    self.snake = Snake(field_size=size)
    self.spawn_food()

    if show_window:
      pg.init()
      pg.display.set_caption("Snake")

      self.window = pg.display.set_mode((self.size * 10, self.size * 10))
      self.fps = pg.time.Clock()

      pg.event.get()

  def reset(self) -> None:
    self.snake = Snake(field_size=self.size)
    self.spawn_food()

  def score(self) -> int:
    return len(self.snake.body) - 4

  def spawn_food(self) -> None:
    self.food = np.array(
      (
        random.randrange(1, self.size),
        random.randrange(1, self.size),
      )
    )

  def is_in_bounds(self, pos: NDArray) -> bool:
    return np.all(np.logical_and(pos < self.size, pos > 0))

  def square_to_value(self, pos: NDArray) -> int:
    if pos == self.food:
      return 1
    elif pos in self.snake.body:
      return -1
    else:
      return 0

  def get_state(self) -> NDArray:
    data = np.zeros((self.size, self.size), dtype=np.float32)
    data[*self.food] = 1
    data[*self.snake.body[0]] = -0.5
    for segment in self.snake.body[1:]:
      data[*segment] = -1

    return data.T

  def show_score(self) -> None:
    font = pg.font.SysFont("roboto", 20)
    surface = font.render(f"Score: {self.score()}", True, WHITE)
    rect = surface.get_rect()
    self.window.blit(surface, rect)

  def show_death(self) -> None:
    font = pg.font.SysFont("roboto", 50)
    surface = font.render(f"Score: {self.score()}", True, RED)
    rect = surface.get_rect()
    rect.midtop = (self.size * 10 / 2, self.size * 10 / 4)
    self.window.blit(surface, rect)
    pg.display.flip()

  def render(self) -> None:
    self.window.fill(BLACK)

    for segment in self.snake.body:
      pg.draw.rect(self.window, GREEN, pg.Rect(*(segment * 10), 10, 10))

    pg.draw.rect(self.window, WHITE, pg.Rect(*(self.food * 10), 10, 10))

    self.show_score()

    pg.display.update()

  def step(self, turn: Dir) -> int:
    """Take a step in a direction. Returns a reward based on the outcome."""
    reward = 0
    head = self.snake.body[0]
    self.snake.direction = turn

    t = turn.value
    self.snake.body.insert(
      0,
      np.array((head[0] + (t & 1) * (2 - t), head[1] + ((t + 1) & 1) * (t - 1))),
    )

    head = self.snake.body[0]

    if np.all(head == self.food):
      self.spawn_food()
      reward = 1
    else:
      self.snake.body.pop()

    if self.snake.is_intersecting() or not self.is_in_bounds(head):
      return -1

    if self.show_window:
      self.render()
      self.fps.tick(self.speed)

    # return ((self.size * sqrt(2)) - hypot(*(self.food - head))) / (self.size * sqrt(2))
    return reward

  def play(self) -> None:
    """Play the game manually with arrow keys"""
    turn = self.snake.direction

    while True:
      for event in pg.event.get():
        if event.type == pg.KEYDOWN:
          match event.key:
            case pg.K_UP:
              if turn != Dir.DOWN:
                turn = Dir.UP
            case pg.K_RIGHT:
              if turn != Dir.LEFT:
                turn = Dir.RIGHT
            case pg.K_DOWN:
              if turn != Dir.UP:
                turn = Dir.DOWN
            case pg.K_LEFT:
              if turn != Dir.RIGHT:
                turn = Dir.LEFT

      result = self.step(turn)
      if result < 0:
        return self.show_death()
