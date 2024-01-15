import pygame as pg
import random

from snake.game.snake import Dir, Snake

WINDOW_WIDTH = 720 / 2
WINDOW_HEIGHT = 480 / 2
SNAKE_SPEED = 15
BLACK = pg.Color(0, 0, 0)
WHITE = pg.Color(255, 255, 255)
RED = pg.Color(255, 0, 0)
GREEN = pg.Color(0, 255, 0)
BLUE = pg.Color(0, 0, 255)


class Field:
  window: pg.Surface
  fps: pg.time.Clock
  snake: Snake
  fruit: tuple[int, int]

  def __init__(self) -> None:
    pg.init()
    pg.display.set_caption("Snake")

    self.window = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    self.fps = pg.time.Clock()

    self.snake = Snake()
    self.spawn_fruit()

  def score(self) -> int:
    return len(self.snake.body) - 4

  def spawn_fruit(self):
    self.fruit = (
      random.randrange(1, WINDOW_WIDTH // 10) * 10,
      random.randrange(1, WINDOW_HEIGHT // 10) * 10,
    )

  def show_score(self):
    font = pg.font.SysFont("roboto", 20)
    surface = font.render(f"Score: {self.score()}", True, WHITE)
    rect = surface.get_rect()
    self.window.blit(surface, rect)

  def game_over(self):
    font = pg.font.SysFont("roboto", 50)
    surface = font.render(f"Score: {self.score()}", True, RED)
    rect = surface.get_rect()
    rect.midtop = (WINDOW_WIDTH / 2, WINDOW_HEIGHT / 4)
    self.window.blit(surface, rect)
    pg.display.flip()

  def run(self, render=True):
    turn = self.snake.direction
    head = self.snake.body[0]

    while True:
      for event in pg.event.get():
        if event.type == pg.KEYDOWN:
          match event.key:
            case pg.K_UP if turn != Dir.DOWN:
              turn = Dir.UP
            case pg.K_RIGHT if turn != Dir.LEFT:
              turn = Dir.RIGHT
            case pg.K_DOWN if turn != Dir.UP:
              turn = Dir.DOWN
            case pg.K_LEFT if turn != Dir.RIGHT:
              turn = Dir.LEFT

      t = turn.value
      self.snake.body.insert(
        0, (head[0] + (t & 1) * (2 - t) * 10, head[1] + (((t + 1) & 1) * (t - 1)) * 10)
      )

      head = self.snake.body[0]

      if head == self.fruit:
        self.spawn_fruit()
      else:
        self.snake.body.pop()

      if render:
        self.window.fill(BLACK)

        for segment in self.snake.body:
          pg.draw.rect(self.window, GREEN, pg.Rect(*segment, 10, 10))

        pg.draw.rect(self.window, WHITE, pg.Rect(*self.fruit, 10, 10))

        if (
          len(set(self.snake.body)) < len(self.snake.body)
          or head[0] < 0
          or head[0] > WINDOW_WIDTH - 10
          or head[1] < 0
          or head[1] > WINDOW_HEIGHT - 10
        ):
          self.game_over()
          break

        self.show_score()

        pg.display.update()
        self.fps.tick(SNAKE_SPEED)
