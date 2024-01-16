from itertools import count
from operator import itemgetter
import torch
import argparse
from snake.brain.dqn import Brain
from snake.game.field import Field
from snake.game.util import report

FIELD_SIZE = 20
SNAKE_SPEED = 1000

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-p", "--path", help="Path to model .pth file")
  parser.add_argument("-m", "--manual", action="store_true", help="Play snake yourself")
  parser.add_argument(
    "-t",
    "--train",
    action="store_true",
    help="Run the Deep Q Learning optimization step",
  )
  parser.add_argument(
    "-e",
    "--episodes",
    help="Number of episodes to train for",
    type=int,
  )
  parser.add_argument(
    "-s",
    "--silent",
    action="store_true",
    help="Speed up training by not rendering the snake window",
  )

  args = parser.parse_args()

  field = Field(size=FIELD_SIZE, speed=SNAKE_SPEED, show_window=(not args.silent))

  if args.manual:
    field.play()
    exit()

  brain = Brain(field_size=FIELD_SIZE)

  if args.path:
    try:
      sd = torch.load(args.path)
      brain.load_state_dict(sd)
      print(f"Loaded model from {args.path}")
    except:
      print(f"Failed to load model from {args.path}")
  else:
    print("Initializing new model")

  if args.train:
    games = []
    if args.episodes is not None:
      for ep in range(args.episodes):
        print("EP", ep)
        game = brain.play_game(field, train=True)
        games.append(game)

        if len(games) % 50 == 0:
          report(games)
          torch.save(brain.policy_net.state_dict(), "./tmp.pth")
          games = []
      torch.save(brain.policy_net.state_dict(), "./tmp.pth")
    else:
      for ep in count():
        game = brain.play_game(field, train=True)
        games.append(game)
        print("EP", ep)

        if len(games) % 50 == 0:
          report(games)
          torch.save(brain.policy_net.state_dict(), "./tmp.pth")
          games = []
  else:
    time, score = brain.play_game(field)
    print(f"Score: {score} | Time survived: {time}")
