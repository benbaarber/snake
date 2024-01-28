from itertools import count
from operator import itemgetter
import torch
import argparse
from snake.brain.dqn import Brain
from snake.game.field import Field
from snake.game.util import report

FIELD_SIZE = 8

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
    "-g",
    "--games",
    help="Number of games to play",
    type=int,
  )
  parser.add_argument(
    "-s",
    "--silent",
    action="store_true",
    help="Speed up training by not rendering the snake window",
  )

  args = parser.parse_args()

  field = Field(
    size=FIELD_SIZE, speed=5 if args.manual else 10, show_window=(not args.silent)
  )

  if args.manual:
    field.play()
    exit()

  brain = Brain(
    field_size=FIELD_SIZE,
    config={
      "buffer_size": 50000,
      "batch_size": 128,
      "gamma": 0.86,
      "eps_start": 0.915,
      "eps_end": 0.1,
      "eps_decay": 18276,
      "lr": 3.58e-3,
      "tau": 2.7e-2,
      "conv1_out": 32,
      "conv2_out": 128,
      "fc1_out": 64,
      "fc2_out": 128,
    },
    train=args.train,
    loaded=(args.path is not None),
  )

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
    for ep in range(args.games) if args.games else count():
      game = brain.play_game(field)
      print("EP:", ep, " - Score:", game["score"], " - Time:", game["time"])
      games.append(game)

      if len(games) % 50 == 0:
        report(games)
        torch.save(brain.policy_net.state_dict(), "./tmp.pt")
        games = []
    torch.save(brain.policy_net.state_dict(), "./tmp.pt")

  else:
    for ep in range(args.games or 1):
      game = brain.play_game(field)
      print(f"Score: {game['score']} | Time survived: {game['time']}")
