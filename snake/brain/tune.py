from ray import tune, train
from ray.tune.search.optuna import OptunaSearch
from ray.air.integrations.wandb import WandbLoggerCallback

from snake.brain.dqn import Brain
from snake.game.field import Field

TRAIN_EPISODES = 500
VAL_EPISODES = 50
EPOCHS = 5
FIELD_SIZE = 16

config = {
  "buffer_size": tune.choice([1000, 5000, 10000, 20000, 50000]),
  "batch_size": tune.choice([32, 64, 128, 256, 512]),
  "gamma": tune.uniform(0.8, 0.9999),
  "eps_start": tune.uniform(0.5, 1),
  "eps_end": tune.uniform(0, 0.5),
  "eps_decay": tune.loguniform(1e3, 1e6),
  "tau": tune.uniform(0.001, 0.1),
  "lr": tune.loguniform(1e-6, 1e-1),
  "conv1_out": tune.choice([16, 32, 64]),
  "conv2_out": tune.choice([32, 64, 128]),
  "fc1_out": tune.choice([64, 128, 256]),
  "fc2_out": tune.choice([128, 256, 512]),
}


a = tune.uniform(0, 1)

if __name__ == "__main__":

  def objective(config):
    field = Field(FIELD_SIZE, 1000, show_window=False)
    brain = Brain(FIELD_SIZE, config=config)

    for _ in range(EPOCHS):
      train_games = [brain.play_game(field, train=True) for _ in range(TRAIN_EPISODES)]
      val_games = [brain.play_game(field) for _ in range(VAL_EPISODES)]
      avg_score = sum(game["score"] for game in val_games) / VAL_EPISODES
      avg_time = sum(game["time"] for game in val_games) / VAL_EPISODES
      train.report(
        {
          "score": avg_score,
          "time": avg_time,
        }
      )

  tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(metric="score", mode="max", search_alg=OptunaSearch()),
    run_config=train.RunConfig(callbacks=[WandbLoggerCallback(project="snake")]),
    param_space=config,
  )

  results = tuner.fit()
