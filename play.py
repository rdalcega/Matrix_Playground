from learn_matrix_playground import learn_matrix_playground
from callbacks import log_progress, log_history
from matrix_games import PrisonersDilemma, BattleOfTheSexes, StagHunt, Chicken
from stable_baselines3 import PPO, A2C

game = BattleOfTheSexes(2)
num_agents = 6
memory = 1
horizon = 10
learner = PPO
n_steps = 100
checkpoints= 10
callback = log_history()
total_timesteps=1e3

learn_matrix_playground(
    game=game,
    num_agents=num_agents,
    memory=memory,
    horizon=horizon,
    learner=learner,
    n_steps=n_steps,
    callback=callback,
    total_timesteps=total_timesteps
)