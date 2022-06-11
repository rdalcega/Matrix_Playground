import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.utils import set_random_seed

import supersuit as ss

import os

from several_algorithms import SeveralAlgorithms
import matrix_playground
from matrix_games import PrisonersDilemma, StagHunt, Chicken 

# DON'T UNDERSTAND THIS
# BUT REMOVING IT IS CATASTROPHIC
import multiprocessing
multiprocessing.set_start_method("fork")

# define 2x2 Matrix Game
game = PrisonersDilemma
# and other globals
num_agents = 10 # must be even
memory = 3
horizon = 20

# training parameters
training_timesteps = 1e3

env = matrix_playground.parallel_env(
        game,
        num_agents,
        memory,
        horizon
)

env = ss.pettingzoo_env_to_vec_env_v1(env)

# DON'T UNDERSTAND THIS EITHER
# BUT REMOVING IT IS ALSO CATASTROPHIC
env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=1,
        num_cpus=1,
        base_class='stable_baselines3'
      )

# SNIPPET OF HANNAH'S CODE
# USING SEVERAL ALGORITHMS
#
if True:
    # Instantiate the agent
    model = SeveralAlgorithms([
        PPO('MlpPolicy', env, verbose=3)
        for j in range(num_agents)
    ], env)
    # Train the agent
    model.learn(total_timesteps=training_timesteps)

# Enjoy trained agent
episodes = 10
file = open("render.txt", "w")
file.write("game=" + str(game) + "\n")
file.write("num_agents=" + str(num_agents) + "\n")
file.write("memory=" + str(memory) + "\n")
file.write("horizon=" + str(horizon) + "\n")
file.close()
obs = env.reset()
for i in range(episodes*horizon):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
