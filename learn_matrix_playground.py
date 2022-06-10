import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.utils import set_random_seed

from several_algorithms import SeveralAlgorithms

import matrix_playground
import supersuit as ss

# DON'T UNDERSTAND THIS
# BUT REMOVING IT IS CATASTROPHIC
import multiprocessing
multiprocessing.set_start_method("fork")

# define 2x2 Matrix Game
game = {
    ("C", "C"): (100, 100),
    ("C", "D"): (0, 200),
    ("D", "C"): (200, 0),
    ("D", "D"): (1, 1)
}
# and other globals
num_agents = 2 # must be even
memory = 3
horizon = 100

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
    model.learn(total_timesteps=int(100000))

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    env.render()
    input()
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)