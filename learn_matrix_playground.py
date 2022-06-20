import gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

import supersuit as ss

import pprint

import os

from decentralized_on_policy_learners import DecentralizedOnPolicyLearners

import matrix_playground

from matrix_games import PrisonersDilemma

"""
NOTE: At the moment the code is set up so that
training is done on a single CPU. It's, in principle,
possible to change this by changing the parameters
num_vec_envs and num_cpus in concat_vec_envs.

However, a prerequisite for this working on
recent versions of MacOS  as
expected is to run the following lines

import multiprocessing
multiprocessing.set_start_method("fork")

The reason this is necessary is that for recent
versions of MacOS and recent versions of Python,
the default way to start new processes
is to "spawn" them from the parent. This means that
when a parent process creates a new thread, 
this thread has a "fresh" python interpreter
process. However, it seems like soome of the libraries
we're using assume that when a parent process creates
a new thread, this thread is a fork of the parent
process. So we have to override the default setting.


"""

def learn_matrix_playground(
    game=PrisonersDilemma(),
    num_agents=2,
    memory=1,
    horizon=10,
    learner=PPO,
    n_steps=100,
    callback=None,
    total_timesteps=1e5):

    # create parallel_env using parameters
    env = matrix_playground.parallel_env(
        game=game,
        num_agents=num_agents,
        memory=memory,
        horizon=horizon
    )

    #env.reset()

    # use supersuit to wrap env, which is
    # a petting zoo parallel_env into
    # a vec_env that can be learned
    # using stable_baselines3

    env = ss.pettingzoo_env_to_vec_env_v1(env)

    # the prior line returns a vector environment
    # that is compatible with gym's API. However,
    # this is not compatible with the stable
    # baselines 3 API. So we need to wrap it
    # in a stable baselines 3 vec env. Supersuit
    # has an internal class that does this.
    # It's called SB3VecEnvWrapper. However,
    # this is used internally in the following
    # public constructor. With parameters
    # num_vec_envs=1 and num_cpus=1, all the
    # following method does is apply
    # SB3VecEnvWrapper to produce a vec env
    # that is compatible with the stable
    # baselines 3 API.

    # I'm not bothering to simply use SB3VecEnvWrapper
    # for two reasons: (1) I tried to import it
    # and it's not obvious to me how. (2) It seems
    # to me like it can't hurt to leave this extra
    # versatility in the code as long as it doesn't
    # really do anything other that SB3VecEnvWrapper
    # with the default parameters. If you disagree
    # with (2) and can easily overcome (1), go for it!
    
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=1,
        num_cpus=1,
        base_class='stable_baselines3'
    )

    # SeveralAlgorithms is an adaptation
    # of the StableBaselines3 that (will?) allow
    # us to train the agents independently,
    # in parallel, each with its own architecture
    # and training method.

    model = DecentralizedOnPolicyLearners([
        learner(
            'MlpPolicy',
            env,
            verbose=3,
            gamma=1,
            learning_rate=0.0003,
            n_steps=n_steps
        ) for ID in range(num_agents)
    ], env)

    # Learn does what you'd expect it to do: it learns!
    # Again, the learners learn independently according
    # to the policies chosen in the prior line.

    # The callback API is the same as the callback
    # API for stable baselines 3, and it is documented
    # here: 
    # https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback
    )

    return