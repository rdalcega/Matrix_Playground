## Imports
import stable_baselines3
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
import numpy as np
from stable_baselines3 import PPO, A2C
import os

import gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from callbacks import log_progress
from matrix_games import PrisonersDilemma, BattleOfTheSexes, StagHunt, Chicken
from stable_baselines3 import PPO, A2C
import supersuit as ss
import pprint
import os
from decentralized_on_policy_learners import DecentralizedOnPolicyLearners
import matrix_playground
import types

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
import torch
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from torch.distributions import Categorical

## Entrypoint for getting custom learners
'''
Given an array of agent policies, and an array of the number of agents following each policy, return a list of learners.

:agent_policies: Array of policies for each agent to follow. Each value in the array is either None (for a regular non-fixed policy agent) or a function get_action_probabilities_from_observations_function(self,obs) ==> [list of action probabilities]
:num_agents_per_policy: the number of agents following each given policy
:return all_agents: a list of learners with the given composition of policies

This is the key entrypoint for getting learners with fixed and/or non-fixed policies. I'm leaving a few examples on how to use this function. If a policy in agent_policies is None, then the function returns a non-fixed policy agent.
Get 10 regular learner agents with non-fixed policies:
    get_all_agents(agent_policies=[None],num_agents_per_policy=[10])
Get 5 regular learner agents with non fixed policies, and 5 fixed-policy agents who choose each action at random.
    get_all_agents(agent_policies=[None, lambda self,obs: [0.5,0.5]],num_agents_per_policy=[5,5])
Get 5 regular learners, 5 learners following policy1, and 5 learners following policy2.
    get_all_agents(agent_policies=[None, policy_function1, policy_function2],num_agents_per_policy=[5,5,5])

Instructions for defining a custom fixed policy:
    The agent policy function must have input parameters
        get_action_probabilities_from_observations_function(self,obs)
    The agent policy function must return a python list of probabilies whose total sum equals one.
    This gives us flexibility in definining fixed policies, you can define a function that returns different action probabilities dependent on the values of the given input observations.
'''
def get_all_agents(env,n_steps,agent_policies=None,num_agents_per_policy=None,num_agents=10):
    # Input validation
    # Define our learners
    if agent_policies is None or num_agents_per_policy is None or agent_policies==(None,) or num_agents_per_policy==(None,):
        # Return only non-fixed learners
        agent_policies=[None]
        num_agents_per_policy=[num_agents]
    print(agent_policies,num_agents_per_policy,num_agents)
    assert(np.sum(num_agents_per_policy)==num_agents)

    # Define all learners
    all_agents = []
    for i in range(len(agent_policies)):
        if agent_policies[i] is None:
            # Create agents with non-fixed policies
            true_learners = [
                PPO(
                    'MlpPolicy',
                    env,
                    verbose=3,
                    gamma=1,
                    learning_rate=0.0003,
                    n_steps=n_steps
                ) for ID in range(num_agents_per_policy[i])
            ]
            all_agents += true_learners
        else:
            # Create agents with fixed policies
            fixed_learners = [getFixedPolicyAgent(env=env, n_steps=n_steps,get_action_probabilities_from_observations_function=agent_policies[i])
                        for ID in range(num_agents_per_policy[i])]
            all_agents += fixed_learners
    # Prettify policy names and print out agents' policies to sanity check
    policy_names = [policy.__name__ if policy is not None else 'real learner' for policy in agent_policies]
    print(f"Create {num_agents} agents with {num_agents_per_policy} agents per policy, following policies {policy_names}")
    return all_agents

## Define several fixed policies
# Note: I'm not sure whether obs including [0,1] means I cooperated and you defected, or you cooperated and I defected. In other words, what these fixed policies actually mean could be very different from what they are intended to mean.
def fixed_policy_always_cooperate(self,obs):
    # Always take action 0=cooperate
    return [1,0]
def fixed_policy_always_defect(self,obs):
    return [0,1]
def fixed_policy_coordinate(self,obs):
    '''
    If in the prior game we both cooperated, or I defected but you cooperated, then I will cooperate with probability 1 on the next game
    If I cooperated and you defected, I will cooperate 25 percent of the time
    If we both defected, I will cooperate 10 percent of the time
    '''
    past_actions = [int(x) for x in obs[0]] # Format: [0,1] or [0,0] or [1,0] or [1,1]
    most_recent_action_pair=past_actions[:2]
    if most_recent_action_pair==[0,0] or most_recent_action_pair==[0,1]:
        return [1,0]
    if most_recent_action_pair==[1,0]:
        return [0.25,0.75]
    else: # most_recent_action_pair==[1,1]:
        return [0.1,0.9]
def fixed_policy_cooperate_proportional(self,obs):
    # Take the avg of the # of times either you or I cooperated in the past, and cooperate with that probability
    past_actions = [int(x) for x in obs[0]]
    p_coop = np.mean(past_actions)
    return [1-p_coop,p_coop]

## Define CustomActorCriticPolicyFixed Class
# Define CustomActorCriticPolicyFixed so that we can control the actions selected by our fixed agent
# For our fixed agents, we set policy=CustomActorCriticPolicyFixed instead of policy='MlpPolicy'
class CustomActorCriticPolicyFixed(ActorCriticPolicy):
    def get_action_probabilities_from_observations(self,obs):
        '''
        :param obs: Observation. tensor of observations whose length depends on the memory parameters.
                E.g. If memory = 2: obs = tensor([[1, 0, 1, 1]], dtype=torch.int8)
                     If memory = 1: obs = tensor([[0, 1]], dtype=torch.int8)
        :return: action_probs - list such that len(action_probs)==num_actions, sum(action_probs)==1, where there is an action_probs[i] chance that we take the i-th action
        '''
        # Add an error message to make sure that the user overrides this function to define an obs --> action probability mapping
        raise Exception("Function not Overriden Error: You created an instance of class CustomActorCriticPolicyFixed, but you didn't override it's function get_action_probabilities_from_observations() with a custom fixed policy. This method should be overriden. Add a statment like  CustomActorCriticPolicyFixed.get_action_probabilities_from_observations = lambda self,obs: [0.5,0.5] before you pass this custom policy to a fixed learner.")
        # Example return value - equal probabilities of each action
        return [0.5,0.5]

    # This function controls what action we take given an input observation
    # We overwrite this function to control the action probabilities
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
            """
            Forward pass in all the networks (actor and critic)

            :param obs: Observation
            :param deterministic: Whether to sample or use deterministic actions
            :return: action, value and log probability of the action
            """

            # Preprocess the observation if needed
            features = self.extract_features(obs)
            latent_pi, latent_vf = self.mlp_extractor(features)
            # Evaluate the values for the given observations
            values = self.value_net(latent_vf)
            # Obtain stablebaselines CategoricalDistribution
            distribution = self._get_action_dist_from_latent(latent_pi)

            # NOTE: important edit: This is where we create a Categorical distribution with fixed action probabilities
            # Additional note on speed: right now we're re-creating a categorical distribution every time that forward() is called. This isn't too slow, but we might be able to speed this up later by, upon game initialization, creating some list of all the categorical distributions we need, and then using the relevant distribution here
            # Define action_probabilities based on the input obs
            action_probabilities = self.get_action_probabilities_from_observations(obs)
            # Format action_probabilities as a torch tensor
            action_probabilities = torch.tensor([action_probabilities])
            distribution.distribution = Categorical(probs=action_probabilities)

            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)
            return actions, values, log_prob



## Create Fixed-policy agents

# We don't currently call learner.predict() anywhere, so we don't technically need to override this function right now. However, that could change in the future, so I'm going to override it anyways here.
def fixed_predict(observation, state=None, episode_start=None, deterministic=False):
    return (np.asarray([0]))

# We don't currently call learner.learn() anywhere, so we don't technically need to override this function right now. However, that could change in the future, so I'm going to override it anyways here.
def fixed_learn(self,total_timesteps, callback=None, log_interval=100, tb_log_name='run', eval_env=None, eval_freq=- 1, n_eval_episodes=5, eval_log_path=None, reset_num_timesteps=True):
    # Return a trained model
    return self

# Override train() to do nothing, so we don't waste time computing unnecessary info.
def fixed_train(self):
    """
        Update policy using the currently gathered rollout buffer.
        In our case for a fixed learner, do nothing to our policy and simply return
    """
    return None


# Create a fixed-policy agent where we define the obs-->action probability mapping, and override its learn(), predict(), and train() functions
def getFixedPolicyAgent(env, n_steps, get_action_probabilities_from_observations_function = lambda self,obs: [0.5,0.5]):
        # Create a fixed agent policy, and define it's get_action_probabilities_from_observations() function
        # get_action_probabilities_from_observations() defines the probability of taking each action, given some observation
        policy = CustomActorCriticPolicyFixed
        policy.get_action_probabilities_from_observations = get_action_probabilities_from_observations_function

        # Initialize learner with our custom policy
        learner = PPO(
            policy=policy,
            env=env,
            verbose=0,
            gamma=1,
            learning_rate=0.01,
            n_steps=n_steps
        )

        # Override the learn, predict, and train methods of the fixed learner
        learner.learn = types.MethodType(fixed_learn, learner)
        learner.predict = types.MethodType(fixed_predict, learner)
        learner.train = types.MethodType(fixed_train, learner)

        return learner

## Testing this class
# This part of the file isn't important to understand, it's just driver code here to test that the fixed agents code is working as intended.
# Define constants for game
game = BattleOfTheSexes(2)
num_agents = 10
memory = 1
horizon = 10
learner = PPO
n_steps = 100
checkpoints= 10
save_model_checkpoints=False
callback = log_progress(
    game=game,
    num_agents=num_agents,
    memory=memory,
    horizon=horizon,
    checkpoints=checkpoints,
    save_model_checkpoints=save_model_checkpoints,
)
total_timesteps=200

def get_callback(game,num_agents,memory,horizon,checkpoints,save_model_checkpoints):
    return log_progress(
        game=game,
        num_agents=num_agents,
        memory=memory,
        horizon=horizon,
        checkpoints=checkpoints,
        save_model_checkpoints=save_model_checkpoints,
    )

def get_env(num_agents=num_agents):
    # create parallel_env using parameters
    env = matrix_playground.parallel_env(
        game=game,
        num_agents=num_agents,
        memory=memory,
        horizon=horizon
    )
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=1,
        num_cpus=1,
        base_class='stable_baselines3'
    )
    return env

def testFixedAgents():
    num_true_agents = 9
    num_fixed_agents = 1

    num_agents = num_true_agents+num_fixed_agents
    env = get_env(num_agents=num_agents)
    true_learners = [
        PPO(
            'MlpPolicy',
            env,
            verbose=3,
            gamma=1,
            learning_rate=0.0003,
            n_steps=n_steps
        ) for ID in range(num_true_agents)
    ]

    fixed_learners = [getFixedPolicyAgent(env=env, n_steps=n_steps)
                        for ID in range(num_fixed_agents)]
    all_learners = true_learners+fixed_learners
    model_learners = all_learners
    callback = get_callback(game,num_agents,memory,horizon,checkpoints,save_model_checkpoints)
    print(f"True Learners: {len(true_learners)}, Fixed Learners: {len(fixed_learners)}, All Learners: {len(all_learners)}")

    model = DecentralizedOnPolicyLearners(model_learners, env)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback
    )
    print("Model with fixed agents finished learning successfully.")
    return