from learn_matrix_playground import learn_matrix_playground
from callbacks import log_progress
from matrix_games import PrisonersDilemma, BattleOfTheSexes, StagHunt, Chicken
from stable_baselines3 import PPO, A2C
from fixed_policy_learners import fixed_policy_always_cooperate,fixed_policy_always_defect,fixed_policy_coordinate, fixed_policy_cooperate_proportional

game = BattleOfTheSexes(2)
num_agents = 10
memory = 2
horizon = 10
learner = PPO
n_steps = 100
checkpoints= 10
save_model_checkpoints=False
agent_policies=None
num_agents_per_policy=None

'''
Example parameter definitions for 3 non-fixed, 7 fixed policy agents
If you run play.py with these parameters, you'll get output
   "Create 10 agents with [3, 5, 2] agents per policy, following policies ['real learner',
   'fixed_policy_always_cooperate', 'fixed_policy_always_defect']"
'''
# agent_policies=[None,fixed_policy_always_cooperate,fixed_policy_always_defect]
# num_agents_per_policy=[3,5,2]


callback = log_progress(
    game=game,
    num_agents=num_agents,
    memory=memory,
    horizon=horizon,
    checkpoints=checkpoints,
    save_model_checkpoints=save_model_checkpoints,
)
total_timesteps=2e4

learn_matrix_playground(
    game=game,
    num_agents=num_agents,
    memory=memory,
    horizon=horizon,
    learner=learner,
    n_steps=n_steps,
    callback=callback,
    total_timesteps=total_timesteps,
    agent_policies=agent_policies,
    num_agents_per_policy=num_agents_per_policy,
)