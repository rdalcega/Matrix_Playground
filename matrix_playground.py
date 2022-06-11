import numpy as np
import random
from pettingzoo import ParallelEnv
from gym.spaces.discrete import Discrete
from gym.spaces.multi_binary import MultiBinary
from pettingzoo.test import api_test, parallel_api_test
from pettingzoo.utils import wrappers
from pettingzoo.utils import parallel_to_aec

from matrix_games import PrisonersDilemma

# Set to True to run api_test and parallel_api_test
RANDOM_DEMO = False
PARALLEL_TEST = False
API_TEST = False


#UTILS

# random_pairs organizes n agents into pairs
# returning a dictionary keyed by agents
# and valued by their corresponding opponents.
def random_pairs(n):
    ordered = np.random.choice(n, n, replace=False)
    pair_selection = {}
    for i in range(n):
        if i%2 == 0:
            pair_selection["agent_" + str(ordered[i])] = "agent_" + str(ordered[i + 1])
        else:
            pair_selection["agent_" + str(ordered[i])] = "agent_" + str(ordered[i - 1])
    return pair_selection

# basically a dictionary of actions to interpretations:
def action_to_interpretation(action):
    if action == 0:
        return "C"
    elif action == 1:
        return "D"
    else:
        assert False, "action should be 0 or 1"

vec_action_to_interpretation = np.vectorize(action_to_interpretation)

# basically a dictionary of interpretations to encodings:
def interpretation_to_encoding(interpretation):
    if interpretation == "C":
        return [1, 0]
    elif interpretation == "D":
        return [0, 1]
    else:
        assert False, "interpretation should be C or D"

def agent_ID(agent):
    return int(agent[6:])

# adaptation of random_demo but for
# parallel_env. Otherwise its relationship
# to the render method is glitchy.
def random_demo(env, render=True, episodes=1):
    total_reward = 0
    completed_episodes = 0
    while completed_episodes < episodes:
        observations = env.reset()
        done = False
        while not done:
            if render:
                env.render()
            actions = {
                agent: env.action_spaces[agent].sample() for agent in env.agents
            }
            observations, rewards, dones, info = env.step(actions)
            for agent in rewards:
                total_reward += rewards[agent]
            done = dones["agent_0"]
        completed_episodes += 1
    env.close()
    print("Average Total Reward", total_reward / episodes)
    return

#ENVIRONMENT

def env(game=PrisonersDilemma, num_agents=4, memory=3, horizon=10):
    env = raw_env(game, num_agents, memory, horizon)
    # This wrapper is only for environments which print results to the terminal
    env = wrappers.CaptureStdoutWrapper(env)
    # This wrapper helps error handling for discrete action spaces
    # It's commented because it causes problems for a part of the
    # API test that tests out of bounds actions.
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # This wrapper provides a wide variety of helpful user errors
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env(game=PrisonersDilemma, num_agents=4, memory=3, horizon=10):
    env = parallel_env(game, num_agents, memory, horizon)
    # Because the matrix games are not turn based, all players
    # play in parallel. However, many of the API tests (and possibly
    # other plug and play tools) assume that the environment
    # is not parallel. So we wrap it in an environment that
    # is compatible with the standard AEC API.
    env = parallel_to_aec(env)
    return env

class parallel_env(ParallelEnv):

    #I'm not entirely sure what metadata["name"] is supposed
    #to mean. But I've seen it in the tutorials and a few
    #of the environments in the repo.
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, game=PrisonersDilemma, num_agents=4, memory=3, horizon=10):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        These attributes should not be changed after initialization.
        """

        """
        Arguments:

            game: a dictionary whose keys are pairs of actions and values are
            pairs of rewards, representing the reward matrix of a 2 x 2 matrix
            game. For example,
                {("C", "C"): (3, 3),
                 ("C", "D"): (0, 4),
                 ("D", "C"): (4, 0),
                 ("D", "D"): (1, 1)}
            represents a prisoner's dilemma where 1 is collaborate and 2 is defect

            num_of_agents: an even number representing the number
            of players on the playground. It's important to note that 
            this is not the number of players in the matrix game. Rather,
            this is the size of the population from which players are
            paired.

            memory: the number of actions agents remember for
            each other agent. Alternatively, the size of the
            observation space for each agent, for what each agent
            observes on each round is the last `memory' actions taken
            by the player they are about to play.

            horizon: the number of rounds that are played in each episode
        """

        # check validity of environment parameters:
        assert type(game) is dict, "game must be a dictionary with moves as keys and rewards as values"
        assert ("C", "C") in game, "game must allow mutual collaboration"
        assert ("C", "D") in game, "game must allow collaboration and defection"
        assert ("D", "C") in game, "game must allow defection and collaboration"
        assert ("D", "D") in game, "game must allow mutual defection"
        assert len(game) == 4, "game must be a 2 x 2 matrix game with four possible pairs of moves"

        assert type(num_agents) is int, "num_agents must be an int"
        assert num_agents > 0, "num_agents must be positive"
        assert num_agents % 2 == 0, "num_agents must be even"

        assert type(memory) is int, "memory must be an int"
        assert memory > 0, "memory must be positive"

        assert type(horizon) is int, "horizon must be an int"
        assert horizon > 0, "horizon must be positive"

        # store environment parameters
        self.game = { # translate actions to ints so as to not have to do this repeatedly
            (0, 0): game[("C", "C")],
            (0, 1): game[("C", "D")],
            (1, 0): game[("D", "C")],
            (1, 1): game[("D", "D")]
        }
        self.memory = memory
        self.horizon = horizon

        # Now onto pettingzoo API methods
        self.possible_agents = ["agent_" + str(ID) for ID in range(num_agents)]
        # For some reason I don't understand at the moment, even
        # if all agents have the same observation and action spaces,
        # in the environents whose code I've read
        # the self.observation_spaces consist of a different
        # instantiation of the gym space for each agent. I'll
        # be consistent with this practice to avoid trouble.
        self.action_spaces = dict(
            zip(self.possible_agents, [Discrete(2)]*num_agents)
        ) # every player can play 0 ("C") or 1 ("D")
        self.observation_spaces = dict(
            zip(self.possible_agents,
                [MultiBinary((num_agents - 1)*(1 + 2*2*self.memory))]*num_agents)
        ) # suppose, for example, that self.memory is 3 and num_agents = 4. Then
        # agent_0's observation when its about to play agent_2 will be something like
        # [0, 
        #   1, 0, 1, 0,
        #   0, 1, 0, 1,
        #   1, 0, 1, 0,
        #  1,
        #   1, 0, 0, 1,
        #   0, 1, 0, 1,
        #   0, 1, 1, 0,
        #  0,
        #   1, 0, 1, 0,
        #   1, 0, 1, 0,
        #   1, 0, 1, 0]
        # which is interpreted as
        # { "agent_1": {
        #       "opponent": False,
        #       -1: ["C", "C"],
        #       -2: ["D", "D"],
        #       -3: ["C", "C"] },
        #    "agent_2": {
        #       "opponent": True,
        #       -1: ["C", "D"],
        #       -2: ["D", "D"],
        #       -3: ["D", "C"] },
        #    "agent_3": {
        #       "opponent": False,
        #       -1: ["C", "C"],
        #       -2: ["C", "C"],
        #       -3: ["C", "C"] }
        #}
        # where each thruple indicated the
        # action taken according to the following rule
        # (1, 0) = "C"
        # (0, 1) = "D"

    def observation_space(self, agent):
        """
        A function that retrieves the observation space for a particular agent. This space should never
        change for a particular agent ID.
        """
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """
        A function that retrieves the observation space for a particular agent. This space should never
        change for a particular agent ID.
        """
        return self.action_spaces[agent]

    def render(self, mode="human"):
        if mode == "human":
            if self.num_rounds == self.horizon - 1:
                output = open("render.txt", "a")
                output.write("-"*30 + "\n")
                for i in range(self.num_agents):
                    for j in range(i+1, self.num_agents):
                        agent = "agent_" + str(i)
                        opponent = "agent_" + str(j)
                        output.write(agent + " vs " + opponent + "\n")
                        history = self.histories[agent][opponent]
                        for t in range(len(history)):
                            output.write("\t" + str(history[-1 - t]))
                            if t < self.memory:
                                output.write(" <-- (init)")
                            output.write("\n")
                output.close()

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections, or any other
        environent data which should not be kept around after the user is no longer
        using the environment.
        """
        # for now render outputs to stdout, which we don't need to release.
        # So we do nothing.
        pass

    def reset(self, seed=None):
        """
        Reset needs to initialize the `agents' attribute and must set up the environment so that
        render(), and step() can be called without issues and return the observations
        for each agent.
        """
        self.agents = self.possible_agents[:]
        self.num_rounds = 0
        self.histories = {
            agent: {
                opponent:
                    self.random_init_history()
                    for opponent in self.agents
            } for agent in self.agents
        }
        self.pair_selection = random_pairs(self.num_agents)
        observations = {
            agent: self.history_to_observation(agent, self.pair_selection[agent]) for agent in self.agents
        }
        return observations

    def random_init_history(self):
        as_ints = np.random.randint(0, 2, (self.memory, 2))
        return vec_action_to_interpretation(as_ints)

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
            - observations
            - rewards
            - dones
            - infos
        dictionaries where each dictionary looks like {agent_1:item_1, agent_2:item_2, \ldots }
        """

        # determine rewards according to game
        rewards = {
            agent: self.game[
                (actions[agent],
                actions[self.pair_selection[agent]])
                ][0] for agent in self.agents
        }

        self.num_rounds += 1

        #check if episode is over
        done = (self.num_rounds >= self.horizon)
        dones = {
            agent: done for agent in self.agents
        }

        # update memories to most recent actions
        for agent in self.agents:
            opponent = self.pair_selection[agent]
            old = self.histories[agent][opponent]
            self.histories[agent][opponent] = np.append(
                [[action_to_interpretation(actions[agent]), action_to_interpretation(actions[opponent])]],
                old,
                axis=0)

        # select new pairs and update observations accordingly
        self.pair_selection = random_pairs(self.num_agents)
        observations = {
            agent: self.history_to_observation(agent, self.pair_selection[agent]) for agent in self.agents
        }

        # for now info carries no info
        infos = {agent: {} for agent in self.agents}

        # API test expects self.agents to consist only
        # of agents that are not done. So if the game is over
        # set self.agents to an empty dictionary.

        if done: self.agents = []

        return observations, rewards, dones, infos

    def history_to_observation(self, agent, opponent):
        observation = np.array([])
        for player in self.agents:
            if player == agent:
                pass
            else:
                if player == opponent:
                    observation = np.append(observation, [1])
                else:
                    observation = np.append(observation, [0])
                history = self.histories[agent][opponent]
                for t in range(self.memory):
                    observation = np.append(observation, interpretation_to_encoding(history[t][0]))
                    observation = np.append(observation, interpretation_to_encoding(history[t][1]))
        return np.array(observation, dtype=np.uint8)


#TESTS

if RANDOM_DEMO:

    _env = parallel_env()
    random_demo(_env)

if PARALLEL_TEST:

    _env = parallel_env()
    parallel_api_test(_env, num_cycles=100)
    print("PASSED Parallel API Test!")

if API_TEST:

    _env = env()
    api_test(_env, num_cycles=100, verbose_progress=False)
