from sqlite3 import SQLITE_DROP_TEMP_INDEX
from stable_baselines3.common.callbacks import BaseCallback
import matrix_playground
import os
from datetime import datetime
import pprint
import copy
import matplotlib.pyplot as plt

def action_to_interpretation(action):
    if action == 0:
        return "C"
    elif action == 1:
        return "D"
    else:
        assert False, "action should be 0 or 1"

class Move():
    def __init__(
        self,
        agent_action,
        opponent_action,
        agent_reward,
        opponent_reward
    ):

        self.agent = {
            "action": action_to_interpretation(agent_action),
            "reward": agent_reward
        }
        self.opponent = {
            "action": action_to_interpretation(opponent_action),
            "reward": opponent_reward
        }
    
    def write(self, file):
        file.write(
            self.agent["action"] + 
            "(" + str(self.agent["reward"]) + ")" + 
            ":" + self.opponent["action"] +
            "(" + str(self.agent["reward"]) + ")" + "\n"
        )

class MatchHistory():
    def __init__(
        self,
        agent,
        opponent
    ):
        self.agent = agent
        self.opponent = opponent
        self.moves = []
    
    def add_actions_and_rewards(
        self,
        agent_action,
        opponent_action,
        agent_reward,
        opponent_reward
    ):
        self.moves += [
            Move(agent_action, opponent_action, agent_reward, opponent_reward)
        ]

    def write(self, file):
        file.write("-:" + self.agent + ":" + self.opponent + "\n")
        for move in self.moves:
            move.write(file)

class EpisodeHistory():
    def __init__(
        self,
        rollout_start,
        start_timestamp,
        pair_selection
    ):
        self.rollout_start = rollout_start
        self.start_timestamp = start_timestamp
        self.pair_selection = pair_selection
        self.match_histories = []
        for match_ID, (agent, opponent) in enumerate(pair_selection.items()):
            if match_ID % 2 == 0:
                self.match_histories += [MatchHistory(agent, opponent)]

    def add_actions_and_rewards(
        self,
        actions,
        rewards
    ):
        for match_history in self.match_histories:
            agent = match_history.agent
            opponent = match_history.opponent
            match_history.add_actions_and_rewards(
                actions[agent],
                actions[opponent],
                rewards[agent],
                rewards[opponent]
            )

    def write(
        self,
        file
    ):
        file.write("*:" + str(self.rollout_start) + ":" + str(self.start_timestamp) +  "*\n")
        for match_history in self.match_histories:
            match_history.write(file)

class RolloutSummary():
    def __init__(
        self,
        num_agents
    ):
        self.num_agents = num_agents
        self.agents = [
            "agent_" + ID 
            for ID in range(self.num_agents)
        ]
        self.action_counts = {
            agent: {
                opponent: 0 for opponent in self.agents
            } for agent in self.agents
        }
        self.cumulative_rewards = {
            agent: {
                opponent: 0 for opponent in self.agents
            } for agent in self.agents
        }
        self.cooperation_counts = {
            agent: {
                opponent: 0 for opponent in self.agents
            } for agent in self.agents
        }

    def summarize(self):
        self.average_rewards = {
            agent: {
                opponent: self.cumulative_rewards[agent][opponent]/self.action_counts[agent][opponent]
                for opponent in self.agents
            } for agent in self.agents
        }
        self.average_cooperation = {
            agent: {
                opponent: self.cooperation_counts[agent][opponent]/self.action_counts[agent][opponent]
                for opponent in self.agents
            } for agent in self.agents
        }



class History():
    def __init__(
        self,
        num_agents
    ):

        self.num_agents = num_agents

        self.episode_histories = {}
        self.last_episode_history = None
        self.last_episode_start = 0

        self.rollout_summaries = {}
        self.last_rollout_summary = None
        self.last_rollout_start = 0

    def add_episode(
        self,
        rollout_start,
        timestamp,
        pair_selection
    ):
        if rollout_start:
            # we're about to start a new rollout
            # save a summary of the last rollout
            if self.last_rollout_summary is not None:
                self.last_rollout_summary.summarize()
            # start new rollout
            self.last_rollout_start = timestamp
            self.last_rollout_summary = RolloutSummary(self.num_agents)
            self.rollout_summaries[
                self.last_rollout_start
            ] = self.last_rollout_summary

        self.episode_histories[timestamp] = EpisodeHistory(
            rollout_start,
            timestamp,
            pair_selection,
            self.rollout_summaries[self.l]
        )
        self.last = self.episode_histories[timestamp]

    def add_actions_and_rewards(
        self,
        actions,
        rewards
    ):
        assert self.last is not None, "can only add actions and rewards after adding episode"
        self.last.add_actions_and_rewards(actions, rewards)

    def save(
        self,
        dir_path
    ):
        file = open(dir_path + "/histories.txt", "w")
        self.write(file)
        file.close()

    def write(
        self,
        file
    ):
        for timestamp, episode_history in self.episode_histories.items():
            episode_history.write(file)

class log_history(BaseCallback):
    def __init__(self, save=True, checkpoints=10):
        self.save = save
        self.checkpoints = checkpoints
        
    def _on_training_start(
        self,
        decentralized_on_policy_learners,
        start_time,
        stop_time,
        logs_path="logs"
    ) -> None:
        self.history = History()
        self.episode_start = True
        self.steps_to_save = (stop_time - start_time)/self.checkpoints
        self.steps_since_save = 0
        self.dir_path = logs_path + "/histories"
        if not os.path.isdir(self.dir_path):
            os.mkdir(self.dir_path)

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(
        self,
        decentralized_on_policy_learners,
        rollout_steps,
        actions,
        observations,
        rewards,
        dones,
        infos
    ) -> History:
        num_learners = decentralized_on_policy_learners.num_learners
        sample_learner = decentralized_on_policy_learners.learners[0]
        matrix_playground = decentralized_on_policy_learners.env.venv.vec_envs[0].par_env
        if self.episode_start:
            self.history.add_episode(
                rollout_start=(rollout_steps==0),
                timestamp = sample_learner.num_timesteps,
                pair_selection=matrix_playground.pair_selection
            )
        self.history.add_actions_and_rewards({
            "agent_" + str(ID): actions[ID] for ID in range(num_learners)
        }, { "agent_" + str(ID): rewards[ID] for ID in range(num_learners)
        })
        # next step will be an episode start if
        # and only if agent 0 (or, equivalently,
        # any other agent) is done.
        self.episode_start = dones[0]
        self.steps_since_save += 1
        return self.history

    def _on_rollout_end(self) -> dict:
        if self.save and self.steps_since_save >= self.steps_to_save:
            self.history.save(self.dir_path)

    def on_training_end(self) -> None:
        pass


class log_master(BaseCallback):
    def __init__(self):
        self.log_history = log_history()

    def _on_training_start(
        self,
        decentralized_on_policy_learners,
        start_time,
        stop_time
    ):

        self.game = decentralized_on_policy_learners.env.venv.vec_envs[0].par_env.game
        self.num_learners = decentralized_on_policy_learners.num_learners

        if not os.path.isdir("logs"):
            os.mkdir("logs")
        
        self.logs_path = "logs/" + self.game["label"] + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        os.mkdir(self.logs_path)

        self.log_history._on_training_start(
            decentralized_on_policy_learners,
            start_time,
            stop_time,
            self.logs_path
        )
        

class log_progress(BaseCallback):
    def __init__(
        self,
        game,
        num_agents,
        memory,
        horizon,
        checkpoints=10
        #sample_episodes=5
    ):
        self.game = game
        self.num_agents = num_agents
        self.memory = memory
        self.horizon = horizon

        self.checkpoints=checkpoints

        # the buffers keep the data that
        # has not yet been written to the records
        self.reward_buffers = [None]*self.num_agents
        self.cooperation_buffers = [None]*self.num_agents
        for ID in range(self.num_agents):
            self.reward_buffers[ID] = []
            self.cooperation_buffers[ID] = []
        self.timestamp_buffer = []

        # the lists contain all of the data
        # collected in the experiment so far.
        # we need the entire lists in order to update
        # the plots.
        self.reward_lists = [None]*self.num_agents
        self.cooperation_lists = [None]*self.num_agents
        for ID in range(self.num_agents):
            self.reward_lists[ID] = []
            self.cooperation_lists[ID] = []
        self.timestamp_list = []

        self.avg_reward = []
        self.avg_cooperation = []

        #self.sample_episodes = sample_episodes

        # check if logs folder exists
        if not os.path.isdir("logs"):
            # if not, create it
            os.mkdir("logs")

        # create folder inside of logs to store logs in
        # named by the date and time
        self.dir_name = "logs/" + self.game["label"] + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        os.mkdir(self.dir_name)


        # write a file that includes all of the
        # relevant parameters
        self.info_path = self.dir_name + "/info.txt"
        info = open(self.info_path, "w")
        info.write("game: " + pprint.pformat(game) + "\n")
        info.write("num_agents: " + str(num_agents) + "\n")
        info.write("memory: " + str(memory) + "\n")
        info.write("horizon: " + str(horizon) + "\n")
        info.close()

        # and initialize all the files we'll use throughout
        # for now we're only evaluating individual expected reward
        # and individual fractional_cooperation, as well as
        # saving timestamped models to study their
        # behaviors if necessary
        self.reward_path = self.dir_name + "/reward_record.txt"
        reward = open(self.reward_path, "w")
        reward.write("FORMAT\n")
        reward.write("timestep,agent_0_reward,agent_1_reward,...,agent_num_agents-1_reward\\n\n")
        reward.write("where type of timstep is int and type of reward is float\n")
        reward.write("START\n")
        reward.close()

        self.cooperation_path = self.dir_name + "/cooperation_record.txt"
        cooperation = open(self.cooperation_path, "w")
        cooperation.write("FORMAT\n")
        cooperation.write("timestep,agent_0_cooperation, agent_1_cooperation,..., agent_num_agents-1_cooperation\\n\n")
        cooperation.write("where type of timestep is int and type of cooperation is float\n")
        cooperation.write("START\n")
        cooperation.close()

        #self.sample_episodes_path = self.dir_name + "/sample_episodes.txt"
        #sample_episodes = open(self.sample_episodes_path, "w")
        #reward.write("FORMAT\n")

        # create directory to store models in
        self.models_dir_name = self.dir_name + "/models"
        os.mkdir(self.models_dir_name)

        # create directory to store plots in
        self.plots_dir_name = self.dir_name + "/plots"
        os.mkdir(self.plots_dir_name)

        # initialize all the relevant figures
        self.reward_fig = plt.figure()
        self.reward_plot = self.reward_fig.add_subplot()
        self.avg_reward_fig = plt.figure()
        self.avg_reward_plot = self.avg_reward_fig.add_subplot()
        
        self.cooperation_fig = plt.figure()
        self.cooperation_plot = self.cooperation_fig.add_subplot()
        self.avg_cooperation_fig = plt.figure()
        self.avg_cooperation_plot = self.avg_cooperation_fig.add_subplot()


    def _on_training_start(self, learner, stop_time) -> None:
        """
        This method is called before the first rollout starts.
        """
        info = open(self.info_path, "a")
        info.write("-"*30 + "\n")
        info.write("MODEL\n")
        info.write(pprint.pformat(learner.__dict__))
        info.close()

        self.buffer_size = stop_time/learner.n_steps/self.checkpoints

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        # reset cooperation_counts, cumulative rewards,
        # and histories
        # for upcoming collect_rollouts
        self.cooperation_counts = [0]*self.num_agents
        self.cumulative_rewards = [0]*self.num_agents
        #self.histories = []

    def _on_step(self, env, actions, rewards) -> bool:
        #self.num_rounds += 1
        # we evaluate the behavior of policies
        # in training in order to not have to run the
        # environment outside of training.
        for ID in range(self.num_agents):
            # we need 1 - action because
            # the encoding has 0 as cooperate
            self.cooperation_counts[ID] = self.cooperation_counts[ID] + 1 - actions[ID]
            self.cumulative_rewards[ID] = self.cumulative_rewards[ID] + rewards[ID]
        #if self.num_rounds == self.horizon - 1 and len(self.histories) < self.sample_episodes:
        #    self.histories += copy.deepcopy(env.histories)
        return True

    def _on_rollout_end(self, learners, timestamp) -> None:
        # :param learners: list of learners that
        # have been just trained. These technically
        # are stable baselines models

        # :param timestamp: timestamp at the
        # end of most recent training.

        # learners[i].n_steps is the same for
        # all [i] and it equals the number
        # of steps taken in collect_rollouts

        steps = learners[0].n_steps

        for ID in range(self.num_agents):
            self.cooperation_buffers[ID].append(
                self.cooperation_counts[ID]/steps
            )
            self.reward_buffers[ID].append(
                self.cumulative_rewards[ID]/steps
            )
        # we save timestamp - update_steps because
        # that is the timestamp at the beginning of
        # training, which corresponds to the timestamp
        # at which the tested policies had been trained
        self.timestamp_buffer += [timestamp - steps]

        # this is where I log
        if len(self.timestamp_buffer) >= self.buffer_size:
            # then it's time to write all the data
            # to the files, save the models,
            # and run some trials.

            # write rewards and cooperations
            reward_record = open(self.reward_path, "a")
            cooperation_record = open(self.cooperation_path, "a")
            for i, timestamp in enumerate(self.timestamp_buffer):
                reward_record.write(str(timestamp) + ",")
                cooperation_record.write(str(timestamp) + ",")
                for ID in range(self.num_agents):
                    if ID == 0:
                        self.avg_reward += [self.reward_buffers[ID][i]/self.num_agents]
                        self.avg_cooperation += [self.cooperation_buffers[ID][i]/self.num_agents]
                    else:
                        self.avg_reward[-1] = self.avg_reward[-1] + self.reward_buffers[ID][i]/self.num_agents
                        self.avg_cooperation[-1] = self.avg_cooperation[-1] + self.cooperation_buffers[ID][i]/self.num_agents
                    reward_record.write(str(self.reward_buffers[ID][i]))
                    cooperation_record.write(str(self.cooperation_buffers[ID][i]))
                    if ID < self.num_agents - 1:
                        reward_record.write(",")
                        cooperation_record.write(",")
                reward_record.write("\n")
                cooperation_record.write("\n")
            reward_record.close()
            cooperation_record.close()
            # append data in buffers to lists
            # and reset buffers
            for ID in range(self.num_agents):
                self.reward_lists[ID] += self.reward_buffers[ID]
                self.cooperation_lists[ID] += self.cooperation_buffers[ID]
            self.timestamp_list += self.timestamp_buffer
            for ID in range(self.num_agents):
                self.reward_buffers[ID] = []
                self.cooperation_buffers[ID] = []
            self.timestamp_buffer = []

            # update plots
            self.reward_plot.cla()
            for ID in range(self.num_agents):
                label = "agent_" + str(ID)
                self.reward_plot.plot(self.timestamp_list, self.reward_lists[ID], label=label)

            self.reward_plot.set_xlabel("timestep")
            self.reward_plot.set_ylabel("reward")

            if self.num_agents <= 10:
                self.reward_plot.legend()

            self.reward_plot.set_ylim([0, 1])

            self.reward_plot.set_title("reward per game during training")

            self.reward_fig.savefig(self.plots_dir_name + "/reward.png")

            self.avg_reward_plot.cla()
            self.avg_reward_plot.plot(self.timestamp_list, self.avg_reward)

            self.avg_reward_plot.set_xlabel("timestep")
            self.avg_reward_plot.set_ylabel("reward")

            self.avg_reward_plot.set_ylim([0, 1])

            self.avg_reward_plot.set_title("average reward per game during training")

            self.avg_reward_fig.savefig(self.plots_dir_name + "/avg_reward.png")

             # update plots
            self.cooperation_plot.cla()
            for ID in range(self.num_agents):
                label = "agent_" + str(ID)
                self.cooperation_plot.plot(self.timestamp_list, self.cooperation_lists[ID], label=label)

            self.cooperation_plot.set_xlabel("timestep")
            self.cooperation_plot.set_ylabel("# cooperations / # actions")

            if self.num_agents <= 10:
                self.cooperation_plot.legend()

            self.cooperation_plot.set_ylim([0, 1])

            self.cooperation_plot.set_title("cooperation during training")

            self.cooperation_fig.savefig(self.plots_dir_name + "/cooperation.png")

            self.avg_cooperation_plot.cla()
            self.avg_cooperation_plot.plot(self.timestamp_list, self.avg_cooperation)

            self.avg_cooperation_plot.set_xlabel("timestep")
            self.avg_cooperation_plot.set_ylabel("# cooperations / # actions")

            self.avg_cooperation_plot.set_ylim([0, 1])

            self.avg_cooperation_plot.set_title("average cooperation during training")

            self.avg_cooperation_fig.savefig(self.plots_dir_name + "/avg_cooperation.png")


            # save the models at the current timestamp
            timestamp_dir_name = self.models_dir_name + "/" + str(timestamp)
            os.mkdir(timestamp_dir_name)
            for ID in range(self.num_agents):
                learners[ID].save(timestamp_dir_name + "/agent_" + str(ID))
            #write_histories(timestamp_dir_name + "/sample_episodes.txt", self._histories)

    def _on_training_end(self, learners, timestamp) -> None:
        pass
