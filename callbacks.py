from stable_baselines3.common.callbacks import BaseCallback
import matrix_playground
import os
from datetime import datetime
import pprint
import copy

# HELPERS

def write_histories(path, histories):
    file = open(path, "w")
    for history in histories:
        file.write("-"*30 + "\n")
        for ID, (agent, hist) in enumerate(history.items()):
            if ID % 2 == 0:
                for opponent, past in hist.items():
                    file.write(agent + " vs " + opponent + "\n")
                    for t in range(len(past)):
                        file.write("\t" + str(past[-1-t]))
                        if t < self.memory:
                            file.write(" <-- (init)")
                        file.write("\n")
    file.close()

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

        self.reward_buffers = [[]]*self.num_agents
        self.cooperation_buffers = [[]]*self.num_agents
        self.timestamp_buffer = []

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
        self.models_dir_name = self.dir_name + "/models"
        os.mkdir(self.models_dir_name)


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
            self.cooperation_counts[ID] += 1 - actions[ID]
            self.cumulative_rewards[ID] += rewards[ID]
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
            self.cooperation_buffers[ID] += [
                self.cooperation_counts[ID]/steps
            ]
            self.reward_buffers[ID] += [
                self.cumulative_rewards[ID]/steps
            ]
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
            reward = open(self.reward_path, "a")
            cooperation = open(self.cooperation_path, "a")
            for i, timestamp in enumerate(self.timestamp_buffer):
                reward.write(str(timestamp) + ",")
                cooperation.write(str(timestamp) + ",")
                for ID in range(self.num_agents):
                    reward.write(str(self.reward_buffers[ID][i]))
                    cooperation.write(str(self.cooperation_buffers[ID][i]))
                    if ID < self.num_agents - 1:
                        reward.write(",")
                        cooperation.write(",")
                reward.write("\n")
                cooperation.write("\n")
            # clear buffers
            self.reward_buffers = [[]]*self.num_agents
            self.cooperation_buffers = [[]]*self.num_agents
            self.timestamp_buffer = []
            # save the models at the current timestamp
            timestamp_dir_name = self.models_dir_name + "/" + str(timestamp)
            os.mkdir(timestamp_dir_name)
            for ID in range(self.num_agents):
                learners[ID].save(timestamp_dir_name + "/agent_" + str(ID))
            #write_histories(timestamp_dir_name + "/sample_episodes.txt", self._histories)

    def _on_training_end(self, learners, timestamp) -> None:
        pass
