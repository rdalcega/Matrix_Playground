from stable_baselines3.common.callbacks import BaseCallback
import matrix_playground
import os
from datetime import datetime
import pprint
import copy
import matplotlib.pyplot as plt

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

            if self.num_agents < 10:
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

            if self.num_agents < 10:
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
