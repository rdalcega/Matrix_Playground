import torch as th
import numpy as np
import stable_baselines3
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.utils import obs_as_tensor,safe_mean

import gym

def SLICE_TENSOR(tensor, i):
    return tensor[i:i+1, ...]
def SLICE_LIST(l, i):
    return l[i:i+1]
def ONLY_ITEM_IN_THE_LIST(l):
    assert len(l) == 1
    return l[0]
def VALUE_OF_LIST_WHICH_IS_ALL_THE_SAME_VALUE(l):
    what = l[0]
    for i in range(1, len(l)):
        if what != l[i]:
            raise Exception("List doesn't all have the same value! " + str(l))
    return what

class SeveralAlgorithms:
    """A class that combines several algorithms with separate parameters into one big algorithm.
    It depends a lot on the internals of stable_baselines3."""

    def __init__(
        self,
        algorithmlist,
        env,
        use_sde = False,
        sde_sample_freq = -1
    ):
        self.algorithmlist = algorithmlist
        self.env = env
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq

    def _policies(self):
        return [algorithm.policy for algorithm in self.algorithmlist]
    
    def _setup_learn(self):
        self._last_obs = self.env.reset()
        self._last_episode_starts = np.ones((len(self.algorithmlist),), dtype=bool)
        self.ep_info_buffer = [[] for j in range(len(self.algorithmlist))]

    def _update_info_buffer(self, informations):
        "If we got some information, add it to the appropriate info buffer."
        for idx,info in enumerate(informations):
            episode = info.get("episode")
            if episode is not None:
                self.ep_info_buffer[idx].append(episode)
                #print(episode)

    def save(self, path, exclude = None, include = None):
        for idx in range(len(self.algorithmlist)):
            self.algorithmlist[idx].save(path + "-%d" % idx, exclude, include)

    # I guess
    # something like this, maybe?
    def predict(self, observation, state = None, episode_start = None, deterministic = False):
        predictions = [policy.predict(observation, state, episode_start, deterministic) for policy in self._policies()]
        actions = [p[0] for p in predictions]
        states = [p[1] for p in predictions]
        return np.concatenate(actions), states

    def collect_rollouts(
        self,
        env,
        callback,
        rollout_buffer_list,
        n_rollout_steps
    ):
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)

        n_algorithms = len(self.algorithmlist)
        policies = [algorithm.policy for algorithm in self.algorithmlist]
        n_envs = 1

        device = VALUE_OF_LIST_WHICH_IS_ALL_THE_SAME_VALUE([algorithm.device for algorithm in self.algorithmlist])

        for policy in policies:
            policy.set_training_mode(False)

        n_steps = 0

        for rollout_buffer in rollout_buffer_list:
            rollout_buffer.reset()

        # Sample new weights for the state dependent exploration
        if self.use_sde:
            for policy in policies:
                n_envs = 1
                policy.reset_noise(n_envs)

        #callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                for policy in policies:
                    policy.reset_noise(n_envs)

            act_list = [None] * n_algorithms
            val_list = [None] * n_algorithms
            lpr_list = [None] * n_algorithms
            clipped_action_list = [None] * n_algorithms

            for algorithm_ix in range(n_algorithms):
                with th.no_grad():
                    slice_of_observation = SLICE_TENSOR(self._last_obs, algorithm_ix)
                    obs_tensor = obs_as_tensor(slice_of_observation, device)
                    actions, values, log_probs = policies[algorithm_ix](obs_tensor)
                actions = actions.cpu().numpy()

                # Rescale and perform action
                clipped_actions = actions
                # Clip the actions to avoid out of bound error
                #if isinstance(self.action_space, gym.spaces.Box):
                    #clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)


                act_list[algorithm_ix] = actions
                val_list[algorithm_ix] = values
                lpr_list[algorithm_ix] = log_probs
                clipped_action_list[algorithm_ix] = clipped_actions

            with th.no_grad():
                act_list = np.array(act_list)
                val_list = th.tensor(val_list)
                lpr_list = th.tensor(lpr_list)

            flattened_clipped_action_list = [ONLY_ITEM_IN_THE_LIST(actions) for actions in clipped_action_list]

            new_obs, rewards, dones, infos = env.step(flattened_clipped_action_list)

            for algorithm in self.algorithmlist:
                algorithm.num_timesteps += 1

            # Give access to local variables
            #callback.update_locals(locals())
            #if callback.on_step() is False:
            #    return False

            self._update_info_buffer(infos)
            n_steps += 1

            if VALUE_OF_LIST_WHICH_IS_ALL_THE_SAME_VALUE([isinstance(algorithm.action_space, gym.spaces.Discrete) for algorithm in self.algorithmlist]):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.algorithmlist[idx].policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.algorithmlist[idx].policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.algorithmlist[idx].gamma * terminal_value

            for algorithm_ix in range(n_algorithms):
                last_obs = SLICE_TENSOR(self._last_obs, algorithm_ix)
                action = SLICE_TENSOR(act_list, algorithm_ix)
                value = SLICE_TENSOR(val_list, algorithm_ix)
                log_prob = SLICE_TENSOR(lpr_list, algorithm_ix)
                
                reward = SLICE_LIST(rewards, algorithm_ix)
                last_done = SLICE_LIST(self._last_episode_starts, algorithm_ix)
                rollout_buffer_list[algorithm_ix].add(last_obs, action, reward, last_done, value, log_prob)

            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            for algorithm_ix in range(n_algorithms):
                values = policies[algorithm_ix].predict_values(
                    obs_as_tensor(
                        SLICE_TENSOR(new_obs, algorithm_ix),
                        device
                    )
                )
                rollout_buffer_list[algorithm_ix].compute_returns_and_advantage(
                    last_values = values,
                    dones = SLICE_LIST(dones, algorithm_ix),
                )

        #callback.on_rollout_end()

        return True
 
    def learn(
        self,
        total_timesteps,
        callback = None,
        log_interval = 1,
        eval_env = None,
        eval_freq = -1,
        n_eval_episodes = 5,
        tb_log_name = "SeveralAlgorithms",
        eval_log_path = None,
        reset_num_timesteps = True
    ):
        iteration = 0

        self._setup_learn()

        total_timesteps_consistency_check = None


        n_algorithms = len(self.algorithmlist)

        modified_total_timesteps = [None] * n_algorithms

        for algorithm_ix in range(n_algorithms):
            modified_total_timesteps[algorithm_ix], _ = self.algorithmlist[algorithm_ix]._setup_learn(
                total_timesteps, eval_env, None, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
            )

        total_timesteps = VALUE_OF_LIST_WHICH_IS_ALL_THE_SAME_VALUE(modified_total_timesteps)

        n_steps = VALUE_OF_LIST_WHICH_IS_ALL_THE_SAME_VALUE(
            [algorithm.n_steps for algorithm in self.algorithmlist]
        )

        while VALUE_OF_LIST_WHICH_IS_ALL_THE_SAME_VALUE([algorithm.num_timesteps for algorithm in self.algorithmlist]) < total_timesteps:
            continue_training = self.collect_rollouts(
                self.env,
                callback,
                [algorithm.rollout_buffer for algorithm in self.algorithmlist],
                n_rollout_steps = n_steps
            )

            if continue_training is False:
                break

            iteration += 1

            print("Iteration %d, number of timesteps %s" % (iteration, str([algorithm.num_timesteps for algorithm in self.algorithmlist])))

            for idx in range(n_algorithms):
                print("Algorithm %d/%d stats:" % (idx, n_algorithms))
                print("    Reward mean: %f" % safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer[idx]]))
                print("    Length mean: %f" % safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer[idx]]))
                self.ep_info_buffer[idx][:] = []

            for idx in range(n_algorithms):
                print("Training %d/%d." % (idx, n_algorithms))
                self.algorithmlist[idx].train()

        return self
