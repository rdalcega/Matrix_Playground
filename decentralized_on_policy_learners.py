import torch
import numpy as np
import stable_baselines3
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.utils import obs_as_tensor, safe_mean

import gym

def _slice(iterable, i):
    return iterable[i: i+1]

def is_constant_iterable(iterable):
    constant = iterable[0]
    for item in iterable:
        if item != constant:
            return False
    return True

class DecentralizedOnPolicyLearners:

    def __init__(
        self,
        learners,
        env
    ):

        self.learners = learners
        self.num_learners = len(self.learners)
        self.env = env

    def policies(self, ID=None):
        if ID is None:
            return [learner.policy
                for learner in self.learners]
        else:
            return self.learners[ID].policy

    def predict(
        self,
        obs,
        state=None,
        episode_start=None,
        deterministic=False
    ):
        # an important note on the deterministic
        # parameter is that, as far as I can tell,
        # these OnPolicy Algorithms don't do
        # epsilon-greedy exploration. Instead,
        # they output a probability distribution
        # over actions and choose an action
        # according to that probability distribution.

        # except if deterministic is set to True,
        # the model will choose the action with
        # the highest probability.

        # In order to encourage exploration,
        # deterministic should always be set to
        # False over the course of training.

        if state is not None or episode_start is not None:
            raise NotImplemented
        
        policies = self.policies()

        def _predict(ID):
            return policies[ID].predict(
                _slice(obs, ID), deterministic=deterministic
            )
        
        predictions = [
            _predict(ID) for ID in range(self.num_learners)
        ]
        # a prediction is not just an action, but an action
        # along with a prediction of what the next state
        # will be. 
        actions = [
            prediction[0] for prediction in predictions
        ]
        # these are apparently used in recurrent policies
        # we're not using them now but might as well
        # allow for the versatility
        states = [
            prediction[1] for prediction in predictions
        ]
        return actions, states

    def collect_rollouts(
        self,
        env,
        callback,
        rollout_buffers,
        n_steps
    ):
        on_rollout_start = getattr(callback, "_on_rollout_start", None)
        if on_rollout_start is not None:
            on_rollout_start()

        assert self._last_obs is not None, "No previous observation was provided"
        # policies are kept fixed while we
        # collect rollouts, so we might
        # as well get them now
        policies = self.policies()
        # a learners "device" is just the 
        # "device on which the code should be
        # run. Setting it to auto, the code
        # will be run on the GPU if possible."
        # it's not entirely clear to me at this
        # point why it's important that
        # all devices be the same.
        assert is_constant_iterable(
            [learner.device for learner in self.learners]
        ), "all learners must use the same device"
        device = self.learners[0].device
        # set_training_mode does nothing other
        # than what you'd expect it to do. Policies
        # inherit from torch Neural Network Modules
        # which have a parameter called, you guessed
        # it: training! The following sets them
        # all to False. As harmless as this seems,
        # I'm not yet sure what kind of bug it
        # prevents
        for policy in policies:
            policy.set_training_mode(False)
        # for on policy algorithms, experience
        # should be discarded after each policy
        # update. The rollout_buffer "corresponds to...
        # transitions collected using the current policy."
        # Since right before calling collect_rollouts
        # we trained the policies using "train",
        # the experience that's in the rollout
        # buffer is now technically "off policy".
        # we ust wipe it. Or, to be more kind,
        # reset it.
        for rollout_buffer in rollout_buffers:
            rollout_buffer.reset()
        # for each rollout, we start having
        # taken no steps and take n_steps.
        # to keep track we use
        steps = 0
        while steps < n_steps:
            # and we do whatever's in here
            # for n_steps.

            # the way to think of each element
            # in the rollout is as a tuple
            # (observation, action, reward,
            #       start_of_episode, value, log_probability)
            # where start_of_episode is True if this
            # observation occured at the beginning
            # of an episode, value is the estimated value
            # following the current policy
            # of the state arrived after taking action,
            # and log_prob is the log_probability of the
            # action following the current policy (self information?)

            # this loop just collects all this information
            # for each learner and adds it to its corresponding
            # rollouts. that's the idea. everything else
            # is technical.

            actions = [None]*self.num_learners
            values = [None]*self.num_learners
            log_probabilities = [None]*self.num_learners

            for ID in range(self.num_learners):
                # torch.no_grad() is a context
                # manager that disables gradient
                # calculation. As best as I can
                # tell at the moment, this is
                # a way of ensure that no
                # training is happening as
                # we compute actions using
                # the neural nets underlying
                # the policies.

                # with is an interesting construct
                # I hadn't run across before. Basically,
                # it takes in a "Context Manager" object
                # and calls its __enter__ method. When the
                # with statement is over, it calls its
                # __exit__ method. I presume under the
                # hood torch.no_grad() just sets some
                # parameter the indicates whether gradient
                # calculations are possible to False.
                with torch.no_grad():
                    observation = obs_as_tensor(_slice(self._last_obs, ID), device)
                    action, value, log_probability = policies[ID](observation)
                # .cpu method of a tensor
                # returns a copy of the tensor
                # that's stored in cpu.
                # and .numpy method converts
                # it to a numpy array. We have to
                # use cpu first because the return
                # of the output of numpy and 
                # the object that its called on
                # share the same storage
                action = action.cpu().numpy()

                actions[ID] = action
                values[ID] = value
                log_probabilities[ID] = log_probability

            with torch.no_grad():
                actions = np.array(actions)
                values = torch.tensor(values)
                log_probabilities = torch.tensor(log_probabilities)

            # technically, actions list is a list of lists with
            # one element. So we flatten it before inputting
            # it into env.step
            flattened_actions = [action[0] for action in actions]

            observations, rewards, dones, infos = env.step(flattened_actions)

            on_step = getattr(callback, "_on_step", None)
            if on_step is not None:
                on_step(
                    self,
                    steps,
                    flattened_actions,
                    observations,
                    rewards,
                    dones,
                    infos)

            # the following snippet is commented 
            # out because matrix_playground always
            # returns empty infos, so the condition
            # of the if will never be satisfied.
            """
            for ID, done in enumerate(dones):
                if (
                    done
                    and infos[ID].get("terinal_observation") is not None
                    and infos[ID].get("TimeLimit.truncated", False) 
                ):
                    terminal_obs = self.learners[ID].policy.obs_to_tensor(infos[ID]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.learners[ID].policy.predict_values(terminal_obs)[0]
                    rewards[ID] = self.learns[ID].gamma * terminal_value
            """

            # at this point we have all the
            # information we need to extend
            # the rollout buffers, so we do!
            for ID in range(self.num_learners):
                observation = _slice(self._last_obs, ID)
                action = _slice(actions, ID)
                reward = _slice(rewards, ID)
                start_of_episode = _slice(self._last_dones, ID)
                value = _slice(values, ID)
                log_probability = _slice(log_probabilities, ID)

                rollout_buffers[ID].add(
                    observation,
                    action,
                    reward,
                    start_of_episode,
                    value,
                    log_probability
                )

            # finally some book keeping
            self._last_obs = observations
            self._last_dones = dones
            for learner in self.learners:
                learner.num_timesteps += 1
            steps += 1

        # now the loop is over.
        # we've collected all the rollouts
        # for the upcoming update.
        with torch.no_grad():
            for ID in range(self.num_learners):
                value = policies[ID].predict_values(
                    obs_as_tensor(
                        _slice(observations, ID),
                        device
                    )
                )
                done = _slice(dones, ID)
                rollout_buffers[ID].compute_returns_and_advantage(
                    last_values=value,
                    dones=done
                )

        on_rollout_end = getattr(callback, "_on_rollout_end", None)
        if on_rollout_end is not None:
            #on_rollout_end(self.learners, self.learners[0].num_timesteps)
            on_rollout_end()

        return True


    def _setup_learn(self):
        # takes care of a few preliminaries
        # reset the environment and collect
        # the initial observations
        self._last_obs = self.env.reset()
        # unclear what this is for now
        self._last_dones = np.full(self.num_learners, True, dtype=bool)
        # unclear what this does
        self._ep_info_buffer = [
            [] for j in range(self.num_learners)
        ]

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1,
        eval_env=None,
        eval_freq=-1,
        n_eval_episodes=5,
        tb_log_name="SeveralAlgorithms",
        log_path=None,
        reset_num_timesteps=True
    ):

        """
        :param total_timesteps: The total number
        of samples (env steps) to train on

        :param eval_env: environment to use
        for evaluation if different from
        environment to use for training.

        :param callback: object with a few
        methods that are called at different
        points in learning

        :param eval_freq: how many steps
        between evaluation

        :param n_eval_episodes: how many episodes
        to play per evaluation

        :param log_path: path to a folder where
        the evaluations will be saved

        :param reset_num_timesteps: whether to
        reset the num_timesteps attribute

        :param tb_log_name: the name of the run
        for tensorboard log
        """

        update_number = 0
        self._setup_learn()
        total_timesteps_consistency_check = None

        # naturally, we want to set modified_total_timesteps
        # to total_timesteps for every learner. However,
        # the parameter reset_num_timesteps of the internal
        # method BaseAlgorithm._setup_learn gives the option
        # to not reset the internal parameter num_timesteps
        # with each new call to _setup_learn. In this case,
        # modified_total_timesteps would be
        # timesteps at the end of last learn + total_timesteps

        # note also that we don't pass callback to _setup_learn.
        start_times = [None]*self.num_learners
        stop_times = [None]*self.num_learners
        for ID, learner in enumerate(self.learners):
            stop_times[ID], _ = learner._setup_learn(
                    total_timesteps=total_timesteps,
                    eval_env=eval_env,
                    callback=None,
                    eval_freq=eval_freq,
                    n_eval_episodes=n_eval_episodes,
                    log_path=log_path,
                    reset_num_timesteps=reset_num_timesteps,
                    tb_log_name=tb_log_name
            )
            start_times[ID] = learner.num_timesteps
        # even though stop_times is not necessarily
        # equal to total_timesteps, it should be constant.
        # otherwise some learners have been learning for longer
        # than others.
        assert is_constant_iterable(
            stop_times
        ), "modified_total_timesteps should be constant"
        assert is_constant_iterable(
            start_times
        ), "start_times should be constant"
        # if so, set stop_time to the first value
        # of the list
        stop_time = stop_times[0]
        start_time = start_times[0]

        # so far we're assuming learners are OnPolicy.
        # the OnPolicy class in stable_baselines
        # has a parameter called n_steps, which is
        # "the number of steps to run for each
        # environment per update." I think the
        # right way to think about this is that
        # n_steps is the number of steps during
        # which the policy is kept fixed.

        # for learning to work, they must all share
        # the same value of n_steps.
        assert is_constant_iterable(
            [learner.n_steps for learner in self.learners]
        ), "all learners must have the same n_steps"
        # if so, set n_steps to the first of them
        n_steps = self.learners[0].n_steps

        on_training_start = getattr(callback, "_on_training_start", None)
        if on_training_start is not None:
            on_training_start(self, start_time, stop_time)

        # after these assertions, we can be sure
        # that self.learners[i].num_timesteps is the
        # same for all i over the course of training
        while self.learners[0].num_timesteps < stop_time:
            print("starting update number", update_number, "at timestep", self.learners[0].num_timesteps)
            # call this the "update" loop
            # learners play together for n_step
            # rounds and learn from their experiences
            # before they play again.

            # the following line makes the
            # learners play together for n_steps
            # steps of self.env and remember their
            # experiences in their rollout buffers
            continue_training = self.collect_rollouts(
                env=self.env,
                callback=callback,
                rollout_buffers=[learner.rollout_buffer for learner in self.learners],
                n_steps=n_steps
            )
            # for any onPolicy algorithm, the .train
            # method consumes current rollout data
            # updates the policy parameters.

            # it might be surprising to you that it
            # doesn't take any parameters. This is just
            # because what it needs is stored in its
            # attributes
            for ID, learner in enumerate(self.learners):
                learner.train()
        
            # do we really need an update_number
            update_number += 1

        return self

