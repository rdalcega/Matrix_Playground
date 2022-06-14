# we need a way to generate random policies to default
# on when no policies are passed.
def generate_random_policy(env, agent):
    def random_policy(obs):
        return env.action_space(agent).sample()
    return random_policy

def evaluate_cooperation(env, episodes=1, for_agent="agent_0", using_policies=None):

    # "agents" attribute is defined in env.reset(), not in env.__init__()
    # so, even though it's redundant, we call reset in order
    # for the definition of using_policies to make
    # sense when it's passed as None.
    env.reset()
    # if using_policies is None, we use random policies
    if using_policies is None:
        using_policies = {
            agent: generate_random_policy(env, agent)
                   for agent in env.agents
        }
    # we run the environment for "episodes" episodes
    # using the specified policies, count
    # the instances of cooperation from "for_agent"'s
    # policies
    total_cooperations = 0
    total_defections = 0
    completed_episodes = 0
    while completed_episodes < episodes:
        observations = env.reset()
        done = False
        while not done:
            actions = {
                agent: using_policies[agent](observations[agent])
                        for agent in env.agents
            }
            observations, rewards, dones, infos = env.step(actions)
            if actions[for_agent] == 0: # we interpret 0 as "C"
                total_cooperations += 1
            else: # we interpret 1 as "D"
                total_defections += 1
            done = dones["agent_0"]
        completed_episodes += 1
    # we return the fraction of total
    # actions that were cooperations and
    # the fraction of total actions that were
    # defections
    total_actions = total_cooperations + total_defections
    fraction_of_cooperations = total_cooperations/total_actions
    fraction_of_defections = total_defections/total_actions
    return fraction_of_cooperations, fraction_of_defections