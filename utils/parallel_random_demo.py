# adaptation of random_demo but for
# parallel_env. Otherwise its relationship
# to the render method is glitchy.
def random_demo(env, render=True, episodes=5):
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