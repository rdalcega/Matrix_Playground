# Matrix_Playground

NOTE: There's something seriously wrong with matrix_playground.py. In particular, the environment pairs players up in a different way on every round. And each players observation is their history with the player they're about to play. This is a problem because two players with history {(C, C), (C, C), (C, C)} could both play {(C, C)} but transition to an observation {(D, D), (D, D), (D, D)} just because that's their history with the person they're about to play. Because of this, I think it's impossible for networks to learn in the ways we'd expect them to. This explains why the current code creates expected behavior when there are only 2 agents, but not when there are more than 2 agents. I have a fix for this that I'm working on at the moment. It consists of giving observations of agent_0 in a setting with four agents where they're about to play agent_2 the following structure:
    
    { "agent_1": {
        "opponent": False,
        -1: ["C", "C"],
        -2: ["D", "D"],
        -3: ["C", "C"] },
       "agent_2": {
        "opponent": True,
        -1: ["C", "D"],
        -2: ["D", "D"],
        -3: ["D", "C"] },
       "agent_3": {
        "opponent": False,
        -1: ["C", "C"],
        -2: ["C", "C"],
        -3: ["C", "C"] }
      }
  
  Note its only in a dictionary for our interpretability. From the perspective of the learners, all of this is encoded in a vector with binary values.
        

This repository doesn't have much structure at the moment.

The pettingzoo environment for the game is defined in matrix_playground.py

The decentralized learning algorithm is defined in several_algorithms.py

learn_matrix_playground.py trains the learners on matrix_playground.

Near the top of the file you can set

    game: a dictionary like
      { ("C", "C"): (3, 3),
        ("C", "D"): (0, 4),
        ("D", "C"): (4, 0),
        ("D", "D"): (1, 1) }
     indicating pairs of rewards corresponding to pairs of plays
     
     num_agents: the number of agents in the playground
     
     memory: the number of prior games (with their current opponent) an agent remembers
     
     horizon: the number of games in one episode of the game
     
     training_timesteps: the number of environment timesteps that the agents are trained on
     
PACKAGES: I think all you need for this to run is pettingzoo, stable_baselines3, and supersuit. I've been installing them in a virtualenv so as not to clutter my computer with packages I don't use often.
     
 
 TODOS:
 
    - play around! Try out different games just to see what happens.
    
    - the code is annotated with comments like "don't understand this". figure out what those lines are doing.
    
    - really ensure that the learners in learn_matrix_playground.py are learning independently. This is unclear to me at the moment. All I can tell is
    that the learners are really learning different policies because they react differently to the same observation.
    
    - edit several_algorithms so that it works with all the relevant RL algorithms in stable baselines. At the moment if I try to train with DQN it
    bugs out.
    
    - figure out how to store the model at different stages during learning in order to test progress.
    
    - what's the right way to do a TODO list on github? It's not this I'm sure.
    
    - more stuff I can't think of at the moment but what's above will keep me busy for a good bit.
