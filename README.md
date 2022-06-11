
This repository doesn't have much structure at the moment.

The pettingzoo environment for the game is defined in matrix_playground.py

The decentralized learning algorithm is defined in several_algorithms.py

learn_matrix_playground.py trains the learners on matrix_playground.

matrix_games.py includes some hardcoded matrix games we're interested in.

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
 
    - play around with learn_matrix_playground.py! Set game, num_agents, memory, and horizon to your taste and see what happens.
    
    - the code is annotated with comments like "don't understand this". Figure out what those lines are doing.
    
    - really ensure that the learners in learn_matrix_playground.py are learning independently. This is unclear to me at the moment. All I can tell is
    that the learners are really learning different policies because they react differently to the same observation.
    
    - edit several_algorithms so that it works with all the relevant RL algorithms in stable baselines. At the moment if I try to train with DQN it
    bugs out.
    
    - Right now observations are reencoded on every step from the histories. This is obviously quite inefficient since only a small portion of the bits in the encoding change on any step. Fix this. Make it so that observations are only edited where they change on every time step.
    
    - figure out how to store the model at different stages during learning in order to test progress.
    
    - what's the right way to do a TODO list on github? It's not this I'm sure.
    
    - more stuff I can't think of at the moment but what's above will keep me busy for a good bit.
