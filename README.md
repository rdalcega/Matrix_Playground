# Matrix_Playground

Entry point is play.py. Play with the parameters at the top of the file and run it in the terminal after installing all the necessary packages.

Be cautious when repeatedly running this code. It is storing a lot of information with every run in a folder called logs. For now, if you don't want that information, just delete the folder. Soon I'll implement a flag that allows to enable and disable storage.

decentralized_on_policy_learners.py is basically nothing other than a version of several_algorithms.py but with different names and with some lines (that I think only applied in the continuous setting, removed). Some code in decentralized_on_policy_learners.py, like the predict function, are still not used anywhere. consider removing them.

Everything other than callbacks.py is full of comments that try to explain what's going on.

TODO:

    add plotting functionality to log_progress callback.
    clean callback.py and references to the callback in decentralized_on_policy_learners.py
