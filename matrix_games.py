# here we define functions
# that take in some parameters
# and output a game.

# one way to think about
# these functions is that
# they give parametrized
# families of games

def Coordination(bias=1):
    # :param bias: the reward
    # obtained from mutual cooperation
    # as a proportion of the
    # reward obtained from
    # mutual defection

    assert bias >= 1, "mutual cooperation must be at least as good as mutual defection"

    optimal = 1
    suboptimal = 1/bias

    return { ("C", "C"): (optimal, optimal),
             ("C", "D"): (0, 0),
             ("D", "C"): (0, 0),
             ("D", "D"): (suboptimal, suboptimal),
             "label": "COORD"
            }

def PrisonersDilemma(greed=4/3, fear=1, inefficiency=1/3):
    # :param greed: the reward obtained
    # by defecting if your opponent
    # cooperation as a proportion of the
    # reward obtained from mutual cooperation

    # :param fear: the loss of reward
    # if I cooperate but my opponent
    # defects as a proportion
    # of my reward with mutual defection.
    # 0 means no fear
    # 1 means everything is at stake

    # :param inefficiency: the reward obtained
    # from mutual defection as a proportion
    # of the reward obtained from mutual
    # cooperation.

    assert greed > 1, "there must be greed in prisoner's dilemma"
    assert 0 < fear, "there must be fear in prisoner's dilemma"
    assert fear <= 1, "fear must be <= 1"
    assert 0 < inefficiency, "mutual defection must have some reward in prisoners dilemma"
    assert inefficiency < 1, "mutual defection must be worse than mutual cooperation in prisoners dilemma"

    reward = 1/greed
    temptation = 1
    punishment = reward*inefficiency
    sucker = punishment*(1 - fear)

    return {
        ("C", "C"): (reward, reward),
        ("C", "D"): (sucker, temptation),
        ("D", "C"): (temptation, sucker),
        ("D", "D"): (punishment, punishment),
        "label": "PRDIL"
    }

def BattleOfTheSexes(bias=10):
    # :param bias: the reward of dominating
    # when the opponent submits as a proportion
    # of the reward of submitting when the
    # opponent dominates

    assert bias > 1, "domination must be better than submission in battle of the sexes"

    dominate = 1
    submit = dominate/bias

    return {
        ("C", "C"): (0, 0),
        ("C", "D"): (submit, dominate),
        ("D", "C"): (dominate, submit),
        ("D", "D"): (0, 0),
        "label": "BAOSE"
    }

def StagHunt(advantage=4/3, fear=1, inefficiency=1/4):
    # :param advantage: the reward of mutual
    # cooperation as a proportion of the reward
    # of defecting when the opponent cooperates

    # :param fear: the loss of reward
    # if I cooperate but my opponent
    # defects as a proportion
    # of my reward with mutual defection.
    # 0 means no fear
    # 1 means everything is at stake

    # :param inefficiency: the reward obtained
    # from mutual defection as a proportion
    # of the reward obtained from mutual
    # cooperation.

    assert 1 < advantage, "cooperation must give the highest reward in stag hunt"
    assert 0 < fear, "there must be fear in stag hunt"
    assert fear <= 1, "fear must be <= 1"
    assert 0 < inefficiency, "mutual defection must have some reward in stag hunt"
    assert inefficiency < 1, "mutual defection must be worse than mutual cooperation in stag hunt"

    reward = 1
    temptation = reward/advantage
    punishment = reward*inefficiency
    sucker = punishment*(1-fear)

    assert punishment < temptation, "being punished must be worse than being tempted in stag hunt"

    return {
        ("C", "C"): (reward, reward),
        ("C", "D"): (sucker, temptation),
        ("D", "C"): (temptation, sucker),
        ("D", "D"): (punishment, punishment),
        "label": "STAGH"
    }

def Chicken(greed=4/3, inefficiency=1/3):
    # :param greed: the reward obtained
    # by defecting if your opponent
    # cooperation as a proportion of the
    # reward obtained from mutual cooperation

     # :param inefficiency: the reward obtained
    # from being a sucker as a proportion
    # of the reward obtained from mutual
    # cooperation.

    assert greed > 1, "there must be greed in chicken"
    assert inefficiency < 1, "being a sucker must be worse than mutual cooperation"

    temptation = 1
    reward = temptation/greed
    sucker = reward*inefficiency
    punishment = 0

    return {
        ("C", "C"): (reward, reward),
        ("C", "D"): (sucker, temptation),
        ("D", "C"): (temptation, sucker),
        ("D", "D"): (punishment, punishment),
        "label": "CHICK"
    }