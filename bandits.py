from pymc import rbeta
import strategy
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import plot
from pymc import rbeta
from strategy import EpsilonGreedy

rand = np.random.rand

class Bandits(object):
    """
    This class represents N bandits machines.
    parameters:
        p_array: a (n,) Numpy array of probabilities >0, <1.
    methods:
        pull( i ): return the results, 0 or 1, of pulling 
                   the ith bandit.
    """
    def __init__(self, p_array):
        self.p = p_array
        self.optimal = np.argmax(p_array)
        
    def pull( self, i ):
        #i is which arm to pull
        return rand() < self.p[i]
    
    def __len__(self):
        return len(self.p)


class Experiment(object):
    """
    Implements a online, learning strategy to solve
    the Multi-Armed Bandit problem.
    
    parameters:
        bandits: a Bandit class with .pull method
    
    methods:
        sample_bandits(n): sample and train on n pulls.
    attributes:
        N: the cumulative number of samples
        choices: the historical choices as a (N,) array
        bb_score: the historical score as a (N,) array
    """
    
    def __init__(self, bandits):
        
        self.bandits = bandits
        n_bandits = len( self.bandits )
        self.wins = np.zeros( n_bandits )
        self.trials = np.zeros(n_bandits )
        self.Q = np.zeros(n_bandits)
        self.N = 0
        self.choices = []
        self.bb_score = []
        self.strategy = strategy

    def sample_bandits( self, n=1 ):
        
        bb_score = np.zeros( n )
        choices = np.zeros( n )
        
        for k in range(n):
            #Pick arm according to selection strategy
            #action = np.argmax( rbeta( 1 + self.wins, 1 + self.trials - self.wins) )

            action = EpsilonGreedy(self.Q,self.N,0.1)
            print action
            #sample the chosen bandit
            result = self.bandits.pull(action)
            
            #update priors and score
            self.wins[action] += result
            self.trials[action] += 1
            bb_score[k] = result 
            self.N += 1
            choices[k] = action
            self.Q[action] += (1/self.trials[action]) * (result - self.Q[action])
    
        self.bb_score = np.r_[ self.bb_score, bb_score ]
        self.choices = np.r_[ self.choices, choices ]
        return choices,bb_score    


class BayesianStrategy( object ):
    """
    Implements a online, learning strategy to solve
    the Multi-Armed Bandit problem.
    
    parameters:
        bandits: a Bandit class with .pull method
    
    methods:
        sample_bandits(n): sample and train on n pulls.
    attributes:
        N: the cumulative number of samples
        choices: the historical choices as a (N,) array
        bb_score: the historical score as a (N,) array
    """
    
    def __init__(self, bandits):
        
        self.bandits = bandits
        n_bandits = len(self.bandits)
        self.wins = np.zeros( n_bandits )
        self.trials = np.zeros(n_bandits )
        self.N = 0
        self.Q = np.zeros(n_bandits)
        self.choices = []
        self.bb_score = []
        self.count = np.zeros(n_bandits)

    def sample_bandits( self, n=1 ):
        
        bb_score = np.zeros(n)
        choices = np.zeros(n)
        means = np.zeros(n)
        count = np.zeros(n)
        cummultative_regret = []
        cummultative_reward = []

        for k in range(n):
            #sample from the bandits's priors, and select the largest sample
            action = np.argmax( rbeta( 1 + self.wins, 1 + self.trials - self.wins) )
            
            #sample the chosen bandit
            result = self.bandits.pull(action)
            
            #update priors and score
            self.wins[action] += result
            self.trials[action] += 1 # Number of actions
            bb_score[ k ] = result 
            self.N += 1
            choices[ k ] = action
            best = np.max(self.Q)
            self.Q[action] += (1/self.trials[action]) * (result - self.Q[action])
            regret = best - result
            cummultative_regret.append(action)
            cummultative_reward.append(regret)

        self.bb_score = np.r_[ self.bb_score, bb_score ]
        self.choices = np.r_[ self.choices, choices ]

        return cummultative_reward,cummultative_regret


def regret(probabilities,choices):
    w_opt = np.max(probabilities)
    print choices
    print probabilities[choices]
    return (w_opt - probabilities[choices]).cumsum()


if __name__ == "__main__":
    print "test"
    real_distribution = np.array([0.15,0.2,0.1,0.5])
    bandits = Bandits(real_distribution)
    strat = Experiment(bandits)
    strat.sample_bandits(10)
    _regret = regret(real_distribution,strat.choices)
    print _regret




"""

    total_experiments = 1
    number_pulls = 5
    total_bandits = 3
    
    avgCumReward = np.zeros(number_pulls)
    avgCumRegret = np.zeros(total_bandits)


    distribution =  np.random.rand(total_bandits)
 


    for s in range(0,total_experiments):
        distribution = [0.85,0.60,0.75]
        bandits = Bandits(distribution)
        count,reward = BayesianStrategy(bandits).sample_bandits(number_pulls)
        print count
        print reward
        avgCumReward += reward
        avgCount = count
        _regret = regret(distribution, strat.choices)
 

    plt.show()
    sns.set(style="darkgrid")
    plt.legend() 
    plt.plot(avgCumReward/np.float(total_experiments),label="eps = 0.0")
    plt.show()
    plt.plot(avgCumRegret/np.float(total_experiments),label="eps = 0.1") 
    plt.show()
"""