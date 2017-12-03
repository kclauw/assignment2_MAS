from pymc import rbeta
import strategy
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import math
import plot
from pymc import rbeta
import abc
from strategy import UCB1,UCB2,Softmax,EpsilonGreedy,Random,EnGreedy,ThompsonSampling,UCB1Chernoff


import seaborn as sns


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
        r = rand()
        return r < self.p[i]
    
    def __len__(self):
        return len(self.p)




class Experiment(object):

    def __init__(self, bandits,strategy):
        
        self.bandits = bandits
        self.n_bandits = len(self.bandits)
        self.trials = np.zeros(self.n_bandits)
        self.rewards = np.zeros(self.n_bandits)
        self.Q = np.ones(self.n_bandits)
        self.N = 0
        self.optimal_probability = np.max(self.bandits.p)
        self.strategy = strategy
        self.best_action = -1
        self.r = np.zeros(self.n_bandits)
        self.extra_times = 0
        self.success = np.zeros(self.n_bandits)
        self.failure = np.zeros(self.n_bandits)

    def update(self,action,reward):
        self.N += 1
        self.trials[action] += 1
        self.Q[action] += (1/float(self.trials[action])) * (reward -self.Q[action])
        self.rewards[action] += reward


    def sample_bandits(self, n=1 ):
        total_regret = np.zeros(n)
        total_reward = np.zeros(n)
        for k in range(n):
 
            #Action selection
            action = self.strategy.execute(self,k)


            #Retrieve reward
            reward = self.bandits.pull(action)

            #Update values
            self.update(action,reward)

            #Update total reward and regret
            regret = self.optimal_probability - self.bandits.p[action]
            total_regret[k] = regret
            total_reward[k] = reward

            #Update success failure distributions -> required by Thompson Sampling
            if reward == 1:
                self.success[action]+=1
            else:
                self.failure[action]+=1




        return np.cumsum(total_reward),np.cumsum(total_regret)


class OptimisticInitializationStrategy(object):
    
    def __init__(self, bandits):
        
        self.bandits = bandits
        self.n_bandits = len(self.bandits)
        self.wins = np.zeros(self.n_bandits)
        self.trials = np.zeros(self.n_bandits)
        self.Q = np.ones(self.n_bandits)
        self.N = 0
        self.strategy = strategy
        self.optimal_probability = np.max(self.bandits.p)
 
    def sample_bandits(self, n=1 ):
        
        total_regret = np.zeros(n)
        total_reward = np.zeros(n)

        for k in range(n):

            #Select random action
            action = np.argmax(self.Q)

            reward = self.bandits.pull(action)
            #Update Q-values
            self.trials[action] += 1 # Number of actions
            self.Q[action] += (1/float(self.trials[action])) * (reward - self.Q[action])
            regret = self.optimal_probability - self.bandits.p[action]
            total_regret[k] = regret
            total_reward[k] = reward
        return np.cumsum(total_reward),np.cumsum(total_regret)



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
        self.optimal_probability = np.max(self.bandits.p)

    def sample_bandits(self, n=1 ):
        
        bb_score = np.zeros(n)
        choices = np.zeros(n)
        means = np.zeros(n)
        count = np.zeros(n)
        total_regret = []
        total_reward = []

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
            choices[k] = action
            best = np.max(self.Q)
            self.Q[action] += (1/self.trials[action]) * (result - self.Q[action])
            regret = self.optimal_probability - self.bandits.p[action]
            total_regret.append(regret)
            total_reward.append(result)

        self.bb_score = np.r_[self.bb_score, bb_score]
        self.choices = np.r_[self.choices, choices]
        return np.cumsum(total_reward),np.cumsum(total_regret)



def runExperimentsAll(max_pulls,real_distribution):
    bandits = Bandits(real_distribution)
    total = np.zeros(max_pulls)
    labels = ["UCB2","UCB1","Softmax","Epsilon","Random","EnGreedy","Thompson","Chernoff"]
    strategies = [UCB2(0.5),UCB1(),Softmax(10),EpsilonGreedy(0.2),Random(),EnGreedy(2,0.1),ThompsonSampling(1,0.1),UCB1Chernoff()]
    rewards = []
    regret = []
    for strat in strategies:
        rew,reg = Experiment(bandits,strat).sample_bandits(max_pulls)
        rewards.append(rew)
        regret.append(reg)

    #Plot the rewards for each strategy
    for i,r in enumerate(rewards):
        plt.plot(r,label=labels[i])
        plt.xlabel("Pulls")
        plt.ylabel("Reward")
        plt.legend(loc='best', numpoints=1, fancybox=True)

    plt.show()

    #Plot the regret for each strategy
    for i,re in enumerate(regret):
        plt.plot(re,label=labels[i])
        plt.xlabel("Pulls")
        plt.ylabel("Regret")
        plt.legend(loc='best', numpoints=1, fancybox=True)
    plt.show()



def runExperimentsExtra(max_pulls,real_distribution):
    bandits = Bandits(real_distribution)
    total = np.zeros(max_pulls)
    labels = ["UCB2","UCB1","EnGreedy","Thompson","Chernoff"]
    strategies = [UCB2(0.5),UCB1(),EnGreedy(2,0.1),ThompsonSampling(1,0.1),UCB1Chernoff()]
    rewards = []
    regret = []
    for strat in strategies:
        rew,reg = Experiment(bandits,strat).sample_bandits(max_pulls)
        rewards.append(rew)
        regret.append(reg)

    #Plot the rewards for each strategy
    for i,r in enumerate(rewards):
        plt.plot(r,label=labels[i])
        plt.xlabel("Pulls")
        plt.ylabel("Reward")
        plt.legend(loc='best', numpoints=1, fancybox=True)

    plt.show()

    #Plot the regret for each strategy
    for i,re in enumerate(regret):
        plt.plot(re,label=labels[i])
        plt.xlabel("Pulls")
        plt.ylabel("Regret")
        plt.legend(loc='best', numpoints=1, fancybox=True)
    plt.show()
    




if __name__ == "__main__":
    max_pulls = 10000
    real_distribution = np.array([0.15,0.2,0.1,0.5])
    #runExperimentsAll(max_pulls, real_distribution)
    runExperimentsExtra(max_pulls, real_distribution)








