import abc,six
import numpy as np
import math
import random
from pymc import rbeta
import scipy.stats as stats
from pymc.utils import quantiles
import argparse

#ABC now works in python 2 and 3 
@six.add_metaclass(abc.ABCMeta)
class SelectionStrategy():
    @abc.abstractmethod
    def policy(self):
        pass
"""
Must Have strategies
"""
     

class Softmax(SelectionStrategy):

    def __init__(self,t):
        self.t = t

    def policy(self,experiment,k):
        a = experiment.Q / self.t
        e = np.exp(a)
        distribution = e / np.sum(e)
        cummultative = 0.0
        for i,p in enumerate(distribution):
            cummultative += p
            if cummultative > random.random():
                return i
        return i
  
 

class EpsilonGreedy(SelectionStrategy):

    def __init__(self,e):
        self.e = e

    def policy(self,experiment,k):
        if self.e >=np.random.random():
            return np.random.randint(experiment.n_bandits)
        else:
            return np.argmax(experiment.Q)


class Greedy(SelectionStrategy):

    def policy(self,experiment,k):
        return np.argmax(experiment.Q)


class Random(SelectionStrategy):

    def policy(self,experiment,k):
        return np.random.randint(experiment.n_bandits)

"""
UCB + EnGreedy algorithms
https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf
"""

class UCB1(SelectionStrategy):
    def policy(self,experiment,k):
        #Play each action once
        for a in range(experiment.n_bandits):
            if experiment.trials[a] == 0:
                return a

        confidence_interval = experiment.Q + np.sqrt(2 * np.log(k+1) / experiment.trials)
        return np.argmax(confidence_interval)

class UCB2(SelectionStrategy): 

    def __init__(self,alpha):
        self.alpha = alpha


  
    def policy(self,experiment,k):



        #Play each action once
        for a in range(experiment.n_bandits):
            if experiment.trials[a] == 0:
                return a

        #Play each extra action
        if experiment.extra_times > 0:
            experiment.extra_times = experiment.extra_times - 1
            return experiment.best_action

        t = np.ceil(np.power(1+self.alpha,experiment.r))
        confidence_interval = experiment.Q + np.sqrt(np.divide((1. + self.alpha)*np.log(np.divide((math.e * (k+1)), t)),2*t))
        best_action = np.argmax(confidence_interval)
        t = np.ceil(np.power(1+self.alpha,experiment.r[best_action]))
        extra = np.maximum(0,(t + 1) - t)
        experiment.best_action = best_action
        experiment.extra_times = extra
        experiment.r[best_action] += 1 

        return best_action

class EnGreedy(SelectionStrategy): 

    def __init__(self,c,d):
        self.c = c
        self.d = d
      

    def GenerateE(self,k,n):
        return min(1,np.divide((self.c*n),self.d**2*(k+1)))
  
    def policy(self,experiment,k):
        if self.GenerateE(k,experiment.n_bandits) >=np.random.random():
            return np.random.randint(experiment.n_bandits)
        else:
            return np.argmax(experiment.Q)
        return best_action


"""

An Empirical Evaluation of Thompson Sampling
#http://papers.nips.cc/paper/4321-an-empirical-evaluation-of-thompson-sampling.pdf

"""
class UCB1Chernoff(SelectionStrategy):
    def policy(self,experiment,k):

        confidence_interval = np.zeros(experiment.n_bandits)

        #Play each action once
        for a in range(experiment.n_bandits):
            if experiment.trials[a] == 0:
                return a
        delta = np.sqrt(1/(k))
        numerator1 = np.array(2 * experiment.Q * np.log(1/delta))
        numerator2 = np.array(2* math.log(1/delta))
        confidence_interval = experiment.Q + np.sqrt(numerator1/experiment.trials) + (numerator2/experiment.trials)
        return np.argmax(confidence_interval)


class ThompsonSampling(SelectionStrategy): 

    def __init__(self,alpha,beta):
        self.alpha = alpha
        self.beta = beta

    def policy(self,experiment,k):
        return np.argmax(rbeta(experiment.success + self.alpha,experiment.failure + self.beta))
            


#Minimax Optimal Strategy
#http://www.di.ens.fr/sierra/pdfscurrent/COLT09a.pdf
#https://arxiv.org/pdf/1510.00757.pdf
class MOSS(SelectionStrategy):
    def __init__(self,s):
        self.s = s

    def policy(self,experiment,k):

        #Play each action once
        for a in range(experiment.n_bandits):
            if experiment.trials[a] == 0:
                return a
        
        confidence_interval = experiment.Q + np.divide(np.max(np.log(experiment.trials/(experiment.n_bandits*self.s)),0),self.s)
        return np.argmax(confidence_interval)


#Standard EXP3 with different distribution
#http://proceedings.mlr.press/v24/seldin12a/seldin12a.pdf
class EXP3(SelectionStrategy):
    def __init__(self,gamma):
        self.gamma = gamma

    def policy(self,experiment,k):

        et = min((1./experiment.n_bandits),np.sqrt(np.log(experiment.n_bandits)/(experiment.n_bandits*(k+1))))
        distribution = (1 - experiment.n_bandits * et) * (experiment.weights / np.sum(experiment.weights))  + et
        
       
        #Sample from distribution
        cummultative = 0
        best_action = len(distribution)-1
        for i,p in enumerate(distribution):
            cummultative += p
            r = random.random()
            if cummultative > r:
                best_action = i
                break

        reward = experiment.bandits.pull(best_action)

        #Normalize reward
        estimated_reward = 1.0 * reward / distribution[best_action]
        
        #Update_weights
      
        experiment.weights[best_action] = experiment.weights[best_action] * np.exp((et / experiment.n_bandits) * estimated_reward)
        

        return best_action




#https://arxiv.org/pdf/1711.03591.pdf
#Efficient-UCBV: An Almost Optimal Algorithm using Variance Estimates

class UCBV(SelectionStrategy):
    """

    """
    def __init__(self,p,z):
        self.p = p

    



    def policy(self,experiment,k):

        #Play each action once 
        if k <= experiment.n_bandits:
            for a in range(len(experiment.Q)):
                if experiment.trials[a] == 0:
                    return a

        #Play the last action in Q until the end of the horizon
        if len(experiment.Q) == 1:
            return 0

        upper_confidence_interval = np.zeros(len(experiment.B))
        lower_confidence_interval = np.zeros(len(experiment.B))
        for i,arm in enumerate(experiment.B):
            upper_confidence_interval[i] = experiment.Q[arm] + np.square(np.divide(self.p*(experiment.variance[arm]+2)*np.log(experiment.z*experiment.num_pulls*experiment.e),4*experiment.trials[arm]))
            lower_confidence_interval[i] = experiment.Q[arm] - np.square(np.divide(self.p*(experiment.variance[arm]+2)*np.log(experiment.z*experiment.num_pulls*experiment.e),4*experiment.trials[arm]))    


        action = np.argmax(upper_confidence_interval)

        experiment.delete_arms = []
        for i in range(len(upper_confidence_interval)):
            if upper_confidence_interval[i] < np.max(lower_confidence_interval):
                experiment.delete_arms.append(i)


        return experiment.B[action]


