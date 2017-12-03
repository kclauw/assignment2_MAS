import abc,six
import numpy as np
import math
from pymc import rbeta



#ABC now works in python 2 and 3 
@six.add_metaclass(abc.ABCMeta)
class SelectionStrategy():
    @abc.abstractmethod
    def execute(self):
        pass


#Tuning bandit algorithms in stochasticenvironments
#http://certis.enpc.fr/~audibert/ucb_alt.pdf
class UCBV(SelectionStrategy):

    def __init__(self,c):
        self.c = c

    def execute(self,experiment,k):
        #Play each action once
        for a in range(experiment.n_bandits):
            if experiment.trials[a] == 0:
                return a

        confidence_interval = experiment.Q + np.square(np.divide(2*np.log(k+1)* experiment.variance,experiment.trials)) + np.divide(self.c*3*np.log(k),experiment.trials)
        return np.argmax(confidence_interval) 


#The KL-UCB Algorithm for Bounded Stochastic Bandits and Beyond
#https://arxiv.org/pdf/1102.2490v5.pdf
class KLUCB(SelectionStrategy):

    def __init__(self,c):
        self.c = c

    def execute(self,experiment,k):
        #Play each action once
        for a in range(experiment.n_bandits):
            if experiment.trials[a] == 0:
                return a

        confidence_interval = experiment.Q + (np.log(k+1) + (self.c * np.log(np.log(k+1))))

        return np.argmax(confidence_interval)

class UCB1(SelectionStrategy):
    def execute(self,experiment,k):
        #Play each action once
        for a in range(experiment.n_bandits):
            if experiment.trials[a] == 0:
                return a

        confidence_interval = experiment.Q + np.sqrt(2 * np.log(k+1) / experiment.trials)
        return np.argmax(confidence_interval)
class UCB1Chernoff(SelectionStrategy):
    def execute(self,experiment,k):

        confidence_interval = np.zeros(experiment.n_bandits)

        #Play each action once
        for a in range(experiment.n_bandits):
            if experiment.trials[a] == 0:
                experiment.Q[a] = 1
                return a
            delta = math.sqrt(1/k+1)
            numerator1 = 2 * experiment.Q[a] * math.log(delta)
            numerator2 = 2* math.log(delta)
            confidence_interval[a] = experiment.Q[a] + math.sqrt(numerator1/experiment.trials[a]) + (numerator2/experiment.trials[a])
        return np.argmax(confidence_interval)


class UCB2(SelectionStrategy): 

    def __init__(self,alpha):
        self.alpha = alpha


    def t(self,r):
        return  int(math.ceil((1 + self.alpha) ** r))

    def execute(self,experiment,k):

        #Play each action once
        for a in range(experiment.n_bandits):
            if experiment.trials[a] == 0:
                return a

        if experiment.extra_times > k:
            return experiment.best_action

        confidence_interval = np.zeros(experiment.n_bandits)
        for a in range(experiment.n_bandits):
            t = self.t(experiment.r[a])
            extra = math.sqrt((1. + self.alpha) * math.log((math.e * float(k+1)) / t) / (2 * t))
            confidence_interval[a] = experiment.Q[a] + extra

        best_action = np.argmax(confidence_interval)
        experiment.extra_times += max(1, self.t(experiment.r[best_action] + 1) - self.t(experiment.r[best_action]))
        experiment.best_arm = best_action
        experiment.r[best_action] += 1
           
        return best_action

class EnGreedy(SelectionStrategy): 

    def __init__(self,c,d):
        self.c = c
        self.d = d

    def GenerateE(self,k,n):
        return min(1, np.divide((self.c * n),(np.power(self.d,2)*k)))

    def execute(self,experiment,k):
        if self.GenerateE(k,experiment.n_bandits) >=np.random.random():
            return np.random.randint(experiment.n_bandits)
        else:
            return np.argmax(experiment.Q)
        return best_action


class ThompsonSampling(SelectionStrategy): 

    def __init__(self,alpha,beta):
        self.alpha = alpha
        self.beta = beta

    def execute(self,experiment,k):
        return np.argmax(rbeta(experiment.success + self.alpha,experiment.failure + self.beta))
            


     

class Softmax(SelectionStrategy):

    def __init__(self,t):
        self.t = t

    def execute(self,experiment,k):
        e = np.exp(experiment.Q/self.t)
        boltzman_distribution = e / np.sum(e)
        accumulated_probability = 0
        for i in range(0,len(boltzman_distribution)):
            accumulated_probability += boltzman_distribution[i]
            if accumulated_probability >= np.random.rand():
                return i
        return len(boltzman_distribution)-1 # only difference



class EpsilonGreedy(SelectionStrategy):

    def __init__(self,e):
        self.e = e

    def execute(self,experiment,k):
        if self.e >=np.random.random():
            return np.random.randint(experiment.n_bandits)
        else:
            return np.argmax(experiment.Q)


class Random(SelectionStrategy):

    def execute(self,experiment,k):
        return np.random.randint(experiment.n_bandits)
