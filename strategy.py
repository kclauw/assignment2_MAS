import bandit
import numpy as np
import random as r
import math




def EpsilonGreedy(N,Q,epsilon):
        rand = np.random.random()
        if epsilon>=rand:
            return np.random.randint(N)
        else:
            return np.argmax(Q)

def random(bandit,epsilon):
        return np.random.randint(bandit.n)

def softmax(bandit,value):
    n = len(bandit.Q_values)
    e = np.zeros(n)
    e = np.exp(bandit.Q_values / value)
    boltzman = e / np.sum(e)
    rand = r.random()
    n = len(boltzman)
    total_boltzman = 0
    for i in range(0,n):
        total_boltzman += boltzman[i]
        if total_boltzman >= rand:
            return i
    return n - 1

def softmax_T(bandit,value,t):
    new_t = 4*(1000-(t+1))/1000
    n = len(bandit.Q_values)
    e = np.exp(bandit.Q_values/new_t)
    boltzman = e / np.sum(e)
    rand = r.random()
    n = len(boltzman)
    total_boltzman = 0
    for i in range(0,n):
        total_boltzman += boltzman[i]
        if total_boltzman >= rand:
            return i
    return n - 1