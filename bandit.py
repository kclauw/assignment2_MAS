import numpy as np
import random
import strategy

class Bandit:
    def __init__(self,arms,n,func=None):
        self.arms = arms
        self.random_values = [random.gauss(0,1) for _ in range(n)];
        for i in range(0,n):
          self.random_values[i] = arms[i].get_new_random()
        self.count = np.zeros(n)
        self.Q_values = np.zeros(n)
        self.n = n
        self.execute = func

    def get_action(self,*args):
        return self.execute(*args)

    def get_Q_value(self,i):
        return self.Q_values[i]

    def get_random_value(self,i):
        return self.random_values[i]

    def get_reward(self,action):
        arm = self.arms[action];
        return self.random_values[action]



    def set_Q_value(self,action,reward):
        self.count[action] += 1
        self.Q_values[action] += (1/self.count[action]) * (reward - self.Q_values[action])