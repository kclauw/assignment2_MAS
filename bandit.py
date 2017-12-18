import numpy as np
import random
import strategy
import random
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