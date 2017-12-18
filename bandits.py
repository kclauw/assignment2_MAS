from pymc import rbeta
import strategy
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import math
from pymc import rbeta
import abc
from strategy import UCB1,UCB2,Softmax,EpsilonGreedy,Random,EnGreedy,ThompsonSampling,UCB1Chernoff,UCBV,Greedy,EXP3,MOSS
import random
import argparse
from bandit import Bandits
import seaborn as sns
import os
import errno





class Experiment(object):

    def __init__(self, bandits,strategy,num_pulls=1):
        
        self.bandits = bandits
        self.n_bandits = len(self.bandits)
        self.trials = np.zeros(self.n_bandits)
        self.rewards = np.zeros(self.n_bandits)
        self.Q = np.zeros(self.n_bandits)
        self.variance = np.zeros(self.n_bandits)
        self.optimal_probability = np.max(self.bandits.p)
        self.strategy = strategy
        self.r = [0 for i in range(self.n_bandits)]
        self.extra_times = 0
        self.success = np.zeros(self.n_bandits)
        self.failure = np.zeros(self.n_bandits)
        self.total_regret = []
        self.total_reward = []
        self.best_action = -1
        self.weights = [1 for i in range(self.n_bandits)]
        self.num_pulls = num_pulls
        #Parameters for UCB-V
        self.m = 0
        self.e = 1
        self.M = (1/2) * math.log((self.num_pulls/math.e),2)
        self.z = self.num_pulls / (self.n_bandits**2)
        self.n = math.log(self.z * self.num_pulls*(self.e**2)) / 2
        self.n= np.divide(math.log(self.z*self.num_pulls*(self.e**2)),2*self.e)
        self.N = self.n_bandits* self.n
        self.B = np.arange(self.n_bandits)
        self.delete_arms = []


    def setQ(self,value):
        self.Q = [value for i in range(self.n_bandits)]


    def update(self,action,reward):
        self.trials[action] += 1

        #Update mean
        self.Q[action] += (1/float(self.trials[action])) * (reward -self.Q[action])
        #Update variance
        self.variance[action] += (reward -self.Q[action]) ** 2
        self.rewards[action] += reward

       
        #Update success failure distributions -> required by Thompson Sampling
        if reward == 1:
            self.success[action]+=1
        else:
            self.failure[action]+=1
     



    def sample_bandits(self, seed):

        random.seed(seed)
        for k in range(0,self.num_pulls):


            #Action selection
            action = self.strategy.policy(self,k)

            #Retrieve reward
            reward = self.bandits.pull(action)

            #Update values  
            self.update(action,reward)





            #UCB-V removal
            #Remove the selected removable arms( After updates)
            if len(self.delete_arms) > 0 and len(self.B) > 1:
                #self.Q = np.delete(self.Q, self.delete_arms, axis=0)
                #self.variance = np.delete(self.variance, self.delete_arms, axis=0)
                #self.trials = np.delete(self.trials, self.delete_arms, axis=0)
                self.B = np.delete(self.B, self.delete_arms, axis=0)
                self.delete_arms = []
            #Reset parameters
            if k >= self.N and self.m <= self.M:
                self.e = (self.e / 2)
                self.n = np.divide(np.log(self.z*self.num_pulls*(self.e**2)),2*self.e)
                self.N = k + len(self.B)*self.n
                self.m += 1
    
            #Update total reward and regret
            regret = self.optimal_probability - self.bandits.p[action]
            self.total_regret.append(regret)
            self.total_reward.append(reward)



        return np.cumsum(self.total_reward),np.cumsum(self.total_regret)





def plot(labels,values,x,y,title,name,directory):
    plt.clf()


    for i in range(0,len(labels)):
        strat = labels[i]
        value = values[labels[i]]
        plt.plot(value,label=strat,linewidth = 1)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.legend(loc='best', numpoints=1, fancybox=True)
    plt.savefig(directory + '/' + name + '_' + title + '.pdf')




def plotExperiment(labels,average_reward,average_regret,best_regret,title,sub=False):
    directory = './results/' + title
    if sub:
        directory = './results/singleExperiments/' + title
    if not os.path.exists(directory):
        os.makedirs(directory)

    plot(labels, average_reward, "Pulls","Reward",title,'average_reward',directory)
    plot(labels, average_regret, "Pulls","Regret",title,'average_regret',directory)
    plot(labels, best_regret, "Pulls","Regret",title,'best_regret',directory)



def runExperiments(max_pulls,real_distribution,labels,strategies,number_runs,title):
    bandits = Bandits(real_distribution)
    total = np.zeros(max_pulls)


    average_reward = dict.fromkeys(labels)
    average_regret = dict.fromkeys(labels)
    best_reward = dict.fromkeys(labels)
    best_regret = dict.fromkeys(labels)
    seeds = [i*123 for i in range(0,number_runs)]

    for i,strategy in enumerate(labels):
        print (strategy)
        total_reward = []
        total_regret = []
        for run in range(0,number_runs): 
            experiment = Experiment(bandits,strategies[i],max_pulls)
            if strategy == "OptimisticInitialization" :
                experiment.setQ(10) #High Q-value
            reward,regret = experiment.sample_bandits(seeds[run])
            total_reward.append(reward)
            total_regret.append(regret)
        average_reward[strategy] = np.mean(total_reward, axis=0 ).tolist()
        average_regret[strategy] = np.mean(total_regret, axis=0 ).tolist()
        best_regret[strategy] = total_regret[np.argmin([run[len(total_regret)] for run in total_regret])]
        plotExperiment([strategy], average_reward, average_regret,best_regret,strategy,sub=True)
    
    plotExperiment(labels, average_reward, average_regret,best_regret, title)


    return average_reward,average_regret,best_regret


def experimentsAll(max_pulls,real_distribution,number_experiment):
    sorted_distribution = np.sort(real_distribution)
    n = len(sorted_distribution)-1
    labels = ["EpsilonGreedy","Softmax","OptimisticInitialization","UCB2","UCB1","EnGreedy","Chernoff","Thompson","Efficient-UCBV","EXP3","MOSS"]
    strategies = [EpsilonGreedy(0.01),Softmax(0.1),Greedy(),UCB2(0.0001),UCB1(),EnGreedy(0.20,sorted_distribution[n]-sorted_distribution[n-1]),UCB1Chernoff(),ThompsonSampling(1,1),UCBV(0.6,(2500/(4**2))),EXP3(0.01),MOSS(1)] 
    average_reward,average_regret,best_regret = runExperiments(max_pulls, real_distribution,labels,strategies,number_experiment,"experimentsAll")
    labels_experiment1 = ["EpsilonGreedy","Softmax","OptimisticInitialization"]
    labels_experiment2 = ["UCB2","UCB1","EnGreedy","Chernoff","Thompson","Efficient-UCBV","EXP3","MOSS"]
    labels_experiment3 = ["UCB2","UCB1","EnGreedy","Chernoff","Thompson","Efficient-UCBV","EXP3"]
    plotExperiment(labels_experiment1, average_reward, average_regret,best_regret, "experiment1")
    plotExperiment(labels_experiment2, average_reward, average_regret,best_regret, "experiment2")
    plotExperiment(labels_experiment3, average_reward, average_regret,best_regret, "experiment3")



if __name__ == "__main__":




    parser = argparse.ArgumentParser()
    # Required positional argument

    plt.figure()
    parser.add_argument('--number_runs', type=int,
                    help='The amount of runs per experiment')
    parser.add_argument('--strategy',
                    help='The exploration strategy')
    args = parser.parse_args()

    max_pulls = 2500

    number_experiment = args.number_runs
    args = parser.parse_args()
    real_distribution = np.array([0.9,0.4,0.4,0.4])
    #real_distribution = np.array([0.1,0.4,0.9,0.6])
    #real_distribution = np.array([0.1,0.9,0.8,0.8,0.7,0.7,0.7,0.6,0.6,0.6])
    #real_distribution = np.array([0.1,0.6,0.6,0.9])
    #real_distribution = np.random.random_sample((5,))

    
    experimentsAll(max_pulls, real_distribution, args.number_runs)






