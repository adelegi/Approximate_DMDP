import numpy as np
import math
from tqdm import tqdm

from sampling_tree import Sampling_tree, Sampling_tree_with_policy_updates




class Randomized_Primal_Dual:
    """
    performs randomized primal - dual algorithm
    """
    
    def __init__(self, n_states, n_actions, rewards, transition_probabilities, gamma=0.95):
        
        self.s = n_states
        self.a = n_actions
        self.gamma = gamma
        
        self.r = rewards # shape (state, action)
        self.p = transition_probabilities # shape (state, action, next_state)
        
    def preprocess(self, T):
        
        self.q = np.ones(self.s) / self.s
        self.xi = np.ones(self.s) / self.s
        self.pi = np.ones((self.s, self.a)) / self.a
        self.v = np.zeros(self.s)
        
        self.theta = 1-self.gamma
        self.beta = self.theta*np.sqrt(np.log(self.s*self.a + 1) / (2*self.s*self.a*T))
        self.alpha = self.beta*self.s / (2*(1-self.gamma)**2)
        self.M = 1 / (1-self.gamma)
        
        self.sample_i = Sampling_tree_with_policy_updates(list((1-self.theta)*self.xi + self.theta*self.q))
        
        sample_a = []
        sample_j = []
        
        for i in range(self.s):
            
            sample_a.append(Sampling_tree_with_policy_updates(list(self.pi[0, :])))
            sample_j.append([])
            
            for a in range(self.a):
                
                sample_j[i].append(Sampling_tree(list(self.p[i, a, :])))
                
        self.sample_a = sample_a
        self.sample_j = sample_j
        
        print("finished preprocessing")
        
        
    def run(self, T):
        
        self.preprocess(T)
        
        average_policy = np.zeros((self.s, self.a))
        
        for t in tqdm(range(T)):
            
            # sampling
            
            #print("step {}".format(t))
            
            i = self.sample_i.sample()
            a = self.sample_a[i].sample()
            j = self.sample_j[i][a].sample()
            
            # update
            
            p_i = (1-self.theta)*self.xi[i] + self.theta*self.q[i]
            
            delta = self.beta*(self.gamma*self.v[j] - self.v[i] + self.r[i, a] - self.M) / p_i / self.pi[i, a] # r
            self.v[i] = max(min(self.v[i] - self.alpha*(self.theta*self.q[i]/p_i - 1), self.M), 0)
            self.v[j] = max(min(self.v[j] - self.alpha*self.gamma, self.M), 0)
            
            new_xi_i = self.xi[i]*(1 + self.pi[i, a]*(np.exp(delta)-1))
            new_pi_ia = self.pi[i, a]*np.exp(delta)
            
            self.sample_i.update_weights(new_xi_i)
            self.sample_a[i].update_weights(new_pi_ia)
            
            self.xi[i] = new_xi_i
            self.pi[i, a] = new_pi_ia
            
            self.xi = self.xi / np.sum(self.xi)
            self.pi[i, :] = self.pi[i, :] / np.sum(self.pi[i, :])
            
            average_policy += self.pi
        
        return average_policy / T