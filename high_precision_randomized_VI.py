import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

from Approximate_DMDP.randomized_value_iteration import randomizedVI


def high_precision_randomized_VI(mdp, eps, delta, analyze=False, v0_value=0):
    """ High Precision Randomized Value Iteration
        Gives an eps-approximate value function with probability 1 - delta
        - analyze: to plot the convergence of the value function
    """
    K = math.log(mdp.M / (eps * (1 - mdp.gamma)), 2)
    K = int(K) + 1
    L = 1. / (1 - mdp.gamma) * math.log(4. / (1 - mdp.gamma))
    L = int(L) + 1

    v0 = np.zeros((mdp.nb_s, 1)) + v0_value
    v_prev = v0
    eps_prev = mdp.M / (1 - mdp.gamma)

    analysis = {}  # To keep convergence informations
    if analyze: # Keep history of convergence
        print("{} iterations and {} iterations for randomized_VI".format(K, L))
        # Information about the problem
        analysis['eps'] = eps
        analysis['delta'] = delta
        analysis['v0_value'] = v0_value
        analysis['M'] = mdp.M
        analysis['nb_s'] = mdp.nb_s
        analysis['nb_a'] = mdp.nb_a
        analysis['gamma'] = mdp.gamma

        analysis['K'] = K
        analysis['L'] = L
        analysis['V_hist'] = [v0]
        analysis['pi'] = []
        analysis['m_hist'] = {}

    for k in tqdm(range(K)):
        eps_k = eps_prev * 0.5
        eps_func = (1 - mdp.gamma) * eps_k / (4 * mdp.gamma)
        v_k, pi_k, m_hist = randomizedVI(mdp, v_prev, L, eps_func, 
                                 delta / K, analyze)

        if analyze:
            analysis['V_hist'].append(v_k)
            analysis['m_hist'][k] = m_hist
            analysis['pi'].append(pi_k)

        eps_prev = eps_k
        v_prev = v_k

    return v_k, pi_k, analysis
