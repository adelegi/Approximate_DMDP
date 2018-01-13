import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

from Approximate_DMDP.randomized_value_iteration import randomizedVI


def high_precision_randomized_VI(mdp, eps, delta, log=False):
    """ High Precision Randomized Value Iteration
        Gives an eps-approximate value function with probability 1 - delta
        - log: to plot the convergence of the value function
    """
    K = math.log(mdp.M / (eps * (1 - mdp.gamma)), 2)
    K = int(K) + 1
    L = 1. / (1 - mdp.gamma) * math.log(4. / (1 - mdp.gamma))
    L = int(L) + 1

    v0 = np.zeros((mdp.nb_s, 1))
    v_prev = v0
    eps_prev = mdp.M / (1 - mdp.gamma)

    if log: # Keep history of convergence
        print("{} iterations and {} iterations for randomizedVI".format(K, L))
        V_hist = [0]
        M_hist = [0]

    for k in tqdm(range(K)):
        eps_k = eps_prev * 0.5
        eps_func = (1 - mdp.gamma) * eps_k / (4 * mdp.gamma)
        v_k, pi_k = randomizedVI(mdp, v_prev, L, eps_func, 
                                 delta / K)

        if log:
            V_hist.append(np.linalg.norm(v_k))
            M_hist.append(np.max(np.abs(v_k - v_prev)))

        eps_prev = eps_k
        v_prev = v_k

    if log:
        plt.figure()
        plt.plot(V_hist[1:])
        plt.title("Convergence of the value function")
        plt.xlabel("Number of iterations")
        plt.ylabel("Norm of the value function")
        plt.plot()

        plt.figure()
        plt.plot(M_hist[1:])
        plt.title("Evolution of v_k - v_prev \
            (related to the number of iterations for the estimation of p_i(a).v)")
        plt.plot()

    return v_k, pi_k, V_hist, M_hist
