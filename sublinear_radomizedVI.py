import numpy as np
import math
from tqdm import tqdm
import time

from Approximate_DMDP.randomized_value_iteration import apx_val, apx_trans

def sampled_randomized_VI(mdp, v0, L, eps, delta, analyze=False):
    """ Sampled Randomized Value Iteration 
    """
    m_hist = []
    m_x_hist = []
        

    # Sample to obtain x approximation of p.v0
    time_start_x = time.time()
    x = np.zeros((mdp.nb_s, mdp.nb_a))
    for i in range(mdp.nb_s):
        for a in range(mdp.nb_a):
            x[i, a] = apx_trans(mdp, v0, np.max(v0), i, a, eps, delta)

    if analyze:
        duration_x = time.time() - time_start_x
        m_x = int(2 * np.max(v0)**2 / (eps**2) * np.log(2 / delta)) + 1
        m_x_hist.append([m_x, duration_x])
        print("{} sec to compute x=p^Tv by doing {} iterations of ApxTrans".format(round(duration_x,4), m_x))
        print("norm inf v =", np.linalg.norm(v0, ord=np.inf))
            
    v_prev = v0.copy()
    for l in range(L):
        start_time_l = time.time()
        v_l, pi_l = apx_val(mdp, v_prev, v0, x, eps, delta / L)
        v_prev = v_l

        if analyze:
            duration_l = time.time() - start_time_l
            m = int(2 * np.max(np.abs(v_prev - v0))**2 / (eps**2) \
                * np.log(2 / delta)) + 1
            m_hist.append(m)
            print("Iteration l={}, |S||A|*{} iterations of ApxTrans in {} sec, because |v_prev - v0|_inf ={}".format(
                l, m, round(duration_l,4), np.max(np.abs(v_prev - v0))))
            m_hist.append([m, duration_l])

    return v_l, pi_l, m_hist, m_x_hist

def sublinear_time_randomized_VI(mdp, eps, delta, analyze=False, v0_value=0, show_time=False):
    """ Sublinear Time Randomized Value Iteration
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
        print("K={} principal iterations, with L={} iterations in each one".format(K, L))
        # Information about the problem
        analysis['m_hist'] = {}
        analysis['m_x_hist'] = {}
        analysis['duration_iter_k'] = {}

    for k in tqdm(range(K)):
        start_time_k = time.time()
        eps_k = eps_prev * 0.5
        eps_func = (1 - mdp.gamma) * eps_k / (4 * mdp.gamma)
        v_k, pi_k, m_hist, m_x_hist = sampled_randomized_VI(mdp, v_prev, L, eps_func, 
                                 delta / K, analyze)

        if analyze:
            duration = time.time() - start_time_k
            print("Iteration k={}: {}sec".format(k, round(duration,4)))
            analysis['m_hist'][k] = m_hist
            analysis['m_x_hist'][k] = m_x_hist
            analysis['duration_iter_k'][K] = duration

        eps_prev = eps_k
        v_prev = v_k

    return v_k, pi_k, analysis
