import numpy as np
import math
from tqdm import tqdm

from Approximate_DMDP.randomized_value_iteration import apx_val, apx_trans


def apx_mon_val(mdp, u, pi, v0, x, eps, delta):
    """ Monotonic Random Value Operator """
    v_tilde = np.zeros((mdp.nb_s, 1))
    pi_tilde = np.zeros((mdp.nb_s, 1))
    q, w = apx_val(mdp, u, v0, x, eps, delta)

    for i in range(mdp.nb_s):
        if q[i, 0] - 2 * mdp.gamma * eps > u[i, 0]:
            v_tilde[i, 0] = q[i, 0] - 2 * mdp.gamma * eps
            pi_tilde[i, 0] = w[i, 0]
        else:
            v_tilde[i, 0] = u[i, 0]
            pi_tilde[i, 0] = pi[i, 0]

    return v_tilde, pi_tilde

def sample_randomize_mon_VI(mdp, v0, pi0, T, eps, delta, analyze=False):
    """ Monotonic Sampled Randomized Value Iteration """

    m_hist = []

    # Sample to obtain x approximation of p.v0
    x = np.zeros((mdp.nb_s, mdp.nb_a))
    for i in range(mdp.nb_s):
        for a in range(mdp.nb_a):
            x[i, a] = apx_trans(mdp, v0, np.max(v0), i, a, eps, delta)

    v_t = v0
    pi_t = pi0
    for t in range(T):
        v_t, pi_t = apx_mon_val(mdp, v_t, pi_t, v0, x, eps/2, delta/T)

        if analyze:
            m = int(2 * np.max(np.abs(v_t - v0))**2 / (eps**2/4) \
                * np.log(2 / (delta/T))) + 1
            m_hist.append(m)

    return v_t, pi_t, m_hist
        
def sublinear_random_mon_VI(mdp, eps, delta, analyze=False):
    """ Montonic Sublinear Time Randomized Value Iteration """

    analysis = {}
    K = math.log(mdp.M / (eps * (1 - mdp.gamma)), 2)
    K = int(K) + 1
    T = 1. / (1 - mdp.gamma) * math.log(4. / (1 - mdp.gamma))
    T = int(T) + 1

    v_k = np.zeros((mdp.nb_s, 1))
    pi_k = np.zeros((mdp.nb_a, 1))
    eps_k = mdp.M / (1 - mdp.gamma)

    if analyze:
        analysis['K'] = K
        analysis['T'] = T
        analysis['eps'] = eps
        analysis['delta'] = delta
        analysis['m_hist'] = []
        analysis['V_hist'] = []
        analysis['pi_hist'] = []

    for k in tqdm(range(K)):
        eps_k = 0.5 * eps_k
        v_k, pi_k, m_hist = sample_randomize_mon_VI(mdp, v_k, pi_k, T,
            (1 - mdp.gamma)*eps/(4*mdp.gamma), delta/K, analyze)

        if analyze:
            analysis['m_hist'].append(m_hist)
            analysis['V_hist'].append(v_k)
            analysis['pi_hist'].append(pi_k)

    return v_k, pi_k, analysis