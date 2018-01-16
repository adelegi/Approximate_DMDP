import numpy as np


def apx_trans(mdp, u, M, i, a, eps, delta):
    """ Approximate Transition
        - give an eps-approxiamtion of p_a(i)^T v_i
    """
    assert(np.max(u) <= M)
    m = int(2 * M**2 / (eps**2) * np.log(2 / delta)) + 1
    result = 0

    for k in range(m):
        j, _ = mdp.step(i, a)
        result += u[j, 0]

    return result / m


def exacte_trans(mdp, u, M, i, a, eps, delta):
    return mdp.transition[i, a, :].dot(u[:, 0])


def apx_val(mdp, u, v0, x, eps, delta):
    """ Approximate Value Operator
        - Compute policy pi and value function v using value iteration method
          with approximated transition p_a(i).v_i
    """
    M = np.max(np.abs(u - v0))
    delta2 = delta / (mdp.nb_s * mdp.nb_a)
    v = np.zeros((mdp.nb_s, 1))
    pi = np.zeros((mdp.nb_s, 1))

    for i in range(mdp.nb_s):
        Q = np.zeros((mdp.nb_a, 1))
        for a in range(mdp.nb_a):
            Q[a, 0] = mdp.gamma * \
                (x[i, a] + apx_trans(mdp, u - v0, M, i, a, eps, delta2))
            Q[a, 0] += mdp.rewards[i, a]
        v[i, 0] = np.max(Q[:, 0])
        pi[i, 0] = np.argmax(Q[:, 0])

    return v, pi


def randomizedVI(mdp, v0, L, eps, delta, analyze=False):
    m_hist = []

    x = np.zeros((mdp.nb_s, mdp.nb_a))
    for i in range(mdp.nb_s):
        x[i, :] = [mdp.transition[i, a, :].dot(v0) for a in range(mdp.nb_a)]

    v_prev = v0.copy()
    for l in range(L):
        v_l, pi_l = apx_val(mdp, v_prev, v0, x, eps, delta / L)
        v_prev = v_l

        if analyze:
            m = int(2 * np.max(np.abs(v_prev - v0))**2 / (eps**2) \
                * np.log(2 / delta)) + 1
            m_hist.append(m)

    return v_l, pi_l, m_hist
