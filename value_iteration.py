import numpy as np


def run_value_iteration(mdp, gamma=0.95, eps=0.1, keep_history=False, iter_max=500):
    """ Value iteration of a MDP to find the optimal policy and its evaluation V
        keep_history -> to keep the convergence history and plot it"""
    n = mdp.nb_s
    a = mdp.nb_a
    Z = np.zeros((a, 1))  # Intermediary values to maximise
    pi = np.zeros((n, 1))  # Policy
    nb_iter = 0
    
    V = np.random.random((n, 1))
    V_prec = V + eps + 1
    if keep_history:
        V_hist = [V]
    
    while nb_iter < iter_max and np.linalg.norm(V - V_prec) > eps:
        nb_iter += 1
        V_prec = V.copy()
        for state in range(n):
            for action in range(a):
                esp = [V_prec[y] * mdp.transition[state, action, y]
                       for y in range(n)]
                Z[action] = mdp.rewards[state, action] + gamma * sum(esp)
            V[state] = np.max(Z)
            pi[state] = np.argmax(Z)
            
            if keep_history:
                V_hist.append(V.copy())
    
    if nb_iter == iter_max:
        print('NO CONVERGENCE in {} iterations'.format(iter_max))
    else:
        print("Convergence in {} iterations, precision of {}".format(nb_iter, eps))
    if keep_history:
        return pi, V, V_hist
    else:
        return pi, V