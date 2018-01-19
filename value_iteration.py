import numpy as np


def run_value_iteration_while(mdp, eps=0.01, keep_history=False, iter_max=500):
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
                Z[action] = mdp.rewards[state, action] + mdp.gamma * sum(esp)
            V[state] = np.max(Z)
            pi[state] = np.argmax(Z)
            
            if keep_history:
                V_hist.append(V.copy())
    
    if nb_iter == iter_max:
        print('NO CONVERGENCE in {} iterations'.format(iter_max))
    #else:
    #    print("Convergence in {} iterations, precision of {}".format(nb_iter, eps))
    if keep_history:
        return V, pi, V_hist
    else:
        return V, pi

def run_value_iteration(mdp, eps, keep_history=False, iter_max=500):
    """ Value iteration of a MDP to find the optimal policy and its evaluation V
        keep_history -> to keep the convergence history and plot it

        Run using a number of iteration depending on eps (the precision)"""
    K = np.log(eps*(1 - mdp.gamma)/mdp.M) / np.log(mdp.gamma)
    K = int(K) + 1

    n = mdp.nb_s
    a = mdp.nb_a
    Z = np.zeros((a, 1))  # Intermediary values to maximise
    pi = np.zeros((n, 1))  # Policy
    
    V = np.zeros((n, 1))
    if keep_history:
        V_hist = [V]
    
    for k in range(K):
        for state in range(n):
            for action in range(a):
                esp = [V[y] * mdp.transition[state, action, y]
                       for y in range(n)]
                Z[action] = mdp.rewards[state, action] + mdp.gamma * sum(esp)
            V[state] = np.max(Z)
            pi[state] = np.argmax(Z)
            
            if keep_history:
                V_hist.append(V.copy())
    
    if keep_history:
        return pi, V, V_hist
    else:
        return pi, np.array(V)