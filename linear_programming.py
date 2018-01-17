import numpy as np
import pulp

def i_a_to_indice(i, a, nb_a):
    """ Transform (i, a) to an unique indice """
    return i*nb_a + a

def indice_to_i_a(indice, nb_a):
    """ Find the couple (i, a) from the indice
        Inverse of "i_a_to_indice" """
    return indice//nb_a, indice%nb_a

def LP_param_DMDP(mdp):
    """ Find the matrices A, r such as the constraint of the LP is A.v >= r
        - A = E - gamma*P , size = (nb_s * nb_a) x nb_s
              with E[(i, a), j] = 1 iff i == j
              and P transition matrix of the MDP
        - r reward vector of size nb_s*nb_a x 1 """
    r = np.zeros((mdp.nb_s * mdp.nb_a, 1))
    A = np.zeros((mdp.nb_s * mdp.nb_a, mdp.nb_s))

    for a in range(mdp.nb_a):
        for s in range(mdp.nb_s):
            indice = i_a_to_indice(s, a, mdp.nb_a)
            r[indice] = mdp.rewards[s, a]

            for j in range(mdp.nb_s):
                if j == s:
                    A[indice][j] = 1 - mdp.gamma*mdp.transition[s, a, j]
                else:
                    A[indice][j] = - mdp.gamma*mdp.transition[s, a, j]
    return A, r

def LP_solving_DMDP(mdp):
    A, r = LP_param_DMDP(mdp)

    # Instantiate our problem class
    model = pulp.LpProblem("LP maximising problem", pulp.LpMinimize)

    # Variable
    v = pulp.LpVariable.dicts("Value Function",
                              (i for i in range(mdp.nb_s)),
                              lowBound=0,
                              cat='Continuous')

    # Objective function
    model += pulp.lpSum([v[i] for i in range(mdp.nb_s)])

    # Constraints
    for j in range(mdp.nb_s * mdp.nb_a):
        model += pulp.lpSum([A[j, k] * v[k] for k in range(mdp.nb_s)]) >= r[j, 0]
        
    # Solve our problem
    model.solve()
    
    if pulp.LpStatus[model.status] == 'Optimal':
        v_final = [v[i].varValue for i in range(mdp.nb_s)]
    else:
        print("No optimal solution found, status =", pulp.LpStatus[model.status])

    # Compute corresponding pi
    pi = np.zeros((mdp.nb_s, 1))
    for state in range(mdp.nb_s):
        values = [mdp.rewards[state, action] \
                  + mdp.gamma * mdp.transition[state, action, :].dot(v_final)
                  for action in range(mdp.nb_a)]
        pi[state, 0] = np.argmax(values)

    return v_final, pi