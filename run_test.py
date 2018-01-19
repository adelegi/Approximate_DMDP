import numpy as np 
import time
import math

### create mdp

from DMDP_class import DMDP, create_random_DMDP

reward_func = lambda s, a: np.random.random()
nb_a = 2
nb_s = 10 #1e6
gamma = 0.8

start_time  = time.time()
mdp = create_random_DMDP(nb_a, nb_s, reward_func, gamma)
print("mdp:", time.time() - start_time)

print("##### mdp done #####")

from sublinear_randomizedVI import sublinear_time_randomized_VI

start_time  = time.time()
v_sub, pi_sub, _ = sublinear_time_randomized_VI(mdp, eps=0.1, delta=0.1)

print("Sub lin time:", time.time() - start_time)
print("Policy:", pi_sub.T)
print("Value function", np.linalg.norm(v_sub.T, ord=inf))

print("######## \n")

from value_iteration import run_value_iteration

start_time  = time.time()
pi_VI, V_VI = run_value_iteration(mdp, eps=0.001)

print("VI time:", time.time() - start_time)
print("Policy:", pi_VI.T)
print("Value function", np.linalg.norm(V_VI.T, ord=inf))