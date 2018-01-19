import numpy as np 
import time
import math
import pickle

### create mdp

from DMDP_class import DMDP, create_random_DMDP

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

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
v_sub, pi_sub, analysis_sub = sublinear_time_randomized_VI(mdp, eps=0.1, delta=0.1, analyze=True)
save_obj(analysis_sub, "duration_sub")


print("Sub lin time:", time.time() - start_time)
print("Policy:", pi_sub.T)
print("Value function", np.linalg.norm(v_sub.T, ord=np.inf))

print("######## \n")

from value_iteration import run_value_iteration

start_time  = time.time()
pi_VI, V_VI = run_value_iteration(mdp, eps=0.001)

print("VI time:", time.time() - start_time)
print("Policy:", pi_VI.T)
print("Value function", np.linalg.norm(V_VI.T, ord=np.inf))