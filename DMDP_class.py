import numpy as np


class DMDP:

    def __init__(self, nb_a, nb_s, R, P, gamma=0.95):

        self.nb_a = nb_a
        self.nb_s = nb_s
        self.gamma = gamma

        self.rewards = R  # R[state, action]
        self.transition = P  # P[state, action, next_state]
        self.M = int(np.max(np.abs(R))) + 1

    def reset(self):
        return np.random.randint(0, nb_s)

    def step(self, state, action):
        next_state = np.random.choice(self.nb_s, 1, p=self.transition[state, action, :])
        reward = self.rewards[state, action]
        return next_state[0], reward


def create_random_DMDP(nb_a, nb_s, reward_func, gamma=0.95):

    R = np.zeros((nb_s, nb_a))
    for s in range(nb_s):
        R[s, :] = np.array([reward_func(s, a) for a in range(nb_a)])

    P = np.random.random((nb_s, nb_a, nb_s))
    for a in range(nb_a):
        for s in range(nb_s):
            P[s, a, :] /= np.sum(P[s, a, :])

    return DMDP(nb_a, nb_s, R, P, gamma)
