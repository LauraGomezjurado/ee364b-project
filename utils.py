import numpy as np
from collections import defaultdict
from typing import Callable
from tqdm import trange
import matplotlib.pyplot as plt
import os

DEF_GAMMA = 0.99

def collect_occupancy(env, policy_fn: Callable[[int], int],
                      n_episodes=500, gamma=DEF_GAMMA):
    """
    Roll out a policy and return an (nS*nA,) discounted occupancy vector.
    """
    d = np.zeros(env.nS * env.nA)
    for _ in range(n_episodes):
        s, _ = env.reset()
        t, done = 0, False
        while not done:
            a = policy_fn(s)
            s_next, _, done, _, _ = env.step(a)
            idx = s * env.nA + a
            d[idx] += (1 - gamma) * (gamma ** t)
            s, t = s_next, t + 1
    return d / n_episodes    # empirical expectation




def plot_curves(history, keys=("f", "unsafe"), save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    for k in keys:
        plt.figure()
        plt.plot(history[k])
        plt.title(k)
        plt.xlabel("iteration ×100")
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{k}_curve.png")
        plt.savefig(save_path)

    plt.show()