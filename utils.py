
import numpy as np
from collections import defaultdict
from typing import Callable
from tqdm import trange
import matplotlib.pyplot as plt
import os

DEF_GAMMA = 0.99

def collect_occupancy(env, policy_fn: Callable[[int, int], int], # Expects policy_fn(state, size)
                      n_episodes=500, gamma=DEF_GAMMA):
    """
    Roll out a policy and return an (nS*nA,) discounted occupancy vector.
    """
    d = np.zeros(env.nS * env.nA)
    for _ in range(n_episodes): # Consider trange(n_episodes, desc="Collecting Occupancy", leave=False) for progress
        s, _ = env.reset()
        t, done = 0, False
        while not done:
            a = policy_fn(s, env.size) # Pass env.size to the policy function
            s_next, _, done, _, _ = env.step(a)
            idx = s * env.nA + a
            d[idx] += (1 - gamma) * (gamma ** t)
            s, t = s_next, t + 1
    return d / n_episodes


def plot_curves(history, keys=("f", "unsafe"), save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    for k_idx, key_name in enumerate(keys):
        plt.figure(figsize=(8, 5))
        if isinstance(history[key_name], list): 
            plt.plot(history.get("iteration", range(len(history[key_name]))), history[key_name])
        else:
            print(f"Warning: Plotting for key '{key_name}' might be non-standard.")
            plt.plot(history[key_name]) # Fallback
            
        plt.title(f"Metric: {key_name.capitalize()}")
        plt.xlabel("Iteration" if "iteration" in history else "Log Point (e.g., ×100 iterations)")
        plt.ylabel(key_name.capitalize())
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{key_name}_curve.png")
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")

    # plt.show() 