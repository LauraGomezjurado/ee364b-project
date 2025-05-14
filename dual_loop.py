import argparse, numpy as np
from tqdm import trange
from envs import GridWorld
from utils import collect_occupancy, DEF_GAMMA, plot_curves
from collections import defaultdict


# hyper-parameters
ETA_LAM = 1.0            # step-size for λ
ETA_MU  = 10.0            # step-size for μ (safety)
TEMP    = 1.0           # soft-Q temperature
ALPHA   = 0.1           # Q-learning α

def expert_policy(s, size=5):
    row, col = divmod(s, size)
    if col < size-1: return 3          # go RIGHT
    if row < size-1: return 1          # then DOWN
    return 0                           # arbitrary at goal

#  soft-Q policy helper 
def solve_soft_q(env, reward_vec, n_iter=100):
    Q = np.zeros((env.nS, env.nA))
    for _ in range(n_iter):
        for s in range(env.nS):
            for a in range(env.nA):
                s_next = env._next_state(s, a)
                max_q  = np.max(Q[s_next])            # one-step lookahead
                Q[s, a] = (1-ALPHA)*Q[s, a] + ALPHA*(reward_vec[s*env.nA+a] +
                                                      DEF_GAMMA * max_q)
    # softmax policy
    pi = np.exp(Q/TEMP) / np.sum(np.exp(Q/TEMP), axis=1, keepdims=True)
    return pi

def rollout_policy(env, pi, n_episodes=50):
    d = np.zeros(env.nS * env.nA)
    for _ in range(n_episodes):
        s, _ = env.reset()
        t, done = 0, False
        while not done:
            a = np.random.choice(env.nA, p=pi[s])
            s_next, _, done, _, _ = env.step(a)
            d[s*env.nA + a] += (1-DEF_GAMMA)*(DEF_GAMMA**t)
            s, t = s_next, t+1
    return d / n_episodes


def main(args):
    env = GridWorld()
    # expert occupancy 
    d_E = collect_occupancy(env, expert_policy, n_episodes=1000)
    # convex parts 
    unsafe_idx = np.repeat(env.unsafe_mask, env.nA)
    c_unsafe   = unsafe_idx.astype(float)             # 0/1 vector
    tau = 0.05                                        # threshold

    # dual vars 
    lam = np.zeros_like(d_E)      # λ
    mu  = 0.0                     # μ
    avg_d = np.zeros_like(d_E)

    history = defaultdict(list)

    for k in trange(1, args.iter+1):
        #policy player
        reward = -(lam + mu * c_unsafe)
        pi     = solve_soft_q(env, reward, n_iter=80)
        d      = rollout_policy(env, pi)

        # dual updates (OGD)
        lam += ETA_LAM * 2*(d - d_E)
        mu  = max(0.0, mu + ETA_MU * (np.dot(d, c_unsafe) - tau))

        # running average and logs
        avg_d = ((k-1)*avg_d + d) / k
        if k % 100 == 0:
            f_val     = np.sum((avg_d - d_E)**2)
            unsafe    = np.dot(avg_d, c_unsafe)
            history["f"].append(f_val)
            history["unsafe"].append(unsafe)
            print(f"iter={k:4d}  f={f_val:.4f}  unsafe={unsafe:.3f}")


    np.savez("logs.npz", **history)
    plot_curves(history)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=5000,
                        help="outer iterations")
    main(parser.parse_args())
