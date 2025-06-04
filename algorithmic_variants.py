#!/usr/bin/env python3
"""
Algorithmic Variants & Ablations

Tests different algorithms for both cost player and policy player:
1. Cost-player algorithms: FTL vs OMD vs FTRL
2. Policy-player oracles: Best-response vs one-step optimistic
3. Frank-Wolfe vs Primal-Dual comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import trange
import argparse

from envs import GridWorld
from dual_loop import expert_policy, solve_soft_q, rollout_policy
from utils import collect_occupancy, DEF_GAMMA

class FTLOptimizer:
    """Follow-the-Leader for cost player"""
    def __init__(self):
        self.past_gradients = []
    
    def update(self, gradient, eta):
        self.past_gradients.append(gradient.copy())
        # FTL: minimize sum of all past linear functions
        avg_gradient = np.mean(self.past_gradients, axis=0)
        return -eta * avg_gradient

class OMDOptimizer:
    """Online Mirror Descent with entropy regularization"""
    def __init__(self, dim):
        self.theta = np.zeros(dim)  # In log space
        
    def update(self, gradient, eta):
        # OMD update in log space
        self.theta -= eta * gradient
        # Project back to probability simplex
        weights = np.exp(self.theta - np.max(self.theta))
        return weights / np.sum(weights)

class FTRLOptimizer:
    """Follow-the-Regularized-Leader"""
    def __init__(self, dim, reg_strength=0.01):
        self.sum_gradients = np.zeros(dim)
        self.reg_strength = reg_strength
        
    def update(self, gradient, eta):
        self.sum_gradients += gradient
        # FTRL with L2 regularization
        return -eta * self.sum_gradients / (1 + self.reg_strength * eta)

def run_cost_player_comparison():
    """Compare FTL vs OMD vs FTRL for cost player"""
    print(" Cost Player Algorithm Comparison")
    
    env = GridWorld()
    d_E = collect_occupancy(env, expert_policy, n_episodes=1000)
    
    unsafe_idx = np.repeat(env.unsafe_mask, env.nA)
    c_unsafe = unsafe_idx.astype(float)
    tau = 0.05
    
    algorithms = {
        'OGD': None,  # Standard OGD (baseline)
        'FTL': FTLOptimizer(),
        'OMD': OMDOptimizer(len(d_E)),
        'FTRL': FTRLOptimizer(len(d_E))
    }
    
    results = {}
    
    for alg_name, optimizer in algorithms.items():
        print(f"   Testing {alg_name}...")
        
        lam = np.zeros_like(d_E)
        mu = 0.0
        avg_d = np.zeros_like(d_E)
        
        history = defaultdict(list)
        
        for k in trange(1, 2001, desc=f"{alg_name}", leave=False):
            # Policy player (same for all)
            reward = -(lam + mu * c_unsafe)
            pi = solve_soft_q(env, reward, n_iter=80)
            d = rollout_policy(env, pi)
            
            # Cost player update (different algorithms)
            gradient = 2*(d - d_E)
            
            if alg_name == 'OGD':
                # Standard OGD
                lam += 1.0 * gradient
            else:
                # Use the specific optimizer
                lam_update = optimizer.update(gradient, 1.0)
                if alg_name == 'OMD':
                    lam = lam_update * np.sum(np.abs(lam))  # Scale back
                else:
                    lam += lam_update
            
            # Constraint dual update (same for all)
            mu = max(0.0, mu + 10.0 * (np.dot(d, c_unsafe) - tau))
            
            avg_d = ((k-1)*avg_d + d) / k
            
            if k % 50 == 0:
                f_val = np.sum((avg_d - d_E)**2)
                unsafe = np.dot(avg_d, c_unsafe)
                history["f"].append(f_val)
                history["unsafe"].append(unsafe)
                history["iteration"].append(k)
        
        results[alg_name] = history
        final_f = history["f"][-1]
        print(f"   {alg_name} final f: {final_f:.6f}")
    
    return results

def run_policy_player_comparison():
    """Compare best-response vs one-step optimistic policy player"""
    print("\n Policy Player Oracle Comparison")
    
    env = GridWorld()
    d_E = collect_occupancy(env, expert_policy, n_episodes=1000)
    
    unsafe_idx = np.repeat(env.unsafe_mask, env.nA)
    c_unsafe = unsafe_idx.astype(float)
    tau = 0.05
    
    methods = {
        'Best-Response': {'q_iters': 200, 'optimistic': False},
        'One-Step': {'q_iters': 1, 'optimistic': False},
        'Optimistic': {'q_iters': 80, 'optimistic': True}
    }
    
    results = {}
    
    for method_name, params in methods.items():
        print(f"   Testing {method_name}...")
        
        lam = np.zeros_like(d_E)
        mu = 0.0
        avg_d = np.zeros_like(d_E)
        
        history = defaultdict(list)
        
        for k in trange(1, 1501, desc=f"{method_name}", leave=False):
            # Policy player with different oracles
            reward = -(lam + mu * c_unsafe)
            
            if params['optimistic']:
                # Optimistic update: use next iteration's dual variables
                future_lam = lam + 1.0 * 2*(avg_d - d_E)  # Predict next lam
                future_mu = max(0.0, mu + 10.0 * (np.dot(avg_d, c_unsafe) - tau))
                reward = -(future_lam + future_mu * c_unsafe)
            
            pi = solve_soft_q(env, reward, n_iter=params['q_iters'])
            d = rollout_policy(env, pi)
            
            # Standard dual updates
            lam += 1.0 * 2*(d - d_E)
            mu = max(0.0, mu + 10.0 * (np.dot(d, c_unsafe) - tau))
            
            avg_d = ((k-1)*avg_d + d) / k
            
            if k % 30 == 0:
                f_val = np.sum((avg_d - d_E)**2)
                unsafe = np.dot(avg_d, c_unsafe)
                history["f"].append(f_val)
                history["unsafe"].append(unsafe)
                history["iteration"].append(k)
        
        results[method_name] = history
        final_f = history["f"][-1]
        print(f"   {method_name} final f: {final_f:.6f}")
    
    return results

def run_frank_wolfe_comparison():
    """Compare Frank-Wolfe vs Primal-Dual"""
    print("\n Frank-Wolfe vs Primal-Dual")
    
    env = GridWorld()
    d_E = collect_occupancy(env, expert_policy, n_episodes=1000)
    
    unsafe_idx = np.repeat(env.unsafe_mask, env.nA)
    c_unsafe = unsafe_idx.astype(float)
    tau = 0.05
    
    results = {}
    
    # Primal-Dual (our standard method)
    print("   Testing Primal-Dual...")
    lam = np.zeros_like(d_E)
    mu = 0.0
    avg_d = np.zeros_like(d_E)
    
    pd_history = defaultdict(list)
    
    for k in trange(1, 1001, desc="Primal-Dual", leave=False):
        reward = -(lam + mu * c_unsafe)
        pi = solve_soft_q(env, reward, n_iter=80)
        d = rollout_policy(env, pi)
        
        lam += 1.0 * 2*(d - d_E)
        mu = max(0.0, mu + 10.0 * (np.dot(d, c_unsafe) - tau))
        
        avg_d = ((k-1)*avg_d + d) / k
        
        if k % 20 == 0:
            f_val = np.sum((avg_d - d_E)**2)
            unsafe = np.dot(avg_d, c_unsafe)
            pd_history["f"].append(f_val)
            pd_history["unsafe"].append(unsafe)
            pd_history["iteration"].append(k)
    
    results['Primal-Dual'] = pd_history
    
    # Frank-Wolfe (simplified version)
    print("   Testing Frank-Wolfe...")
    d_current = np.ones_like(d_E) / len(d_E)  # Start with uniform
    
    fw_history = defaultdict(list)
    
    for k in trange(1, 1001, desc="Frank-Wolfe", leave=False):
        # Compute gradient of Lagrangian
        gradient = 2*(d_current - d_E)
        
        # Add constraint violation penalty
        constraint_violation = max(0, np.dot(d_current, c_unsafe) - tau)
        if constraint_violation > 0:
            gradient += 10.0 * constraint_violation * c_unsafe
        
        # Frank-Wolfe: find direction that minimizes linear approximation
        # This is a simplified version - in practice would solve LP
        reward = -gradient
        pi = solve_soft_q(env, reward, n_iter=80)
        d_fw = rollout_policy(env, pi)
        
        # Line search step size (simplified)
        gamma = 2.0 / (k + 2)  # Standard FW step size
        d_current = (1 - gamma) * d_current + gamma * d_fw
        
        if k % 20 == 0:
            f_val = np.sum((d_current - d_E)**2)
            unsafe = np.dot(d_current, c_unsafe)
            fw_history["f"].append(f_val)
            fw_history["unsafe"].append(unsafe)
            fw_history["iteration"].append(k)
    
    results['Frank-Wolfe'] = fw_history
    
    for method, hist in results.items():
        final_f = hist["f"][-1]
        print(f"   {method} final f: {final_f:.6f}")
    
    return results

def create_algorithm_plots(cost_results, policy_results, fw_results):
    """Create visualization plots for algorithmic comparisons"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Cost player comparison - convergence
    ax = axes[0, 0]
    for alg, hist in cost_results.items():
        ax.plot(hist["iteration"], hist["f"], label=alg, linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective f(d)')
    ax.set_title('Cost Player Algorithms\nConvergence Comparison')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Cost player comparison - final performance
    ax = axes[1, 0]
    algs = list(cost_results.keys())
    final_fs = [cost_results[alg]["f"][-1] for alg in algs]
    bars = ax.bar(algs, final_fs, alpha=0.7)
    ax.set_ylabel('Final Objective f(d)')
    ax.set_title('Final Performance')
    ax.set_yscale('log')
    
    # Add value labels
    for bar, val in zip(bars, final_fs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', rotation=45)
    
    # Policy player comparison - convergence
    ax = axes[0, 1]
    for method, hist in policy_results.items():
        ax.plot(hist["iteration"], hist["f"], label=method, linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective f(d)')
    ax.set_title('Policy Player Oracles\nConvergence Comparison')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Policy player comparison - constraint satisfaction
    ax = axes[1, 1]
    for method, hist in policy_results.items():
        violations = [max(0, u - 0.05) for u in hist["unsafe"]]
        ax.plot(hist["iteration"], violations, label=method, linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Constraint Violation')
    ax.set_title('Constraint Satisfaction')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Frank-Wolfe vs Primal-Dual - convergence
    ax = axes[0, 2]
    for method, hist in fw_results.items():
        ax.plot(hist["iteration"], hist["f"], label=method, linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective f(d)')
    ax.set_title('Frank-Wolfe vs Primal-Dual\nConvergence Rate')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Convergence rate analysis
    ax = axes[1, 2]
    for method, hist in fw_results.items():
        # Compute convergence rate (slope in log space)
        iterations = np.array(hist["iteration"])
        f_vals = np.array(hist["f"])
        
        # Fit line in log space for last half of iterations
        mid_point = len(iterations) // 2
        log_iters = np.log(iterations[mid_point:])
        log_f = np.log(f_vals[mid_point:])
        
        if len(log_iters) > 1:
            slope = np.polyfit(log_iters, log_f, 1)[0]
            ax.bar(method, -slope, alpha=0.7, label=f'{method}: {-slope:.2f}')
    
    ax.set_ylabel('Convergence Rate (-slope)')
    ax.set_title('Convergence Rate Comparison\n(Higher = Faster)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('algorithmic_variants_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Algorithmic Variants & Ablations")
    parser.add_argument("--test", choices=["cost", "policy", "fw", "all"], 
                       default="all", help="Which comparison to run")
    args = parser.parse_args()
    
    print(" ALGORITHMIC VARIANTS & ABLATIONS")
    print("=" * 50)
    
    if args.test in ["cost", "all"]:
        cost_results = run_cost_player_comparison()
    else:
        cost_results = None
        
    if args.test in ["policy", "all"]:
        policy_results = run_policy_player_comparison()
    else:
        policy_results = None
        
    if args.test in ["fw", "all"]:
        fw_results = run_frank_wolfe_comparison()
    else:
        fw_results = None
    
    # Create plots if we ran all tests
    if args.test == "all":
        print("\n Creating visualization plots...")
        create_algorithm_plots(cost_results, policy_results, fw_results)
        print(" Results saved to algorithmic_variants_results.png")
    
    print("\n Algorithmic comparison completed!")

if __name__ == "__main__":
    main() 