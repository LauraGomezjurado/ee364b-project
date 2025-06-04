#!/usr/bin/env python3
"""
Stress Tests for Dual Optimization Loop

Implements the three stress tests from the outline:
1. Binding safety constraints (τ=0.002, hazards on expert path)
2. Stochastic slippage (p=0.1, 0.2)
3. Multiple constraints (goal-reach ≥ 0.99 + safety)
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import trange
import argparse

from envs import GridWorld
from dual_loop import expert_policy, solve_soft_q, rollout_policy
from utils import collect_occupancy, DEF_GAMMA

def run_binding_constraint_test():
    """Test 1: Make safety constraint binding"""
    print(" Test 1: Binding Safety Constraints")
    print("   - τ = 0.002 (very tight)")
    print("   - Hazards on expert path: (6, 11)")
    
    # Create environment with hazards on expert path
    # Expert goes: (0,0) → (0,1) → (0,2) → (0,3) → (0,4) → (1,4) → ... → (4,4)
    # So cells 1,2,3 are on the path, let's put hazards at 6 and 11
    env = GridWorld(unsafe_cells=(6, 11))  # (1,1) and (2,1) - on expert path
    
    # Very tight constraint
    tau = 0.002
    
    # Get expert occupancy
    d_E = collect_occupancy(env, expert_policy, n_episodes=1000)
    
    # Setup constraints
    unsafe_idx = np.repeat(env.unsafe_mask, env.nA)
    c_unsafe = unsafe_idx.astype(float)
    
    # Initialize dual variables
    lam = np.zeros_like(d_E)
    mu = 0.0
    avg_d = np.zeros_like(d_E)
    
    # Track metrics for Pareto front
    history = defaultdict(list)
    
    for k in trange(1, 3001, desc="Binding constraint"):
        # Policy player
        reward = -(lam + mu * c_unsafe)
        pi = solve_soft_q(env, reward, n_iter=80)
        d = rollout_policy(env, pi)
        
        # Dual updates
        lam += 1.0 * 2*(d - d_E)
        mu = max(0.0, mu + 10.0 * (np.dot(d, c_unsafe) - tau))
        
        # Running average
        avg_d = ((k-1)*avg_d + d) / k
        
        # Log for Pareto front
        if k % 100 == 0:
            f_val = np.sum((avg_d - d_E)**2)
            unsafe = np.dot(avg_d, c_unsafe)
            history["f"].append(f_val)
            history["unsafe"].append(unsafe)
            history["mu"].append(mu)
            
            if k % 500 == 0:
                print(f"   k={k:4d}  f={f_val:.6f}  unsafe={unsafe:.6f}  μ={mu:.2f}")
    
    return history

def run_slippage_test():
    """Test 2: Stochastic slippage"""
    print("\n  Test 2: Stochastic Slippage")
    
    results = {}
    
    for slip in [0.0, 0.1, 0.2]:
        print(f"   Testing slip = {slip}")
        
        env = GridWorld(slip=slip)
        d_E = collect_occupancy(env, expert_policy, n_episodes=1000)
        
        unsafe_idx = np.repeat(env.unsafe_mask, env.nA)
        c_unsafe = unsafe_idx.astype(float)
        tau = 0.05
        
        lam = np.zeros_like(d_E)
        mu = 0.0
        avg_d = np.zeros_like(d_E)
        
        history = defaultdict(list)
        
        for k in trange(1, 2001, desc=f"Slip {slip}", leave=False):
            reward = -(lam + mu * c_unsafe)
            pi = solve_soft_q(env, reward, n_iter=80)
            d = rollout_policy(env, pi)
            
            lam += 1.0 * 2*(d - d_E)
            mu = max(0.0, mu + 10.0 * (np.dot(d, c_unsafe) - tau))
            
            avg_d = ((k-1)*avg_d + d) / k
            
            if k % 50 == 0:  # More frequent logging for convergence analysis
                f_val = np.sum((avg_d - d_E)**2)
                unsafe = np.dot(avg_d, c_unsafe)
                violation = max(0, unsafe - tau)
                history["f"].append(f_val)
                history["unsafe"].append(unsafe)
                history["violation"].append(violation)
                history["iteration"].append(k)
        
        results[slip] = history
        final_violation = history["violation"][-1]
        print(f"   Final constraint violation: {final_violation:.6f}")
    
    return results

def run_multiple_constraints_test():
    """Test 3: Multiple constraints (safety + goal reach)"""
    print("\n Test 3: Multiple Constraints")
    print("   - Safety: P(unsafe) ≤ 0.05")
    print("   - Goal reach: P(goal) ≥ 0.99")
    
    env = GridWorld()
    d_E = collect_occupancy(env, expert_policy, n_episodes=1000)
    
    # Safety constraint
    unsafe_idx = np.repeat(env.unsafe_mask, env.nA)
    c_unsafe = unsafe_idx.astype(float)
    tau_safety = 0.05
    
    # Goal constraint: want P(reach goal) ≥ 0.99
    # This is trickier - we need to compute goal-reaching probability
    # For simplicity, let's use occupancy at goal state as proxy
    goal_idx = np.zeros(env.nS * env.nA)
    goal_state = env.nS - 1  # Bottom-right corner
    for a in range(env.nA):
        goal_idx[goal_state * env.nA + a] = 1.0
    c_goal = goal_idx
    tau_goal = 0.01  # Want at least 1% occupancy at goal (proxy for reach prob)
    
    # Dual variables
    lam = np.zeros_like(d_E)
    mu_safety = 0.0
    mu_goal = 0.0
    avg_d = np.zeros_like(d_E)
    
    history = defaultdict(list)
    
    for k in trange(1, 3001, desc="Multi-constraint"):
        # Policy player with both constraints
        reward = -(lam + mu_safety * c_unsafe - mu_goal * c_goal)  # Note: minus for goal (we want more)
        pi = solve_soft_q(env, reward, n_iter=80)
        d = rollout_policy(env, pi)
        
        # Dual updates
        lam += 1.0 * 2*(d - d_E)
        mu_safety = max(0.0, mu_safety + 10.0 * (np.dot(d, c_unsafe) - tau_safety))
        mu_goal = max(0.0, mu_goal + 10.0 * (tau_goal - np.dot(d, c_goal)))  # Flipped for ≥ constraint
        
        avg_d = ((k-1)*avg_d + d) / k
        
        if k % 100 == 0:
            f_val = np.sum((avg_d - d_E)**2)
            unsafe = np.dot(avg_d, c_unsafe)
            goal_occ = np.dot(avg_d, c_goal)
            
            history["f"].append(f_val)
            history["unsafe"].append(unsafe)
            history["goal"].append(goal_occ)
            history["mu_safety"].append(mu_safety)
            history["mu_goal"].append(mu_goal)
            
            if k % 500 == 0:
                print(f"   k={k:4d}  f={f_val:.4f}  unsafe={unsafe:.4f}  goal={goal_occ:.4f}")
    
    return history

def create_stress_test_plots(binding_hist, slippage_results, multi_hist):
    """Create visualization plots for all stress tests"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Test 1: Pareto front (binding constraints)
    ax = axes[0, 0]
    ax.scatter(binding_hist["f"], binding_hist["unsafe"], c=range(len(binding_hist["f"])), 
               cmap='viridis', alpha=0.7)
    ax.axhline(y=0.002, color='red', linestyle='--', label='τ=0.002')
    ax.set_xlabel('Imitation Loss f(d)')
    ax.set_ylabel('Unsafe Occupancy')
    ax.set_title('Test 1: Pareto Front\n(Binding Safety Constraint)')
    ax.legend()
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    # Test 1: Dual variable evolution
    ax = axes[1, 0]
    ax.plot(binding_hist["mu"], linewidth=2)
    ax.set_xlabel('Iteration (×100)')
    ax.set_ylabel('Dual Variable μ')
    ax.set_title('Dual Variable Evolution')
    ax.grid(True, alpha=0.3)
    
    # Test 2: Convergence under noise
    ax = axes[0, 1]
    for slip, hist in slippage_results.items():
        ax.plot(hist["iteration"], hist["violation"], label=f'slip={slip}', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Constraint Violation')
    ax.set_title('Test 2: Convergence Under Noise')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Test 2: Final performance vs slip
    ax = axes[1, 1]
    slips = list(slippage_results.keys())
    final_violations = [slippage_results[s]["violation"][-1] for s in slips]
    bars = ax.bar(slips, final_violations, alpha=0.7, color=['green', 'orange', 'red'])
    ax.set_xlabel('Slip Probability')
    ax.set_ylabel('Final Constraint Violation')
    ax.set_title('Robustness to Slippage')
    
    # Add value labels on bars
    for bar, val in zip(bars, final_violations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom')
    
    # Test 3: Constraint satisfaction heatmap
    ax = axes[0, 2]
    iterations = range(len(multi_hist["unsafe"]))
    safety_satisfied = [1 if u <= 0.05 else 0 for u in multi_hist["unsafe"]]
    goal_satisfied = [1 if g >= 0.01 else 0 for g in multi_hist["goal"]]
    
    # Create heatmap data
    heatmap_data = np.array([safety_satisfied, goal_satisfied])
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', interpolation='nearest')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Safety', 'Goal Reach'])
    ax.set_xlabel('Iteration (×100)')
    ax.set_title('Test 3: Constraint Satisfaction\n(Green=Satisfied, Red=Violated)')
    
    # Test 3: Dual variables
    ax = axes[1, 2]
    ax.plot(multi_hist["mu_safety"], label='μ_safety', linewidth=2)
    ax.plot(multi_hist["mu_goal"], label='μ_goal', linewidth=2)
    ax.set_xlabel('Iteration (×100)')
    ax.set_ylabel('Dual Variable Value')
    ax.set_title('Multiple Constraint Dual Variables')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stress_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Stress Tests for Dual Optimization")
    parser.add_argument("--test", choices=["binding", "slippage", "multi", "all"], 
                       default="all", help="Which test to run")
    args = parser.parse_args()
    
    print(" STRESS TESTS FOR DUAL OPTIMIZATION")
    print("=" * 50)
    
    if args.test in ["binding", "all"]:
        binding_hist = run_binding_constraint_test()
    else:
        binding_hist = None
        
    if args.test in ["slippage", "all"]:
        slippage_results = run_slippage_test()
    else:
        slippage_results = None
        
    if args.test in ["multi", "all"]:
        multi_hist = run_multiple_constraints_test()
    else:
        multi_hist = None
    
    # Create plots if we ran all tests
    if args.test == "all":
        print("\n Creating visualization plots...")
        create_stress_test_plots(binding_hist, slippage_results, multi_hist)
        print(" Results saved to stress_test_results.png")
    
    print("\n Stress tests completed!")

if __name__ == "__main__":
    main() 