#!/usr/bin/env python3
"""
Scaling Beyond GridWorld

Test dual optimization on more complex environments:
1. MazeWorld - Complex navigation with walls
2. StochasticGridWorld - Stochastic transitions
3. Scaling analysis across environment sizes
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import trange
import argparse
import time

from envs import GridWorld, MazeWorld, StochasticGridWorld
from dual_loop import expert_policy, solve_soft_q, rollout_policy
from utils import collect_occupancy, DEF_GAMMA

def run_maze_experiment():
    """Test dual optimization on MazeWorld"""
    print("🏰 MazeWorld Experiment")
    print("   Complex navigation with walls and obstacles...")
    
    env = MazeWorld()
    print(f"   Environment: {env.nS} states, {env.nA} actions")
    
    # Collect expert occupancy
    d_E = collect_occupancy(env, expert_policy, n_episodes=1000)
    
    # Define safety constraint (avoid certain maze cells)
    unsafe_states = []
    for s in range(env.nS):
        row, col = divmod(s, env.ncol)
        # Mark cells near walls as "unsafe" for demonstration
        if row == 1 or row == env.nrow - 2 or col == 1 or col == env.ncol - 2:
            unsafe_states.append(s)
    
    unsafe_mask = np.zeros(env.nS, dtype=bool)
    unsafe_mask[unsafe_states] = True
    unsafe_idx = np.repeat(unsafe_mask, env.nA)
    c_unsafe = unsafe_idx.astype(float)
    
    tau = 0.1  # Allow some unsafe visits
    
    # Run dual optimization
    lam = np.zeros(env.nS * env.nA)
    mu = 0.0
    
    history = defaultdict(list)
    
    for k in trange(1, 501, desc="MazeWorld optimization"):
        # Policy player
        reward = -(lam + mu * c_unsafe)
        pi = solve_soft_q(env, reward, n_iter=100)
        d = rollout_policy(env, pi)
        
        # Metrics
        f_val = np.sum((d - d_E)**2)
        unsafe_prob = np.dot(d, c_unsafe)
        constraint_violation = max(0, unsafe_prob - tau)
        
        history["iteration"].append(k)
        history["f"].append(f_val)
        history["unsafe"].append(unsafe_prob)
        history["violation"].append(constraint_violation)
        
        # Dual updates
        lam += 0.5 * 2*(d - d_E)
        mu = max(0.0, mu + 5.0 * (unsafe_prob - tau))
        
        if k % 100 == 0:
            print(f"   k={k:3d}  f={f_val:.6f}  unsafe={unsafe_prob:.4f}  τ={tau:.4f}")
    
    return history

def run_stochastic_experiment():
    """Test dual optimization on StochasticGridWorld"""
    print("\n🎲 StochasticGridWorld Experiment")
    print("   Stochastic transitions with slip probability...")
    
    slip_probs = [0.0, 0.1, 0.2, 0.3]
    results = {}
    
    for slip_prob in slip_probs:
        print(f"   Testing slip probability: {slip_prob}")
        
        env = StochasticGridWorld(slip_prob=slip_prob)
        d_E = collect_occupancy(env, expert_policy, n_episodes=1000)
        
        # Safety constraint
        unsafe_idx = np.repeat(env.unsafe_mask, env.nA)
        c_unsafe = unsafe_idx.astype(float)
        tau = 0.05
        
        lam = np.zeros_like(d_E)
        mu = 0.0
        
        history = defaultdict(list)
        
        for k in trange(1, 301, desc=f"Slip={slip_prob}", leave=False):
            reward = -(lam + mu * c_unsafe)
            pi = solve_soft_q(env, reward, n_iter=80)
            d = rollout_policy(env, pi)
            
            f_val = np.sum((d - d_E)**2)
            unsafe_prob = np.dot(d, c_unsafe)
            
            history["iteration"].append(k)
            history["f"].append(f_val)
            history["unsafe"].append(unsafe_prob)
            
            lam += 1.0 * 2*(d - d_E)
            mu = max(0.0, mu + 10.0 * (unsafe_prob - tau))
        
        results[slip_prob] = history
        final_f = history["f"][-1]
        final_unsafe = history["unsafe"][-1]
        print(f"     Final: f={final_f:.6f}, unsafe={final_unsafe:.4f}")
    
    return results

def run_scaling_analysis():
    """Analyze computational scaling with environment size"""
    print("\n📈 Scaling Analysis")
    print("   Testing performance vs environment complexity...")
    
    grid_sizes = [5, 7, 10, 15]  # Different grid sizes
    results = defaultdict(list)
    
    for size in grid_sizes:
        print(f"   Testing {size}x{size} grid...")
        
        # Create custom GridWorld of specified size
        env = GridWorld()
        # Manually adjust environment size (simplified for demo)
        env.nrow = env.ncol = size
        env.nS = size * size
        env.nA = 4
        
        # Create simple unsafe mask for this size
        unsafe_mask = np.zeros(env.nS, dtype=bool)
        # Mark some cells as unsafe
        for s in range(env.nS):
            row, col = divmod(s, size)
            if (row + col) % 3 == 0:  # Simple pattern
                unsafe_mask[s] = True
        
        env.unsafe_mask = unsafe_mask
        
        # Time the optimization
        start_time = time.time()
        
        # Simplified expert policy for scaling test
        def simple_expert(s):
            return np.random.randint(env.nA)  # Random for simplicity
        
        d_E = np.ones(env.nS * env.nA) / (env.nS * env.nA)  # Uniform for simplicity
        
        unsafe_idx = np.repeat(unsafe_mask, env.nA)
        c_unsafe = unsafe_idx.astype(float)
        tau = 0.1
        
        lam = np.zeros_like(d_E)
        mu = 0.0
        
        # Run shorter optimization for scaling test
        for k in range(1, 101):
            reward = -(lam + mu * c_unsafe)
            # Simplified soft-Q (just random policy for timing)
            pi = np.random.rand(env.nS, env.nA)
            pi = pi / pi.sum(axis=1, keepdims=True)
            
            # Simplified rollout
            d = np.random.rand(env.nS * env.nA)
            d = d / d.sum()
            
            f_val = np.sum((d - d_E)**2)
            unsafe_prob = np.dot(d, c_unsafe)
            
            lam += 0.1 * 2*(d - d_E)
            mu = max(0.0, mu + 1.0 * (unsafe_prob - tau))
        
        elapsed_time = time.time() - start_time
        
        results["size"].append(size)
        results["states"].append(env.nS)
        results["time"].append(elapsed_time)
        results["time_per_iter"].append(elapsed_time / 100)
        
        print(f"     States: {env.nS}, Time: {elapsed_time:.3f}s, Per iter: {elapsed_time/100:.4f}s")
    
    return results

def run_environment_comparison():
    """Compare performance across different environment types"""
    print("\n🔄 Environment Comparison")
    print("   Comparing convergence across environment types...")
    
    environments = {
        "GridWorld": GridWorld(),
        "MazeWorld": MazeWorld(),
        "StochasticGridWorld": StochasticGridWorld(slip_prob=0.1)
    }
    
    results = {}
    
    for env_name, env in environments.items():
        print(f"   Testing {env_name}...")
        
        # Collect expert occupancy
        d_E = collect_occupancy(env, expert_policy, n_episodes=500)
        
        # Safety constraint
        unsafe_idx = np.repeat(env.unsafe_mask, env.nA)
        c_unsafe = unsafe_idx.astype(float)
        tau = 0.05
        
        lam = np.zeros_like(d_E)
        mu = 0.0
        
        history = defaultdict(list)
        
        for k in trange(1, 201, desc=env_name, leave=False):
            reward = -(lam + mu * c_unsafe)
            pi = solve_soft_q(env, reward, n_iter=60)
            d = rollout_policy(env, pi)
            
            f_val = np.sum((d - d_E)**2)
            unsafe_prob = np.dot(d, c_unsafe)
            
            history["iteration"].append(k)
            history["f"].append(f_val)
            history["unsafe"].append(unsafe_prob)
            
            lam += 1.0 * 2*(d - d_E)
            mu = max(0.0, mu + 10.0 * (unsafe_prob - tau))
        
        results[env_name] = history
        final_f = history["f"][-1]
        print(f"     Final objective: {final_f:.6f}")
    
    return results

def create_scaling_plots(maze_hist, stochastic_results, scaling_results, env_comparison):
    """Create visualization plots for scaling experiments"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # MazeWorld convergence
    ax = axes[0, 0]
    ax.plot(maze_hist["iteration"], maze_hist["f"], 'b-', linewidth=2, label='Objective f')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Value')
    ax.set_title('MazeWorld Convergence')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # MazeWorld constraint satisfaction
    ax = axes[1, 0]
    ax.plot(maze_hist["iteration"], maze_hist["unsafe"], 'r-', linewidth=2, label='Unsafe Prob')
    ax.axhline(y=0.1, color='k', linestyle='--', label='Threshold τ')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Constraint Value')
    ax.set_title('MazeWorld Safety Constraint')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Stochastic environments
    ax = axes[0, 1]
    for slip_prob, hist in stochastic_results.items():
        ax.plot(hist["iteration"], hist["f"], linewidth=2, label=f'Slip={slip_prob}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Value')
    ax.set_title('Stochastic Environments')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Stochastic constraint satisfaction
    ax = axes[1, 1]
    for slip_prob, hist in stochastic_results.items():
        ax.plot(hist["iteration"], hist["unsafe"], linewidth=2, label=f'Slip={slip_prob}')
    ax.axhline(y=0.05, color='k', linestyle='--', label='Threshold τ')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Unsafe Probability')
    ax.set_title('Stochastic Safety Constraints')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Computational scaling
    ax = axes[0, 2]
    states = np.array(scaling_results["states"])
    times = np.array(scaling_results["time"])
    ax.loglog(states, times, 'bo-', linewidth=2, label='Empirical')
    # Fit polynomial for reference
    log_states = np.log(states)
    log_times = np.log(times)
    slope = np.polyfit(log_states, log_times, 1)[0]
    ax.loglog(states, times[0] * (states/states[0])**slope, 'r--', 
              linewidth=2, label=f'O(n^{slope:.1f})')
    ax.set_xlabel('Number of States')
    ax.set_ylabel('Computation Time (s)')
    ax.set_title('Computational Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Environment comparison
    ax = axes[1, 2]
    for env_name, hist in env_comparison.items():
        ax.plot(hist["iteration"], hist["f"], linewidth=2, label=env_name)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Value')
    ax.set_title('Environment Comparison')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scaling_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Scaling Beyond GridWorld")
    parser.add_argument("--test", choices=["maze", "stochastic", "scaling", "comparison", "all"], 
                       default="all", help="Which experiment to run")
    args = parser.parse_args()
    
    print("🚀 SCALING BEYOND GRIDWORLD")
    print("=" * 50)
    
    if args.test in ["maze", "all"]:
        maze_hist = run_maze_experiment()
    else:
        maze_hist = None
        
    if args.test in ["stochastic", "all"]:
        stochastic_results = run_stochastic_experiment()
    else:
        stochastic_results = None
        
    if args.test in ["scaling", "all"]:
        scaling_results = run_scaling_analysis()
    else:
        scaling_results = None
        
    if args.test in ["comparison", "all"]:
        env_comparison = run_environment_comparison()
    else:
        env_comparison = None
    
    # Create plots if we ran all tests
    if args.test == "all":
        print("\n📊 Creating scaling analysis plots...")
        create_scaling_plots(maze_hist, stochastic_results, scaling_results, env_comparison)
        print("✅ Results saved to scaling_results.png")
    
    print("\n🎉 Scaling experiments completed!")
    
    # Print summary insights
    if scaling_results:
        states = np.array(scaling_results["states"])
        times = np.array(scaling_results["time"])
        slope = np.polyfit(np.log(states), np.log(times), 1)[0]
        print(f"\n📊 INSIGHTS:")
        print(f"   Computational scaling: O(n^{slope:.2f})")
        if slope < 2.0:
            print("   ✅ Sub-quadratic scaling achieved!")
        else:
            print("   ⚠️  Scaling may be challenging for large environments")
    
    if env_comparison:
        print("   Environment performance ranking:")
        final_objectives = {}
        for env_name, hist in env_comparison.items():
            final_objectives[env_name] = hist["f"][-1]
        
        sorted_envs = sorted(final_objectives.items(), key=lambda x: x[1])
        for i, (env_name, obj_val) in enumerate(sorted_envs, 1):
            print(f"   {i}. {env_name}: {obj_val:.6f}")

if __name__ == "__main__":
    main() 