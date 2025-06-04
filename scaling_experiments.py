#!/usr/bin/env python3
"""
Scaling Beyond GridWorld

Test dual optimization on more complex environments:
1. MazeWorld - Complex navigation with walls
2. StochasticGridWorld - Stochastic transitions
3. Scaling analysis across environment sizes
4. DynamicGridWorld - Hazards change periodically
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import trange
import argparse
import time

from envs import GridWorld, MazeWorld, StochasticGridWorld, DynamicGridWorld # Added DynamicGridWorld
from dual_loop import expert_policy, solve_soft_q, rollout_policy
from utils import collect_occupancy, DEF_GAMMA

def run_maze_experiment():
    """Test dual optimization on MazeWorld"""
    print("🏰 MazeWorld Experiment")
    print("   Complex navigation with walls and obstacles...")
    
    env = MazeWorld(size=7) # Using default size 7 from envs.py
    # Ensure expert_policy is compatible or MazeWorld has env.size attribute for collect_occupancy
    # MazeWorld has self.size, so collect_occupancy(env, expert_policy,...) is fine.
    print(f"   Environment: {env.nS} states, {env.nA} actions")
    
    d_E = collect_occupancy(env, expert_policy, n_episodes=1000)
    
    # Define safety constraint (avoid certain maze cells)
    # In MazeWorld, unsafe_mask is already defined based on complexity.
    # Let's use the env's predefined unsafe_mask.
    unsafe_idx = np.repeat(env.unsafe_mask, env.nA)
    c_unsafe = unsafe_idx.astype(float)
    
    tau = 0.1
    
    lam = np.zeros(env.nS * env.nA)
    mu = 0.0
    avg_d = np.zeros_like(d_E) # For calculating average performance metrics over time
    
    history = defaultdict(list)
    
    for k in trange(1, 501, desc="MazeWorld optimization"):
        reward = -(lam + mu * c_unsafe)
        pi = solve_soft_q(env, reward, n_iter=100)
        d_k = rollout_policy(env, pi) # Policy for current iter k
        
        avg_d = ((k-1)*avg_d + d_k) / k # Update running average

        # Metrics based on avg_d
        f_val = np.sum((avg_d - d_E)**2)
        unsafe_prob = np.dot(avg_d, c_unsafe)
        constraint_violation = max(0, unsafe_prob - tau)
        
        history["iteration"].append(k)
        history["f"].append(f_val)
        history["unsafe"].append(unsafe_prob)
        history["violation"].append(constraint_violation)
        
        # Dual updates use d_k (current policy's occupancy)
        lam += 0.5 * 2*(d_k - d_E)
        mu = max(0.0, mu + 5.0 * (np.dot(d_k, c_unsafe) - tau))
        
        if k % 100 == 0:
            print(f"   k={k:3d}  f={f_val:.6f}  unsafe={unsafe_prob:.4f}  τ={tau:.4f}")
    
    return history

def run_stochastic_experiment():
    """Test dual optimization on StochasticGridWorld"""
    print("\n🎲 StochasticGridWorld Experiment")
    print("   Stochastic transitions with slip probability...")
    
    slip_probs = [0.0, 0.1, 0.2] # Reduced for quicker run if needed
    results = {}
    
    for slip_prob in slip_probs:
        print(f"   Testing slip probability: {slip_prob}")
        
        env = StochasticGridWorld(slip=slip_prob, size=5) # Ensure size for expert_policy
        d_E = collect_occupancy(env, expert_policy, n_episodes=1000)
        
        unsafe_idx = np.repeat(env.unsafe_mask, env.nA)
        c_unsafe = unsafe_idx.astype(float)
        tau = 0.05
        
        lam = np.zeros_like(d_E)
        mu = 0.0
        avg_d = np.zeros_like(d_E)
        
        history = defaultdict(list)
        
        for k in trange(1, 301, desc=f"Slip={slip_prob}", leave=False):
            reward = -(lam + mu * c_unsafe)
            pi = solve_soft_q(env, reward, n_iter=80)
            d_k = rollout_policy(env, pi)

            avg_d = ((k-1)*avg_d + d_k) / k
            
            f_val = np.sum((avg_d - d_E)**2)
            unsafe_prob = np.dot(avg_d, c_unsafe)
            
            history["iteration"].append(k)
            history["f"].append(f_val)
            history["unsafe"].append(unsafe_prob)
            
            lam += 1.0 * 2*(d_k - d_E)
            mu = max(0.0, mu + 10.0 * (np.dot(d_k, c_unsafe) - tau))
        
        results[slip_prob] = history
        final_f = history["f"][-1]
        final_unsafe = history["unsafe"][-1]
        print(f"     Final: f={final_f:.6f}, unsafe={final_unsafe:.4f}")
    
    return results

def run_scaling_analysis():
    """Analyze computational scaling with environment size"""
    print("\n📈 Scaling Analysis")
    print("   Testing performance vs environment complexity...")
    
    grid_sizes = [5, 7, 10]  # Different grid sizes, 15 can be slow
    results = defaultdict(list)
    
    for size in grid_sizes:
        print(f"   Testing {size}x{size} grid...")
        
        env = GridWorld(size=size)
        # The GridWorld constructor in envs.py correctly sets nS, nA based on size.
        # Unsafe mask is also set, but let's redefine for scaling test consistency if needed.
        # For this test, the default unsafe_mask from GridWorld(size=size) is fine.
        
        start_time = time.time()
        
        # Simplified expert policy for scaling test
        def simple_expert(s, current_size): # Must accept size from collect_occupancy
            # env is in scope here, but current_size is passed by collect_occupancy
            return np.random.randint(env.nA) 
        
        # d_E for scaling test can be uniform or based on simple_expert
        d_E = collect_occupancy(env, simple_expert, n_episodes=100)
        
        unsafe_idx = np.repeat(env.unsafe_mask, env.nA)
        c_unsafe = unsafe_idx.astype(float)
        tau = 0.1
        
        lam = np.zeros_like(d_E)
        mu = 0.0
        
        for k_iter in range(1, 101): # Fixed iterations for timing
            reward = -(lam + mu * c_unsafe)
            pi_flat = np.random.rand(env.nS * env.nA) # Simplified policy generation
            pi = (pi_flat / pi_flat.sum(axis=0, keepdims=True)).reshape(env.nS, env.nA) # Dummy policy
            # Correcting pi generation for scaling test:
            pi_rand = np.random.rand(env.nS, env.nA)
            pi = pi_rand / pi_rand.sum(axis=1, keepdims=True)

            d_k_flat = np.random.rand(env.nS * env.nA) # Simplified rollout
            d_k = d_k_flat / d_k_flat.sum()
            
            lam += 0.1 * 2*(d_k - d_E)
            mu = max(0.0, mu + 1.0 * (np.dot(d_k, c_unsafe) - tau))
        
        elapsed_time = time.time() - start_time
        
        results["size"].append(size)
        results["states"].append(env.nS)
        results["time"].append(elapsed_time)
        results["time_per_iter"].append(elapsed_time / 100)
        
        print(f"     States: {env.nS}, Time: {elapsed_time:.3f}s, Per iter: {elapsed_time/100:.4f}s")
    
    return results

def run_environment_comparison():
    """Compare performance across different environment types"""
    print("\n🔄 Environment Type Comparison") # Renamed to avoid conflict
    print("   Comparing convergence across environment types...")
    
    environments_setup = {
        "GridWorld": {"class": GridWorld, "args": {"size": 5}},
        "MazeWorld": {"class": MazeWorld, "args": {"size": 7, "complexity": "medium"}}, # Maze default size is 7
        "StochasticGridWorld": {"class": StochasticGridWorld, "args": {"size": 5, "slip": 0.1}}
    }
    
    results = {}
    
    for env_name, setup in environments_setup.items():
        print(f"   Testing {env_name}...")
        env = setup["class"](**setup["args"])
        
        d_E = collect_occupancy(env, expert_policy, n_episodes=500)
        
        unsafe_idx = np.repeat(env.unsafe_mask, env.nA)
        c_unsafe = unsafe_idx.astype(float)
        tau = 0.05
        
        lam = np.zeros_like(d_E)
        mu = 0.0
        avg_d = np.zeros_like(d_E)
        
        history = defaultdict(list)
        
        for k in trange(1, 201, desc=env_name, leave=False):
            reward = -(lam + mu * c_unsafe)
            pi = solve_soft_q(env, reward, n_iter=60)
            d_k = rollout_policy(env, pi)

            avg_d = ((k-1)*avg_d + d_k) / k
            
            f_val = np.sum((avg_d - d_E)**2)
            unsafe_prob = np.dot(avg_d, c_unsafe)
            
            history["iteration"].append(k)
            history["f"].append(f_val)
            history["unsafe"].append(unsafe_prob)
            
            lam += 1.0 * 2*(d_k - d_E)
            mu = max(0.0, mu + 10.0 * (np.dot(d_k, c_unsafe) - tau))
        
        results[env_name] = history
        final_f = history["f"][-1]
        print(f"     Final objective: {final_f:.6f}")
    
    return results

def run_dynamic_env_experiment():
    """Test dual optimization on DynamicGridWorld"""
    print("\n🌪️ DynamicGridWorld Experiment") # Changed icon
    print("   Hazards change periodically...")
    
    env = DynamicGridWorld(size=5, change_frequency=150) # Hazards change every 150 internal env steps
    print(f"   Environment: {env.nS} states, {env.nA} actions, hazards change every {env.change_frequency} agent steps.")
    
    # Expert policy aims for the goal, agnostic to dynamic hazards for d_E definition.
    # Use a stable GridWorld of the same size to define d_E.
    static_env_for_expert = GridWorld(size=env.size)
    d_E = collect_occupancy(static_env_for_expert, expert_policy, n_episodes=1000)
    
    tau = 0.05  # Safety threshold

    lam = np.zeros(env.nS * env.nA)
    mu = 0.0
    avg_d = np.zeros_like(d_E)
    
    history = defaultdict(list)
    
    # Store the hash/ID of the previous unsafe_mask configuration to detect changes.
    # Using a sum of boolean mask as a simple ID. Could be more robust.
    previous_unsafe_mask_id = np.sum(env.unsafe_mask) 

    for k in trange(1, 801, desc="DynamicWorld optimization"): # Run for more iterations
        # IMPORTANT: Update c_unsafe based on the CURRENT state of env.unsafe_mask
        current_unsafe_mask = env.unsafe_mask.copy() # Get current mask from env
        unsafe_idx = np.repeat(current_unsafe_mask, env.nA)
        c_unsafe_this_iter = unsafe_idx.astype(float) # Use this for reward and dual update

        current_unsafe_mask_id = np.sum(current_unsafe_mask)
        hazard_config_changed_this_iter = (current_unsafe_mask_id != previous_unsafe_mask_id)
        if hazard_config_changed_this_iter:
            # print(f"Iter {k}: Hazard configuration changed. Old ID: {previous_unsafe_mask_id}, New ID: {current_unsafe_mask_id}")
            # print(f"   New unsafe cells: {np.where(current_unsafe_mask)[0].tolist()}")
            pass # Suppress print in loop for cleaner tqdm
        previous_unsafe_mask_id = current_unsafe_mask_id

        # Policy player
        reward = -(lam + mu * c_unsafe_this_iter) # Use c_unsafe for this iteration
        pi = solve_soft_q(env, reward, n_iter=80) # env is DynamicGridWorld
        
        # Rollout policy. env.unsafe_mask can change *during* this rollout.
        d_k = rollout_policy(env, pi, n_episodes=30) # Fewer episodes for faster iteration
        
        # Dual updates use c_unsafe_this_iter
        lam_grad = 2 * (d_k - d_E)
        mu_grad = np.dot(d_k, c_unsafe_this_iter) - tau
        
        lam += 1.0 * lam_grad
        mu = max(0.0, mu + 10.0 * mu_grad)
        
        avg_d = ((k-1)*avg_d + d_k) / k
        
        f_val = np.sum((avg_d - d_E)**2)
        # Safety w.r.t c_unsafe_this_iter (what the agent was trying to adhere to for this iter)
        unsafe_prob_iter = np.dot(avg_d, c_unsafe_this_iter) 
        
        history["iteration"].append(k)
        history["f"].append(f_val)
        history["unsafe"].append(unsafe_prob_iter) # Log safety based on this iter's known constraints
        history["mu"].append(mu)
        history["hazard_config_changed"].append(1 if hazard_config_changed_this_iter else 0) # Store as 0 or 1
        
        if k % 100 == 0:
            print(f"   k={k:4d}  f={f_val:.6f}  unsafe={unsafe_prob_iter:.4f}  μ={mu:.2f} Changed={hazard_config_changed_this_iter}")
            if hazard_config_changed_this_iter:
                 print(f"     Iter {k}: Hazard configuration changed. Unsafe cells: {np.where(current_unsafe_mask)[0].tolist()}")
    
    return history


def create_scaling_plots(maze_hist, stochastic_results, scaling_results, env_comparison_results, dynamic_results):
    """Create visualization plots for scaling experiments"""
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 18)) # Adjusted for 6 plots
    fig.suptitle("Scaling Experiments: Performance and Robustness Analysis", fontsize=16)

    # MazeWorld convergence
    ax = axes[0, 0]
    if maze_hist:
        ax.plot(maze_hist["iteration"], maze_hist["f"], 'b-', linewidth=2, label='Objective f')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Value')
        ax.set_title('MazeWorld Convergence')
        ax.set_yscale('log')
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    # MazeWorld constraint satisfaction
    ax = axes[1, 0] # Moved to below MazeWorld convergence
    if maze_hist:
        ax.plot(maze_hist["iteration"], maze_hist["unsafe"], 'r-', linewidth=2, label='Unsafe Prob')
        ax.axhline(y=0.1, color='k', linestyle='--', label='Threshold τ=0.1') # Maze tau is 0.1
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Constraint Value (Unsafe)')
        ax.set_title('MazeWorld Safety Constraint')
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Stochastic environments - Objective
    ax = axes[0, 1]
    if stochastic_results:
        for slip_prob, hist in stochastic_results.items():
            ax.plot(hist["iteration"], hist["f"], linewidth=2, label=f'Slip={slip_prob}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Value')
    ax.set_title('Stochastic Environments: Objective')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Stochastic environments - Constraint
    ax = axes[1, 1] # Moved to below Stochastic Objective
    if stochastic_results:
        for slip_prob, hist in stochastic_results.items():
            ax.plot(hist["iteration"], hist["unsafe"], linewidth=2, label=f'Slip={slip_prob}')
        ax.axhline(y=0.05, color='k', linestyle='--', label='Threshold τ=0.05') # Stochastic tau is 0.05
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Unsafe Probability')
    ax.set_title('Stochastic Environments: Safety')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Computational scaling
    ax = axes[2, 0] # Bottom left
    if scaling_results and scaling_results["states"]:
        states = np.array(scaling_results["states"])
        times = np.array(scaling_results["time"])
        ax.loglog(states, times, 'bo-', linewidth=2, label='Empirical Time')
        if len(states) > 1:
            log_states = np.log(states)
            log_times = np.log(times)
            slope = np.polyfit(log_states, log_times, 1)[0]
            ax.loglog(states, np.exp(log_times[0] + slope * (log_states - log_states[0])), 'r--', 
                      linewidth=2, label=f'Fit: O(N^{slope:.2f})')
    ax.set_xlabel('Number of States (log scale)')
    ax.set_ylabel('Computation Time (s) (log scale)')
    ax.set_title('Computational Scaling Analysis')
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.3)
    
    # Dynamic Environment Performance
    ax = axes[2, 1] # Bottom right
    if dynamic_results:
        ax.plot(dynamic_results["iteration"], dynamic_results["f"], 'm-', linewidth=2, label='Objective f (Dynamic)')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Value (log scale)', color='m')
        ax.tick_params(axis='y', labelcolor='m')
        ax.set_title('DynamicGridWorld Performance')
        ax.set_yscale('log')
        ax.legend(loc='upper left')

        ax_twin = ax.twinx()
        ax_twin.plot(dynamic_results["iteration"], dynamic_results["unsafe"], 'c-', linewidth=2, label='Unsafe Prob (Dynamic)')
        ax_twin.axhline(y=0.05, color='k', linestyle=':', label='τ=0.05')
        ax_twin.set_ylabel('Unsafe Probability', color='c')
        ax_twin.tick_params(axis='y', labelcolor='c')
        ax_twin.legend(loc='upper right')
        
        # Mark hazard change points
        change_iters = [dynamic_results["iteration"][i] for i, changed in enumerate(dynamic_results["hazard_config_changed"]) if changed and i > 0]
        for xc in change_iters:
            ax.axvline(x=xc, color='gray', linestyle='--', linewidth=1, alpha=0.8)
        if change_iters: # Add a dummy entry for legend
             ax.plot([], [], color='gray', linestyle='--', label='Hazard Change')
        ax.legend(loc='center left')


    # Environment comparison (Plotting this separately or finding another spot)
    # For now, remove env_comparison from this main plot to keep it 3x2
    # if env_comparison_results:
    #     ax_comp = fig.add_subplot(3,3,7) # Or another arrangement
    #     for env_name, hist in env_comparison_results.items():
    #         ax_comp.plot(hist["iteration"], hist["f"], linewidth=2, label=env_name)
    #     ax_comp.set_xlabel('Iteration')
    #     ax_comp.set_ylabel('Objective Value')
    #     ax_comp.set_title('Environment Comparison')
    #     ax_comp.set_yscale('log')
    #     ax_comp.legend()
    #     ax_comp.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect to make space for suptitle
    plt.savefig('scaling_results.png', dpi=300, bbox_inches='tight')
    # plt.show() # Usually called by the calling script.

def main():
    parser = argparse.ArgumentParser(description="Scaling Beyond GridWorld")
    parser.add_argument("--test", choices=["maze", "stochastic", "scaling", "comparison", "dynamic", "all"], 
                       default="all", help="Which experiment to run")
    args = parser.parse_args()
    
    print("🚀 SCALING BEYOND GRIDWORLD")
    print("=" * 50)
    
    maze_hist, stochastic_results, scaling_results_data, env_comp_results, dynamic_env_hist = None, None, None, None, None

    if args.test in ["maze", "all"]:
        maze_hist = run_maze_experiment()
        
    if args.test in ["stochastic", "all"]:
        stochastic_results = run_stochastic_experiment()
        
    if args.test in ["scaling", "all"]:
        scaling_results_data = run_scaling_analysis()
        
    if args.test in ["comparison", "all"]:
        env_comp_results = run_environment_comparison()

    if args.test in ["dynamic", "all"]:
        dynamic_env_hist = run_dynamic_env_experiment()
            
    if args.test == "all" or any([maze_hist, stochastic_results, scaling_results_data, env_comp_results, dynamic_env_hist]):
        print("\n📊 Creating scaling analysis plots...")
        # Pass None if a specific test wasn't run
        create_scaling_plots(
            maze_hist if maze_hist else defaultdict(list), 
            stochastic_results if stochastic_results else {}, 
            scaling_results_data if scaling_results_data else defaultdict(list), 
            env_comp_results if env_comp_results else {},
            dynamic_env_hist if dynamic_env_hist else defaultdict(list)
        )
        print("✅ Results saved to scaling_results.png")
        plt.show() # Show plot after saving
    
    print("\n🎉 Scaling experiments completed!")
    
    if scaling_results_data and scaling_results_data["states"]:
        states = np.array(scaling_results_data["states"])
        times = np.array(scaling_results_data["time"])
        if len(states) > 1 :
            slope = np.polyfit(np.log(states), np.log(times), 1)[0]
            print(f"\n📊 INSIGHTS (Scaling):")
            print(f"   Computational scaling: O(N^{slope:.2f}) where N is number of states.")

    if env_comp_results:
        print("\n📊 INSIGHTS (Environment Comparison):")
        print("   Environment performance ranking (lower final objective is better):")
        final_objectives = {}
        for env_name, hist in env_comp_results.items():
            if hist["f"]:
                 final_objectives[env_name] = hist["f"][-1]
        
        sorted_envs = sorted(final_objectives.items(), key=lambda x: x[1])
        for i, (env_name, obj_val) in enumerate(sorted_envs, 1):
            print(f"   {i}. {env_name}: {obj_val:.6f}")

if __name__ == "__main__":
    main()