#!/usr/bin/env python3
"""
Scaling Beyond GridWorld

Test dual optimization on more complex environments:
1. MazeWorld - Complex navigation with walls
2. StochasticGridWorld - Stochastic transitions
3. Scaling analysis across environment sizes
4. DynamicGridWorld - Hazards change periodically (NOW WITH HOSTILE ALTERNATING HAZARDS)
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import trange
import argparse
import time

from envs import GridWorld, MazeWorld, StochasticGridWorld, DynamicGridWorld
from dual_loop import expert_policy, solve_soft_q, rollout_policy
from utils import collect_occupancy, DEF_GAMMA

TAU_DYNAMIC_EXPERIMENT = 0.01

# --- (Keep all run_..._experiment functions as they were in the last complete version) ---
def run_maze_experiment():
    # ... (content of run_maze_experiment, unchanged from your last full version) ...
    print("🏰 MazeWorld Experiment")
    print("   Complex navigation with walls and obstacles...")
    
    env = MazeWorld(size=7)
    print(f"   Environment: {env.nS} states, {env.nA} actions")
    
    d_E = collect_occupancy(env, expert_policy, n_episodes=1000)
    
    unsafe_idx = np.repeat(env.unsafe_mask, env.nA)
    c_unsafe = unsafe_idx.astype(float)
    
    tau = 0.1
    
    lam = np.zeros(env.nS * env.nA)
    mu = 0.0
    avg_d = np.zeros_like(d_E)
    
    history = defaultdict(list)
    
    for k in trange(1, 501, desc="MazeWorld optimization"):
        reward = -(lam + mu * c_unsafe)
        pi = solve_soft_q(env, reward, n_iter=100)
        d_k = rollout_policy(env, pi)
        
        avg_d = ((k-1)*avg_d + d_k) / k

        f_val = np.sum((avg_d - d_E)**2)
        unsafe_prob = np.dot(avg_d, c_unsafe)
        constraint_violation = max(0, unsafe_prob - tau)
        
        history["iteration"].append(k)
        history["f"].append(f_val)
        history["unsafe"].append(unsafe_prob)
        history["violation"].append(constraint_violation)
        
        lam += 0.5 * 2*(d_k - d_E)
        mu = max(0.0, mu + 5.0 * (np.dot(d_k, c_unsafe) - tau))
        
        if k % 100 == 0:
            print(f"   k={k:3d}  f={f_val:.6f}  unsafe={unsafe_prob:.4f}  τ={tau:.4f}")
    
    return history

def run_stochastic_experiment():
    # ... (content of run_stochastic_experiment, unchanged from your last full version) ...
    print("\n🎲 StochasticGridWorld Experiment")
    print("   Stochastic transitions with slip probability...")
    
    slip_probs = [0.0, 0.1, 0.2]
    results = {}
    tau_stochastic = 0.05
    
    for slip_prob in slip_probs:
        print(f"   Testing slip probability: {slip_prob}")
        
        env = StochasticGridWorld(slip=slip_prob, size=5)
        d_E = collect_occupancy(env, expert_policy, n_episodes=1000)
        
        unsafe_idx = np.repeat(env.unsafe_mask, env.nA)
        c_unsafe = unsafe_idx.astype(float)
        
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
            mu = max(0.0, mu + 10.0 * (np.dot(d_k, c_unsafe) - tau_stochastic))
        
        results[slip_prob] = history
        final_f = history["f"][-1]
        final_unsafe = history["unsafe"][-1]
        print(f"     Final: f={final_f:.6f}, unsafe={final_unsafe:.4f}")
    
    return results, tau_stochastic

def run_scaling_analysis():
    # ... (content of run_scaling_analysis, unchanged from your last full version) ...
    print("\n📈 Scaling Analysis")
    print("   Testing performance vs environment complexity...")
    
    grid_sizes = [5, 7, 10]
    results = defaultdict(list)
    
    for size_val in grid_sizes:
        print(f"   Testing {size_val}x{size_val} grid...")
        
        env = GridWorld(size=size_val)
        
        start_time = time.time()
        
        def simple_expert(s, current_size):
            return np.random.randint(env.nA) 
        
        d_E = collect_occupancy(env, simple_expert, n_episodes=100)
        
        unsafe_idx = np.repeat(env.unsafe_mask, env.nA)
        c_unsafe = unsafe_idx.astype(float)
        tau = 0.1
        
        lam = np.zeros_like(d_E)
        mu = 0.0
        
        for k_iter in range(1, 101):
            reward = -(lam + mu * c_unsafe)
            pi_rand = np.random.rand(env.nS, env.nA)
            pi = pi_rand / pi_rand.sum(axis=1, keepdims=True)

            d_k_flat = np.random.rand(env.nS * env.nA)
            d_k = d_k_flat / d_k_flat.sum()
            
            lam += 0.1 * 2*(d_k - d_E)
            mu = max(0.0, mu + 1.0 * (np.dot(d_k, c_unsafe) - tau))
        
        elapsed_time = time.time() - start_time
        
        results["size"].append(size_val)
        results["states"].append(env.nS)
        results["time"].append(elapsed_time)
        results["time_per_iter"].append(elapsed_time / 100)
        
        print(f"     States: {env.nS}, Time: {elapsed_time:.3f}s, Per iter: {elapsed_time/100:.4f}s")
    
    return results

def run_environment_comparison():
    # ... (content of run_environment_comparison, unchanged from your last full version) ...
    print("\n🔄 Environment Type Comparison")
    print("   Comparing convergence across environment types...")
    
    environments_setup = {
        "GridWorld": {"class": GridWorld, "args": {"size": 5}},
        "MazeWorld": {"class": MazeWorld, "args": {"size": 7, "complexity": "medium"}},
        "StochasticGridWorld": {"class": StochasticGridWorld, "args": {"size": 5, "slip": 0.1}}
    }
    results = {}
    tau_env_comp = 0.05
    
    for env_name, setup in environments_setup.items():
        print(f"   Testing {env_name}...")
        env = setup["class"](**setup["args"])
        
        d_E = collect_occupancy(env, expert_policy, n_episodes=500)
        
        unsafe_idx = np.repeat(env.unsafe_mask, env.nA)
        c_unsafe = unsafe_idx.astype(float)
        
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
            mu = max(0.0, mu + 10.0 * (np.dot(d_k, c_unsafe) - tau_env_comp))
        
        results[env_name] = history
        final_f = history["f"][-1]
        print(f"     Final objective: {final_f:.6f}")
    
    return results

def run_dynamic_env_experiment():
    # ... (content of run_dynamic_env_experiment, unchanged from your last full version) ...
    print("\n🌪️ DynamicGridWorld Experiment (Alternating Hostile Hazards)")
    print("   Hazards change periodically between predefined hostile configurations.")
    
    tau = TAU_DYNAMIC_EXPERIMENT
    print(f"   Using safety threshold tau = {tau}")

    env = DynamicGridWorld(size=5, change_frequency=200, hazard_mode="alternating_hostile")
    env.total_agent_steps = 0 

    print(f"   Environment: {env.nS} states, {env.nA} actions, hazards change every {env.change_frequency} agent steps.")
    
    static_env_for_expert = GridWorld(size=env.size)
    d_E = collect_occupancy(static_env_for_expert, expert_policy, n_episodes=1000)
    
    lam = np.zeros(env.nS * env.nA)
    mu = 0.0
    avg_d = np.zeros_like(d_E)
    
    history = defaultdict(list)
    previous_unsafe_mask_tuple = tuple(np.where(env.unsafe_mask)[0])

    for k in trange(1, 1201, desc="Hostile DynamicWorld"): 
        current_unsafe_mask_actual = env.unsafe_mask.copy() 
        unsafe_idx = np.repeat(current_unsafe_mask_actual, env.nA)
        c_unsafe_this_iter = unsafe_idx.astype(float)

        current_unsafe_mask_tuple = tuple(np.where(current_unsafe_mask_actual)[0])
        hazard_config_changed_this_iter = (current_unsafe_mask_tuple != previous_unsafe_mask_tuple)
        previous_unsafe_mask_tuple = current_unsafe_mask_tuple

        reward = -(lam + mu * c_unsafe_this_iter)
        pi = solve_soft_q(env, reward, n_iter=80) 
        d_k = rollout_policy(env, pi, n_episodes=30) 
                                                  
        lam_grad = 2 * (d_k - d_E)
        mu_constraint_value = np.dot(d_k, c_unsafe_this_iter) 
        mu_grad_raw = mu_constraint_value - tau
        
        lam += 1.0 * lam_grad
        mu = max(0.0, mu + 10.0 * mu_grad_raw) 
        
        avg_d = ((k-1)*avg_d + d_k) / k
        
        f_val = np.sum((avg_d - d_E)**2)
        unsafe_prob_avg = np.dot(avg_d, c_unsafe_this_iter) 
        
        history["iteration"].append(k)
        history["f"].append(f_val)
        history["unsafe_avg"].append(unsafe_prob_avg)
        history["unsafe_instantaneous"].append(mu_constraint_value)
        history["mu"].append(mu)
        history["mu_grad_raw"].append(mu_grad_raw)
        history["hazard_config_changed"].append(1 if hazard_config_changed_this_iter else 0)
        
        if k % 50 == 0: 
            print(f"   k={k:4d} f={f_val:.4f} unsafe_avg={unsafe_prob_avg:.4f} unsafe_inst={mu_constraint_value:.4f} μ={mu:.2f} μ_grad_raw={mu_grad_raw:.4f} Changed={hazard_config_changed_this_iter} UnsafeNow={current_unsafe_mask_tuple}")
    
    return history, tau


# Helper function for moving average (for plotting only)
def moving_average(data, window_size):
    if not data or window_size <= 0:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def create_scaling_plots(maze_hist, stochastic_results_tuple, scaling_results_data, 
                         env_comparison_results, dynamic_results_tuple,
                         is_dynamic_test_only=False): # New flag
    """Create visualization plots for scaling experiments"""
    
    stochastic_results, tau_stochastic = stochastic_results_tuple if stochastic_results_tuple else ({}, 0.05)
    dynamic_results, tau_dynamic = dynamic_results_tuple if dynamic_results_tuple else (defaultdict(list), TAU_DYNAMIC_EXPERIMENT)

    if is_dynamic_test_only:
        # Create a figure with 2 subplots, one above the other
        fig, axes = plt.subplots(2, 1, figsize=(10, 12)) # Adjusted for 2 vertical plots
        fig.suptitle("Dynamic Environment: Adaptive Safety Performance", fontsize=16)
        ax_dynamic_detail = axes[0]
        ax_dynamic_main = axes[1]
    else:
        # Original 3x2 layout
        fig, axes_orig = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle("Scaling Experiments: Performance and Robustness Analysis", fontsize=16)
        # --- Maze Plots (axes_orig[0,0] and axes_orig[1,0]) ---
        # ... (maze plotting code as before, checking if maze_hist has data)
        ax_maze_f = axes_orig[0,0]
        if maze_hist and maze_hist["iteration"]:
            ax_maze_f.plot(maze_hist["iteration"], maze_hist["f"], 'b-', linewidth=2, label='Objective f')
            ax_maze_f.set_title('MazeWorld Convergence'); ax_maze_f.set_xlabel('Iteration'); ax_maze_f.set_ylabel('Objective Value'); ax_maze_f.set_yscale('log'); ax_maze_f.legend(); ax_maze_f.grid(True, alpha=0.3)
        else:
            ax_maze_f.text(0.5, 0.5, 'No Maze Data', ha='center', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.5)); ax_maze_f.grid(True, alpha=0.3)

        ax_maze_s = axes_orig[1,0]
        if maze_hist and maze_hist["iteration"]:
            ax_maze_s.plot(maze_hist["iteration"], maze_hist["unsafe"], 'r-', linewidth=2, label='Unsafe Prob (Avg)')
            ax_maze_s.axhline(y=0.1, color='k', linestyle='--', label='Threshold τ=0.1')
            ax_maze_s.set_title('MazeWorld Safety'); ax_maze_s.set_xlabel('Iteration'); ax_maze_s.set_ylabel('Unsafe Prob'); ax_maze_s.legend(); ax_maze_s.grid(True, alpha=0.3)
        else:
            ax_maze_s.text(0.5, 0.5, 'No Maze Data', ha='center', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.5)); ax_maze_s.grid(True, alpha=0.3)

        # --- Stochastic Plots (axes_orig[0,1] and axes_orig[1,1]) ---
        # ... (stochastic plotting code as before, checking for data)
        ax_stoch_f = axes_orig[0,1]
        if stochastic_results:
            # ... (full stochastic objective plot)
            has_data = any(hist.get("iteration") for hist in stochastic_results.values())
            if has_data:
                for slip_prob, hist in stochastic_results.items(): ax_stoch_f.plot(hist["iteration"], hist["f"], linewidth=2, label=f'Slip={slip_prob}')
                ax_stoch_f.set_title('Stochastic Env: Objective'); ax_stoch_f.set_xlabel('Iteration'); ax_stoch_f.set_ylabel('Objective (log)'); ax_stoch_f.set_yscale('log'); ax_stoch_f.legend()
            else: ax_stoch_f.text(0.5, 0.5, 'No Stochastic Data', ha='center', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.5))
        else: ax_stoch_f.text(0.5, 0.5, 'No Stochastic Data', ha='center', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.5))
        ax_stoch_f.grid(True, alpha=0.3)

        ax_stoch_s = axes_orig[1,1]
        if stochastic_results:
            # ... (full stochastic safety plot)
            has_data = any(hist.get("iteration") for hist in stochastic_results.values())
            if has_data:
                for slip_prob, hist in stochastic_results.items(): ax_stoch_s.plot(hist["iteration"], hist["unsafe"], linewidth=2, label=f'Slip={slip_prob}')
                ax_stoch_s.axhline(y=tau_stochastic, color='k', linestyle='--', label=f'τ={tau_stochastic}')
                ax_stoch_s.set_title('Stochastic Env: Safety'); ax_stoch_s.set_xlabel('Iteration'); ax_stoch_s.set_ylabel('Unsafe Prob'); ax_stoch_s.legend()
            else: ax_stoch_s.text(0.5, 0.5, 'No Stochastic Data', ha='center', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.5))
        else: ax_stoch_s.text(0.5, 0.5, 'No Stochastic Data', ha='center', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.5))
        ax_stoch_s.grid(True, alpha=0.3)
        
        ax_dynamic_detail = axes_orig[2, 0] # Use bottom-left for dynamic details in full plot
        ax_dynamic_main = axes_orig[2, 1]   # Use bottom-right for dynamic main in full plot

    # --- Dynamic Environment Plots ---
    if dynamic_results and dynamic_results["iteration"]:
        iterations = dynamic_results["iteration"]
        # Moving average for noisier plots
        ma_window = 10 # Adjust window size as needed
        mu_smoothed = moving_average(dynamic_results["mu"], ma_window)
        mu_grad_raw_smoothed = moving_average(dynamic_results["mu_grad_raw"], ma_window)
        unsafe_inst_smoothed = moving_average(dynamic_results["unsafe_instantaneous"], ma_window)
        
        # Adjust iterations for smoothed data due to 'valid' convolution mode
        iter_smoothed_start = (ma_window -1) //2 
        iter_smoothed_end = len(iterations) - (ma_window // 2)
        if len(iterations) >= ma_window :
             iter_smoothed = iterations[iter_smoothed_start : iter_smoothed_end] if ma_window % 2 == 1 else iterations[ma_window-1:]
        else: # not enough data to smooth
            iter_smoothed = iterations
            mu_smoothed = dynamic_results["mu"] # use raw if not enough data
            mu_grad_raw_smoothed = dynamic_results["mu_grad_raw"]
            unsafe_inst_smoothed = dynamic_results["unsafe_instantaneous"]


        # Plot 1: Mu and Mu Grad Raw (Dynamic Detail)
        lns_mu1 = ax_dynamic_detail.plot(iter_smoothed, mu_smoothed, 'b-', linewidth=1.5, label='μ (Safety Dual, MA)')
        ax_dynamic_detail.set_xlabel('Iteration')
        ax_dynamic_detail.set_ylabel('μ Value', color='b')
        ax_dynamic_detail.tick_params(axis='y', labelcolor='b')
        ax_dynamic_detail.set_ylim(bottom=-0.05) # Ensure 0 is visible

        ax_twin_detail = ax_dynamic_detail.twinx()
        lns_mu2 = ax_twin_detail.plot(iter_smoothed, mu_grad_raw_smoothed, 'g--', linewidth=1.5, label='μ Grad Raw (MA)')
        ax_twin_detail.set_ylabel('μ Grad Raw Value', color='g')
        ax_twin_detail.tick_params(axis='y', labelcolor='g')
        ax_twin_detail.axhline(y=0, color='k', linestyle=':', linewidth=1.0)
        min_grad = min(mu_grad_raw_smoothed) if len(mu_grad_raw_smoothed)>0 else -0.01
        max_grad = max(mu_grad_raw_smoothed) if len(mu_grad_raw_smoothed)>0 else 0.01
        ax_twin_detail.set_ylim(min(min_grad*1.1, -0.01), max(max_grad*1.1, 0.01) if max_grad > 0 else 0.02)


        lns_mu = lns_mu1 + lns_mu2
        labs_mu = [l.get_label() for l in lns_mu]
        ax_dynamic_detail.legend(lns_mu, labs_mu, loc='upper right', fontsize='small')
        ax_dynamic_detail.set_title('Dynamic Env: Dual Variable Adaptation')
        ax_dynamic_detail.grid(True, alpha=0.4, linestyle='--')

        # Plot 2: Objective and Safety (Dynamic Main)
        lns1 = ax_dynamic_main.plot(iterations, dynamic_results["f"], 'm-', linewidth=2, label='Objective f')
        ax_dynamic_main.set_xlabel('Iteration')
        ax_dynamic_main.set_ylabel('Objective Value (log scale)', color='m')
        ax_dynamic_main.tick_params(axis='y', labelcolor='m')
        ax_dynamic_main.set_yscale('log')

        ax_twin_main = ax_dynamic_main.twinx()
        lns2 = ax_twin_main.plot(iterations, dynamic_results["unsafe_avg"], 'c-', linewidth=2, label='Unsafe Prob (Avg)')
        # Plot smoothed instantaneous unsafe probability for trend, raw for detail
        lns3 = ax_twin_main.plot(iter_smoothed, unsafe_inst_smoothed, 'lime', linestyle='-', linewidth=1.0, alpha=0.7, label='Unsafe Prob (Inst., MA)')
        # ax_twin_main.plot(iterations, dynamic_results["unsafe_instantaneous"], 'lime', linestyle=':', linewidth=0.5, alpha=0.5, label='_nolegend_') # Raw, very noisy
        
        ax_twin_main.axhline(y=tau_dynamic, color='k', linestyle='--', linewidth=1.2, label=f'τ={tau_dynamic}')
        ax_twin_main.set_ylabel('Unsafe Probability')
        ax_twin_main.set_ylim(-0.005, max(0.05, 1.5 * max(dynamic_results["unsafe_avg"] if dynamic_results["unsafe_avg"] else [0.01]))) # Adjust ylim based on data

        change_iters = [iterations[i] for i, changed in enumerate(dynamic_results["hazard_config_changed"]) if changed and i > 0]
        for xc in change_iters:
            ax_dynamic_main.axvline(x=xc, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        
        lns = lns1 + lns2 + lns3
        if change_iters:
            lns4_dummy = ax_dynamic_main.plot([], [], color='gray', linestyle='--', label='Hazard Change')[0]
            lns.append(lns4_dummy)
        labs = [l.get_label() for l in lns]
        ax_dynamic_main.legend(lns, labs, loc='best', fontsize='small')
        ax_dynamic_main.set_title('Dynamic Env: Performance & Safety')

    else: # No dynamic results
        if is_dynamic_test_only:
            ax_dynamic_detail.text(0.5, 0.5, 'No Dynamic Data', ha='center', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.5))
            ax_dynamic_main.text(0.5, 0.5, 'No Dynamic Data', ha='center', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.5))
        # In full plot mode, the specific axes would show their individual "no data" messages if needed.

    ax_dynamic_main.grid(True, alpha=0.4, linestyle='--')

    if not is_dynamic_test_only: # If it's the full plot, finalize other axes too.
         pass # Assuming they were handled above

    plt.tight_layout(rect=[0, 0, 1, 0.96] if not is_dynamic_test_only else [0,0,1,0.94]) # Adjust for suptitle
    plt.savefig('scaling_results.png', dpi=300, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser(description="Scaling Beyond GridWorld")
    parser.add_argument("--test", choices=["maze", "stochastic", "scaling", "comparison", "dynamic", "all"], 
                       default="all", help="Which experiment to run")
    args = parser.parse_args()
    
    print("🚀 SCALING BEYOND GRIDWORLD")
    print("=" * 50)
    
    maze_hist, stochastic_res_tuple, scaling_data, env_comp_res, dynamic_res_tuple = None, None, None, None, None
    is_dynamic_only_run = (args.test == "dynamic")

    if args.test in ["maze", "all"]:
        maze_hist = run_maze_experiment()
    if args.test in ["stochastic", "all"]:
        stochastic_res_tuple = run_stochastic_experiment()
    if args.test in ["scaling", "all"]:
        scaling_data = run_scaling_analysis()
    if args.test in ["comparison", "all"]:
        env_comp_res = run_environment_comparison()
    if args.test in ["dynamic", "all"]:
        dynamic_res_tuple = run_dynamic_env_experiment()
            
    should_plot = (is_dynamic_only_run and dynamic_res_tuple and dynamic_res_tuple[0].get("iteration")) or \
                  (args.test == "all" and any([
                      maze_hist and maze_hist.get("iteration"),
                      stochastic_res_tuple and stochastic_res_tuple[0], 
                      scaling_data and scaling_data.get("states"),
                      env_comp_res, 
                      dynamic_res_tuple and dynamic_res_tuple[0].get("iteration")]))

    if should_plot:
        print("\n📊 Creating scaling analysis plots...")
        create_scaling_plots(
            maze_hist if maze_hist else defaultdict(list), 
            stochastic_res_tuple, 
            scaling_data if scaling_data else defaultdict(list), 
            env_comp_res if env_comp_res else {},
            dynamic_res_tuple,
            is_dynamic_test_only=is_dynamic_only_run # Pass the flag
        )
        print("✅ Results saved to scaling_results.png")
        plt.show()
    else:
        print("\n📊 No data to plot for the selected test(s) or test run was not 'dynamic' or 'all'.")
    
    print("\n🎉 Scaling experiments completed!")
    
    # ... (Insights printing as before) ...
    if scaling_data and scaling_data.get("states"):
        states_arr = np.array(scaling_data["states"]) 
        times_arr = np.array(scaling_data["time"])   
        if len(states_arr) > 1 :
            slope = np.polyfit(np.log(states_arr), np.log(times_arr), 1)[0]
            print(f"\n📊 INSIGHTS (Scaling):")
            print(f"   Computational scaling (simplified test): O(N^{slope:.2f}) where N is number of states.")

    if env_comp_res:
        print("\n📊 INSIGHTS (Environment Comparison):")
        print("   Environment performance ranking (lower final objective is better):")
        final_objectives = {}
        for env_name_key, hist_val in env_comp_res.items(): 
            if hist_val.get("f"):
                 final_objectives[env_name_key] = hist_val["f"][-1]
        
        sorted_envs = sorted(final_objectives.items(), key=lambda x: x[1])
        for i, (env_name_key, obj_val) in enumerate(sorted_envs, 1):
            print(f"   {i}. {env_name_key}: {obj_val:.6f}")


if __name__ == "__main__":
    main()