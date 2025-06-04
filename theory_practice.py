#!/usr/bin/env python3
"""
Theory-meets-Practice Checks

Empirical verification of theoretical predictions:
1. Regret plots showing O(1/√K) convergence (Theorem 3.3 of Miryoosefi)
2. Occupancy error vs rollout budget showing 1/√N noisy-oracle bound
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import trange
import argparse

from envs import GridWorld
from dual_loop import expert_policy, solve_soft_q, rollout_policy
from utils import collect_occupancy, DEF_GAMMA

def compute_lagrangian(d, d_E, lam, mu, c_unsafe, tau):
    """Compute the Lagrangian L(d, λ, μ)"""
    f_val = np.sum((d - d_E)**2)
    constraint_term = mu * (np.dot(d, c_unsafe) - tau)
    imitation_term = np.dot(lam, d - d_E)
    return f_val + constraint_term + imitation_term

def run_regret_analysis():
    """Track empirical regret to verify O(1/√K) convergence"""
    print(" Empirical Regret Analysis")
    print("   Verifying O(1/√K) convergence rate...")
    
    env = GridWorld()
    d_E = collect_occupancy(env, expert_policy, n_episodes=1000)
    
    unsafe_idx = np.repeat(env.unsafe_mask, env.nA)
    c_unsafe = unsafe_idx.astype(float)
    tau = 0.05
    
    # Track all iterates for regret computation
    all_d = []
    all_lam = []
    all_mu = []
    
    lam = np.zeros_like(d_E)
    mu = 0.0
    
    regret_history = defaultdict(list)
    
    for k in trange(1, 2001, desc="Regret analysis"):
        # Policy player
        reward = -(lam + mu * c_unsafe)
        pi = solve_soft_q(env, reward, n_iter=80)
        d = rollout_policy(env, pi)
        
        # Store iterates
        all_d.append(d.copy())
        all_lam.append(lam.copy())
        all_mu.append(mu)
        
        # Dual updates
        lam += 1.0 * 2*(d - d_E)
        mu = max(0.0, mu + 10.0 * (np.dot(d, c_unsafe) - tau))
        
        # Compute regret every 50 iterations
        if k % 50 == 0 and k >= 100:
            # Average regret: (1/K) * sum_{t=1}^K L(d^t, λ^t, μ^t)
            avg_regret = 0.0
            for t in range(k):
                lagrangian = compute_lagrangian(all_d[t], d_E, all_lam[t], all_mu[t], c_unsafe, tau)
                avg_regret += lagrangian
            avg_regret /= k
            
            regret_history["iteration"].append(k)
            regret_history["avg_regret"].append(avg_regret)
            regret_history["theoretical_bound"].append(1.0 / np.sqrt(k))  # O(1/√K) bound
            
            if k % 200 == 0:
                print(f"   k={k:4d}  avg_regret={avg_regret:.6f}  bound={1.0/np.sqrt(k):.6f}")
    
    return regret_history

def run_occupancy_error_analysis():
    """Test occupancy error vs rollout budget (1/√N bound)"""
    print("\n🎯 Occupancy Error vs Rollout Budget")
    print("   Verifying 1/√N noisy-oracle bound...")
    
    env = GridWorld()
    d_E_true = collect_occupancy(env, expert_policy, n_episodes=10000)  # Ground truth
    
    # Test different rollout budgets
    rollout_budgets = [10, 25, 50, 100, 200, 500, 1000]
    num_trials = 20  # Multiple trials for statistical significance
    
    results = defaultdict(list)
    
    for budget in rollout_budgets:
        print(f"   Testing budget N={budget}...")
        
        errors = []
        for trial in range(num_trials):
            # Estimate occupancy with limited budget
            d_estimated = collect_occupancy(env, expert_policy, n_episodes=budget)
            
            # Compute L2 error
            error = np.linalg.norm(d_estimated - d_E_true)
            errors.append(error)
        
        # Statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        theoretical_bound = 1.0 / np.sqrt(budget)
        
        results["budget"].append(budget)
        results["mean_error"].append(mean_error)
        results["std_error"].append(std_error)
        results["theoretical_bound"].append(theoretical_bound)
        
        print(f"     Mean error: {mean_error:.6f} ± {std_error:.6f}")
        print(f"     Theoretical bound: {theoretical_bound:.6f}")
    
    return results

def run_dual_variable_convergence():
    """Analyze dual variable convergence patterns"""
    print("\n Dual Variable Convergence Analysis")
    
    env = GridWorld()
    d_E = collect_occupancy(env, expert_policy, n_episodes=1000)
    
    unsafe_idx = np.repeat(env.unsafe_mask, env.nA)
    c_unsafe = unsafe_idx.astype(float)
    tau = 0.05
    
    # Test different step sizes
    step_sizes = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    results = {}
    
    for eta in step_sizes:
        print(f"   Testing step size η={eta}...")
        
        lam = np.zeros_like(d_E)
        mu = 0.0
        avg_d = np.zeros_like(d_E)
        
        history = defaultdict(list)
        
        for k in trange(1, 1001, desc=f"η={eta}", leave=False):
            reward = -(lam + mu * c_unsafe)
            pi = solve_soft_q(env, reward, n_iter=80)
            d = rollout_policy(env, pi)
            
            # Dual updates with different step sizes
            lam += eta * 2*(d - d_E)
            mu = max(0.0, mu + 10.0 * eta * (np.dot(d, c_unsafe) - tau))
            
            avg_d = ((k-1)*avg_d + d) / k
            
            if k % 25 == 0:
                f_val = np.sum((avg_d - d_E)**2)
                unsafe = np.dot(avg_d, c_unsafe)
                lam_norm = np.linalg.norm(lam)
                
                history["iteration"].append(k)
                history["f"].append(f_val)
                history["unsafe"].append(unsafe)
                history["lam_norm"].append(lam_norm)
                history["mu"].append(mu)
        
        results[eta] = history
        final_f = history["f"][-1]
        print(f"     Final f: {final_f:.6f}")
    
    return results

def create_theory_plots(regret_hist, occupancy_results, dual_results):
    """Create visualization plots for theory verification"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Regret convergence
    ax = axes[0, 0]
    iterations = np.array(regret_hist["iteration"])
    avg_regret = np.array(regret_hist["avg_regret"])
    theoretical = np.array(regret_hist["theoretical_bound"])
    
    ax.loglog(iterations, avg_regret, 'b-', linewidth=2, label='Empirical Regret')
    ax.loglog(iterations, theoretical * avg_regret[0] / theoretical[0], 'r--', 
              linewidth=2, label='O(1/√K) Reference')
    ax.set_xlabel('Iteration K')
    ax.set_ylabel('Average Regret')
    ax.set_title('Regret Convergence\n(Theorem 3.3 Verification)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Regret slope analysis
    ax = axes[1, 0]
    # Compute empirical slope
    log_iters = np.log(iterations[5:])  # Skip first few points
    log_regret = np.log(avg_regret[5:])
    slope = np.polyfit(log_iters, log_regret, 1)[0]
    
    ax.plot(log_iters, log_regret, 'b-', linewidth=2, label=f'Empirical (slope={slope:.2f})')
    ax.plot(log_iters, log_iters * (-0.5) + log_regret[0] + 0.5 * log_iters[0], 
            'r--', linewidth=2, label='Theoretical (-0.5)')
    ax.set_xlabel('log(K)')
    ax.set_ylabel('log(Regret)')
    ax.set_title('Regret Slope Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Occupancy error vs budget
    ax = axes[0, 1]
    budgets = np.array(occupancy_results["budget"])
    mean_errors = np.array(occupancy_results["mean_error"])
    std_errors = np.array(occupancy_results["std_error"])
    theoretical_bounds = np.array(occupancy_results["theoretical_bound"])
    
    ax.errorbar(budgets, mean_errors, yerr=std_errors, fmt='bo-', 
                linewidth=2, label='Empirical Error')
    ax.plot(budgets, theoretical_bounds * mean_errors[0] / theoretical_bounds[0], 
            'r--', linewidth=2, label='1/√N Reference')
    ax.set_xlabel('Rollout Budget N')
    ax.set_ylabel('Occupancy Error ||d - d_E||')
    ax.set_title('Occupancy Error vs Budget\n(Noisy Oracle Bound)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Error scaling verification
    ax = axes[1, 1]
    # Fit power law to empirical data
    log_budgets = np.log(budgets)
    log_errors = np.log(mean_errors)
    slope = np.polyfit(log_budgets, log_errors, 1)[0]
    
    ax.plot(log_budgets, log_errors, 'bo-', linewidth=2, 
            label=f'Empirical (slope={slope:.2f})')
    ax.plot(log_budgets, log_budgets * (-0.5) + log_errors[0] + 0.5 * log_budgets[0], 
            'r--', linewidth=2, label='Theoretical (-0.5)')
    ax.set_xlabel('log(N)')
    ax.set_ylabel('log(Error)')
    ax.set_title('Error Scaling Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Step size sensitivity
    ax = axes[0, 2]
    for eta, hist in dual_results.items():
        ax.plot(hist["iteration"], hist["f"], label=f'η={eta}', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective f(d)')
    ax.set_title('Step Size Sensitivity')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Dual variable norms
    ax = axes[1, 2]
    for eta, hist in dual_results.items():
        ax.plot(hist["iteration"], hist["lam_norm"], label=f'η={eta}', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('||λ||')
    ax.set_title('Dual Variable Growth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('theory_practice_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Theory-meets-Practice Verification")
    parser.add_argument("--test", choices=["regret", "occupancy", "dual", "all"], 
                       default="all", help="Which analysis to run")
    args = parser.parse_args()
    
    print("🔬 THEORY-MEETS-PRACTICE VERIFICATION")
    print("=" * 50)
    
    if args.test in ["regret", "all"]:
        regret_hist = run_regret_analysis()
    else:
        regret_hist = None
        
    if args.test in ["occupancy", "all"]:
        occupancy_results = run_occupancy_error_analysis()
    else:
        occupancy_results = None
        
    if args.test in ["dual", "all"]:
        dual_results = run_dual_variable_convergence()
    else:
        dual_results = None
    
    # Create plots if we ran all tests
    if args.test == "all":
        print("\n Creating theory verification plots...")
        create_theory_plots(regret_hist, occupancy_results, dual_results)
        print(" Results saved to theory_practice_results.png")
    
    print("\n Theory verification completed!")
    
    # Print summary insights
    if regret_hist:
        # Check if empirical slope matches theory
        iterations = np.array(regret_hist["iteration"][5:])
        regret = np.array(regret_hist["avg_regret"][5:])
        slope = np.polyfit(np.log(iterations), np.log(regret), 1)[0]
        print(f"\n INSIGHTS:")
        print(f"   Regret slope: {slope:.3f} (theory: -0.5)")
        if abs(slope + 0.5) < 0.2:
            print("    Regret matches O(1/√K) theory!")
        else:
            print("    Regret deviates from theory")
    
    if occupancy_results:
        budgets = np.array(occupancy_results["budget"])
        errors = np.array(occupancy_results["mean_error"])
        slope = np.polyfit(np.log(budgets), np.log(errors), 1)[0]
        print(f"   Occupancy error slope: {slope:.3f} (theory: -0.5)")
        if abs(slope + 0.5) < 0.2:
            print("    Occupancy error matches 1/√N theory!")
        else:
            print("    Occupancy error deviates from theory")

if __name__ == "__main__":
    main() 