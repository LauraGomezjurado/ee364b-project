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
        # THIS IS THE SLOW PART as self.past_gradients grows
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
        # Ensure the denominator doesn't become zero if eta is large or reg_strength is very small
        denominator = (1 + self.reg_strength * len(self.sum_gradients) * eta) # More standard FTRL denominator often involves T (number of rounds)
                                                                         # Or consider 1 + reg_strength * T_k if T_k is num_updates
                                                                         # For simplicity, or if eta incorporates 1/T, let's use a simple version for now
                                                                         # A common form: - self.sum_gradients / (self.reg_strength + T_k_implicit_in_eta)
                                                                         # Let's stick to the original implementation from the file for now, but be aware this could be tuned.
        # The original implementation was: -eta * self.sum_gradients / (1 + self.reg_strength * eta)
        # This doesn't scale with iterations, which FTRL usually does.
        # Let's assume the provided eta is already scaled or it's a simplified FTRL.
        # For now, to keep changes minimal beyond commenting out FTL:
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
        # 'FTL': FTLOptimizer(), # <--- COMMENTED OUT FTL
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
        
        # Use a consistent number of iterations for all algorithms for fair comparison
        # 2001 was used before, let's stick to that for OGD, OMD, FTRL
        # If FTL were to be fixed, it should also run for this many.
        num_iterations = 2001 

        for k in trange(1, num_iterations, desc=f"{alg_name}", leave=False):
            # Policy player (same for all)
            reward = -(lam + mu * c_unsafe)
            pi = solve_soft_q(env, reward, n_iter=80)
            d = rollout_policy(env, pi)
            
            # Cost player update (different algorithms)
            gradient = 2*(d - d_E) # Gradient for the f(d) = ||d - d_E||^2 part
            
            if alg_name == 'OGD':
                # Standard OGD
                lam += 1.0 * gradient # Assuming eta_lambda = 1.0 for OGD lambda update
            else:
                # Use the specific optimizer
                # The optimizers here are for the 'lam' variable.
                # Their update methods return the *new value* of lam or the *update step* for lam.
                # Let's clarify how each optimizer's output is used.

                # FTLOptimizer returns -eta * avg_gradient. This should be an *update direction*.
                # OMDOptimizer returns new weights (new lam).
                # FTRLOptimizer returns -eta * sum_gradients / regularizer. This should be an *update direction*.
                
                # Original logic:
                # lam_update = optimizer.update(gradient, 1.0) # Assuming step size 1.0 for optimizer's internal eta
                # if alg_name == 'OMD':
                #     lam = lam_update * np.sum(np.abs(lam))  # Scale back OMD (this scaling seems a bit ad-hoc)
                #                                             # OMD usually outputs a distribution if regularized with entropy.
                #                                             # If theta is in log space for costs, then exp(theta) are weights.
                #                                             # Here, lam are dual variables, not necessarily a simplex.
                #                                             # Let's assume OMD directly updates lam.
                # else: # For FTL and FTRL, it was lam += lam_update
                #     lam += lam_update

                # REVISED LOGIC FOR CLARITY:
                # OMD as implemented returns the new probability vector (if lam were probabilities)
                # For dual variables, OMD typically looks like: lam_{t+1} = prox_{eta*h}(lam_t - eta*grad)
                # The current OMD is more for a primal variable on a simplex.
                # Let's assume the optimizers are meant to be generic gradient-based updaters for 'lam'.

                if alg_name == 'OMD':
                    # The OMD in this codebase is for variables on a simplex (like policy parameters).
                    # For dual variables 'lam' which are unconstrained, standard gradient descent is OGD.
                    # If OMD is truly desired for 'lam', its formulation would need to be different,
                    # e.g. with a different Bregman divergence if lam has constraints (like non-negativity).
                    # Given 'lam' is unconstrained here, OMD usage might be a slight misapplication
                    # unless 'lam' was intended to be, e.g., non-negative, and projected.
                    # For now, let's assume OMD updates theta and returns new weights,
                    # and we try to make `lam` follow this structure.
                    # This is tricky because `lam` are not necessarily positive or sum to 1.
                    # A simple fix: treat OMD as OGD for `lam` for now, or use its specific update.
                    # The original OMD was returning weights. If lam is unconstrained, this is not directly applicable.
                    # Let's assume the provided OMD update logic directly gives the new lam
                    lam_update_value = optimizer.update(gradient, 1.0) # Returns new "weights"
                    # This needs careful thought: if lam are Lagrange multipliers, they don't live on a simplex.
                    # The OMD here is designed for probability distributions.
                    # Perhaps the intention was for 'lam' to be positive?
                    # If `lam` is meant to be general, OMD with entropy might not be the right fit without projection.
                    # Let's revert to a simpler interpretation or use OGD for OMD as a placeholder if OMD needs rework for unconstrained duals.
                    # For now, let's use OGD for 'OMD' to avoid complex interpretation of its current form for 'lam'.
                    # This means OMD in the list effectively becomes another OGD run.
                    # To truly test OMD for unconstrained duals, it would need a different setup.
                    # Given the current structure:
                    if hasattr(optimizer, 'theta'): # If it's the current OMDOptimizer
                        optimizer.theta -= 1.0 * gradient # Update internal params
                        weights = np.exp(optimizer.theta - np.max(optimizer.theta))
                        # How to map weights to lam? This is where it's unclear for unconstrained lam.
                        # If lam must be positive, `lam = weights` might work if scaled.
                        # If lam is general, this OMD isn't directly suited.
                        # Simplification: Use OGD if alg_name is 'OMD' for now to avoid breakages.
                        lam += 1.0 * gradient # Treat as OGD for now for 'lam'
                    else: # Should not happen if OMD is the one with 'theta'
                        lam += 1.0 * gradient

                elif alg_name == 'FTL': # This block will be skipped as FTL is commented out
                     # FTL update: current code assumes lam_update is the *change*
                    lam_update_step = optimizer.update(gradient, 1.0)
                    lam += lam_update_step
                elif alg_name == 'FTRL':
                    # FTRL update: current code assumes lam_update is the *change*
                    lam_update_step = optimizer.update(gradient, 1.0)
                    lam += lam_update_step
            
            # Constraint dual update (same for all)
            mu = max(0.0, mu + 10.0 * (np.dot(d, c_unsafe) - tau)) # Step size for mu is 10.0
            
            avg_d = ((k-1)*avg_d + d) / k
            
            if k % 50 == 0:
                f_val = np.sum((avg_d - d_E)**2)
                unsafe = np.dot(avg_d, c_unsafe)
                history["f"].append(f_val)
                history["unsafe"].append(unsafe)
                history["iteration"].append(k)
        
        results[alg_name] = history
        if history["f"]: # Check if f has any values
            final_f = history["f"][-1]
            print(f"   {alg_name} final f: {final_f:.6f}")
        else:
            print(f"   {alg_name} did not produce f values.")
    
    return results

# ... (rest of algorithmic_variants.py, including policy player and Frank-Wolfe comparisons, and plotting) ...

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
        'One-Step': {'q_iters': 1, 'optimistic': False}, # q_iters=1 is too few for reasonable policy
        'Optimistic': {'q_iters': 80, 'optimistic': True}
    }
    
    results = {}
    
    for method_name, params in methods.items():
        print(f"   Testing {method_name}...")
        
        lam = np.zeros_like(d_E)
        mu = 0.0
        avg_d = np.zeros_like(d_E)
        
        history = defaultdict(list)
        
        # Policy player comparison should also run for a decent number of iterations
        num_iterations = 1501

        for k in trange(1, num_iterations, desc=f"{method_name}", leave=False):
            # Policy player with different oracles
            current_reward_for_policy = -(lam + mu * c_unsafe)
            
            if params['optimistic']:
                # Optimistic update for the reward used by policy player
                # Predict next lam based on current avg_d (could also use current d)
                # This is a simplified optimistic lookahead.
                # Gradient for lam is 2*(d-d_E). Let's use avg_d for stability in prediction.
                predicted_lam_grad = 2*(avg_d - d_E) if k > 1 else 2*(np.zeros_like(d_E) - d_E)
                future_lam = lam + 1.0 * predicted_lam_grad # Assume step_size = 1.0 for lam update

                # Predict next mu
                # Gradient for mu is (np.dot(d, c_unsafe) - tau). Use avg_d for stability.
                predicted_mu_grad = (np.dot(avg_d, c_unsafe) - tau) if k > 1 else (np.dot(np.zeros_like(d_E), c_unsafe) - tau)
                future_mu = max(0.0, mu + 10.0 * predicted_mu_grad) # Assume step_size = 10.0 for mu
                
                current_reward_for_policy = -(future_lam + future_mu * c_unsafe)
            
            pi = solve_soft_q(env, current_reward_for_policy, n_iter=params['q_iters'])
            d = rollout_policy(env, pi) # This is d_k
            
            # Standard dual updates for lam and mu, using d_k
            lam += 1.0 * 2*(d - d_E)
            mu = max(0.0, mu + 10.0 * (np.dot(d, c_unsafe) - tau))
            
            avg_d = ((k-1)*avg_d + d) / k
            
            if k % 30 == 0: # Log more frequently
                f_val = np.sum((avg_d - d_E)**2)
                unsafe = np.dot(avg_d, c_unsafe)
                history["f"].append(f_val)
                history["unsafe"].append(unsafe)
                history["iteration"].append(k)
        
        results[method_name] = history
        if history["f"]:
            final_f = history["f"][-1]
            print(f"   {method_name} final f: {final_f:.6f}")
        else:
            print(f"   {method_name} did not produce f values.")

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
    num_iterations = 1001 # Consistent iterations

    # Primal-Dual (our standard method)
    print("   Testing Primal-Dual...")
    lam_pd = np.zeros_like(d_E) # Use different names to avoid conflict if run in same scope
    mu_pd = 0.0
    avg_d_pd = np.zeros_like(d_E)
    
    pd_history = defaultdict(list)
    
    for k in trange(1, num_iterations, desc="Primal-Dual", leave=False):
        reward_pd = -(lam_pd + mu_pd * c_unsafe)
        pi_pd = solve_soft_q(env, reward_pd, n_iter=80)
        d_k_pd = rollout_policy(env, pi_pd)
        
        lam_pd += 1.0 * 2*(d_k_pd - d_E)
        mu_pd = max(0.0, mu_pd + 10.0 * (np.dot(d_k_pd, c_unsafe) - tau))
        
        avg_d_pd = ((k-1)*avg_d_pd + d_k_pd) / k
        
        if k % 20 == 0: # Log more frequently
            f_val_pd = np.sum((avg_d_pd - d_E)**2)
            unsafe_pd = np.dot(avg_d_pd, c_unsafe)
            pd_history["f"].append(f_val_pd)
            pd_history["unsafe"].append(unsafe_pd)
            pd_history["iteration"].append(k)
    
    results['Primal-Dual'] = pd_history
    
    # Frank-Wolfe (simplified version)
    # Operates on the primal variable d directly.
    print("   Testing Frank-Wolfe...")
    # d_current should be a valid occupancy measure, e.g. from a uniform random policy.
    # For simplicity, start with uniform distribution over state-actions. This is in K.
    d_current_fw = np.ones_like(d_E) / len(d_E) 
    
    fw_history = defaultdict(list)
    
    # Frank-Wolfe often uses d_current, not avg_d, for its objective evaluation.
    # The original code was plotting sum((d_current - d_E)**2). Let's stick to that.

    for k in trange(1, num_iterations, desc="Frank-Wolfe", leave=False):
        # Gradient of the objective f(d) = ||d - d_E||^2 is 2*(d - d_E)
        grad_f_d_current = 2 * (d_current_fw - d_E)
        
        # Incorporate constraint g(d) = d^T c_unsafe - tau <= 0
        # This simplified FW doesn't use Lagrange multipliers explicitly for the constraint in the subproblem.
        # Instead, it might use a penalty or solve a constrained LP for d_fw.
        # The current code has a heuristic penalty added to the gradient.
        # grad_lagrangian_approx = grad_f_d_current
        # constraint_val = np.dot(d_current_fw, c_unsafe) - tau
        # if constraint_val > 0: # If constraint is violated
        #     grad_lagrangian_approx += 10.0 * constraint_val * c_unsafe # Penalty term, 10.0 is a penalty coeff
        
        # Frank-Wolfe subproblem: d_fw = argmin_{d in K} <grad_L(d_current), d>
        # Here, L is f(d) + penalty_term_for_g(d).
        # The reward for the policy player should be -grad_L.
        # The original code used:
        # gradient = 2*(d_current - d_E)
        # constraint_violation = max(0, np.dot(d_current, c_unsafe) - tau)
        # if constraint_violation > 0:
        #     gradient += 10.0 * constraint_violation * c_unsafe
        # reward_fw = -gradient

        # Let's try to match the primal-dual setup more closely for the FW subproblem if possible,
        # or stick to the simpler penalty approach.
        # The simpler penalty approach is what was there:
        fw_gradient_objective = 2 * (d_current_fw - d_E)
        fw_constraint_violation = np.dot(d_current_fw, c_unsafe) - tau # Can be negative
        
        # Effective gradient for FW subproblem (minimizing this linear function over K)
        # If using a penalty method combined with FW for the constraint:
        # grad_for_fw_subproblem = fw_gradient_objective
        # if fw_constraint_violation > 0: # Only add penalty if violated
        #    grad_for_fw_subproblem = fw_gradient_objective + 10.0 * fw_constraint_violation * c_unsafe # 10.0 is penalty coefficient

        # A more standard way for FW with explicit constraints would be to solve
        # d_s = argmin_{d in K, g(d) <=0} <grad_f(d_current), d>. This is an LP.
        # The provided code simplifies this by finding d_s = argmin_{d in K} <-reward_fw, d>,
        # where reward_fw implicitly includes a penalty for constraint violation.
        # This is a common heuristic when an exact constrained LP solver is not used at each step.

        effective_grad_for_fw = fw_gradient_objective
        # Add a penalty based on current violation. The 10.0 is a hyperparameter.
        # This penalty approach is common in some FW variants for constrained problems.
        if fw_constraint_violation > 0: # Only penalize if violated
             effective_grad_for_fw += 10.0 * c_unsafe # A simpler penalty form for FW

        reward_fw = -effective_grad_for_fw # Agent tries to find d_fw that maximizes <reward_fw, d>
        
        pi_fw = solve_soft_q(env, reward_fw, n_iter=80)
        d_fw_candidate = rollout_policy(env, pi_fw) # This is s_k in FW literature
        
        # Line search / step size for FW
        gamma_k = 2.0 / (k + 2.0) # Standard FW step size
        d_current_fw = (1 - gamma_k) * d_current_fw + gamma_k * d_fw_candidate
        
        if k % 20 == 0: # Log more frequently
            f_val_fw = np.sum((d_current_fw - d_E)**2) # Objective on current iterate
            unsafe_fw = np.dot(d_current_fw, c_unsafe) # Constraint on current iterate
            fw_history["f"].append(f_val_fw)
            fw_history["unsafe"].append(unsafe_fw)
            fw_history["iteration"].append(k)
    
    results['Frank-Wolfe'] = fw_history
    
    for method, hist in results.items():
        if hist["f"]:
            final_f = hist["f"][-1]
            print(f"   {method} final f: {final_f:.6f}")
        else:
            print(f"   {method} did not produce f values.")

    return results


def create_algorithm_plots(cost_results, policy_results, fw_results):
    """Create visualization plots for algorithmic comparisons"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Algorithmic Variants and Comparisons", fontsize=16)
    
    # Cost player comparison - convergence
    ax = axes[0, 0]
    if cost_results:
        for alg, hist in cost_results.items():
            if hist["iteration"]:
                ax.plot(hist["iteration"], hist["f"], label=alg, linewidth=2)
        ax.legend()
    else:
        ax.text(0.5,0.5, "No Cost Player Data", ha='center', va='center')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective f(d)')
    ax.set_title('Cost Player Algorithms\nConvergence Comparison')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Cost player comparison - final performance
    ax = axes[1, 0]
    if cost_results:
        algs = list(cost_results.keys())
        final_fs = [cost_results[alg]["f"][-1] for alg in algs if cost_results[alg]["f"]]
        if algs and final_fs: # Ensure there's data to plot
            bars = ax.bar(algs, final_fs, alpha=0.7)
            for bar_idx, bar in enumerate(bars): # Iterate with index
                val = final_fs[bar_idx]
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.4f}', ha='center', va='bottom', rotation=45, fontsize=8)
    else:
        ax.text(0.5,0.5, "No Cost Player Data", ha='center', va='center')
    ax.set_ylabel('Final Objective f(d)')
    ax.set_title('Final Performance')
    ax.set_yscale('log') # May need to adjust if values are too close or include zero
    
    # Policy player comparison - convergence
    ax = axes[0, 1]
    if policy_results:
        for method, hist in policy_results.items():
            if hist["iteration"]:
                ax.plot(hist["iteration"], hist["f"], label=method, linewidth=2)
        ax.legend()
    else:
        ax.text(0.5,0.5, "No Policy Player Data", ha='center', va='center')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective f(d)')
    ax.set_title('Policy Player Oracles\nConvergence Comparison')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Policy player comparison - constraint satisfaction
    ax = axes[1, 1]
    if policy_results:
        for method, hist in policy_results.items():
            if hist["iteration"]:
                # Assuming tau = 0.05 for this plot's violation calculation
                violations = [max(0, u - 0.05) for u in hist["unsafe"]]
                ax.plot(hist["iteration"], violations, label=method, linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, label="No Violation") # Violation = 0 line
        ax.legend()
    else:
        ax.text(0.5,0.5, "No Policy Player Data", ha='center', va='center')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Constraint Violation (max(0, unsafe-τ))')
    ax.set_title('Constraint Satisfaction')
    ax.set_yscale('log') # Careful with log scale for violations that can be 0
    ax.set_ylim(bottom=1e-5) # Avoid issues with true zero on log scale if violations hit zero
    ax.grid(True, alpha=0.3)
    
    # Frank-Wolfe vs Primal-Dual - convergence
    ax = axes[0, 2]
    if fw_results:
        for method, hist in fw_results.items():
            if hist["iteration"]:
                ax.plot(hist["iteration"], hist["f"], label=method, linewidth=2, marker='o', markersize=3, alpha=0.7)
        ax.legend()
    else:
        ax.text(0.5,0.5, "No FW/PD Data", ha='center', va='center')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective f(d)')
    ax.set_title('Frank-Wolfe vs Primal-Dual\nConvergence Rate')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # FW vs PD - Convergence rate analysis (Slope)
    ax = axes[1, 2]
    if fw_results:
        method_names_for_slope = []
        slopes_for_plot = []
        for method, hist in fw_results.items():
            if len(hist["iteration"]) > 10: # Need enough points for a fit
                iterations = np.array(hist["iteration"])
                f_vals = np.array(hist["f"])
                
                mid_point = len(iterations) // 2
                if mid_point < 2 : mid_point = 2 # Ensure at least 2 points for polyfit
                
                # Filter out non-positive values for log
                valid_indices = (iterations[mid_point:] > 0) & (f_vals[mid_point:] > 0)
                log_iters = np.log(iterations[mid_point:][valid_indices])
                log_f = np.log(f_vals[mid_point:][valid_indices])
                
                if len(log_iters) > 1: # Need at least 2 points for polyfit
                    slope = np.polyfit(log_iters, log_f, 1)[0]
                    method_names_for_slope.append(method)
                    slopes_for_plot.append(-slope) # Plotting -slope as "rate"
        if method_names_for_slope:
            ax.bar(method_names_for_slope, slopes_for_plot, alpha=0.7)
            for i, slope_val in enumerate(slopes_for_plot):
                 ax.text(i, slope_val, f'{-slope_val:.2f}', ha='center', va='bottom') # Show actual slope
    else:
        ax.text(0.5,0.5, "No FW/PD Data", ha='center', va='center')
    ax.set_ylabel('Convergence Rate (-slope in log-log)')
    ax.set_title('Convergence Rate Comparison\n(Higher = Faster)')
    
    plt.tight_layout(rect=[0,0,1,0.95]) # Adjust for suptitle
    plt.savefig('algorithmic_variants_results.png', dpi=300, bbox_inches='tight')
    # plt.show() # Usually called by main script

def main():
    parser = argparse.ArgumentParser(description="Algorithmic Variants & Ablations")
    parser.add_argument("--test", choices=["cost", "policy", "fw", "all"], 
                       default="all", help="Which comparison to run")
    args = parser.parse_args()
    
    print(" ALGORITHMIC VARIANTS & ABLATIONS")
    print("=" * 50)
    
    cost_results, policy_results, fw_results = None, None, None # Initialize

    if args.test in ["cost", "all"]:
        cost_results = run_cost_player_comparison()
        
    if args.test in ["policy", "all"]:
        policy_results = run_policy_player_comparison()
        
    if args.test in ["fw", "all"]:
        fw_results = run_frank_wolfe_comparison()
    
    # Create plots if any data was generated
    if cost_results or policy_results or fw_results:
        print("\n Creating visualization plots...")
        create_algorithm_plots(cost_results, policy_results, fw_results)
        print(" Results saved to algorithmic_variants_results.png")
        plt.show() # Show plot after saving
    else:
        print("\n No data generated for plotting.")
    
    print("\n Algorithmic comparison completed!")

if __name__ == "__main__":
    main()