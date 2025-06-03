#!/usr/bin/env python3
"""
Quick wins implementation - immediate experiments you can run tonight!

This script implements:
1. Hazard relocation experiment (unsafe_cells=(6,11))
2. Slip parameter experiment (slip=0.2)
3. Basic visualization and comparison

Run with: python quick_wins.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from collections import defaultdict

from envs import GridWorld
from dual_loop import main as run_original_dual_loop
from utils import collect_occupancy, plot_curves
import argparse

def expert_policy(s, size=5):
    """Expert policy for imitation"""
    row, col = divmod(s, size)
    if col < size-1: return 3          # go RIGHT
    if row < size-1: return 1          # then DOWN
    return 0                           # arbitrary at goal

def run_experiment(env_params, algo_params, name, iterations=3000):
    """Run a single experiment with given parameters"""
    print(f"\n🚀 Running experiment: {name}")
    print(f"   Environment: {env_params}")
    print(f"   Algorithm: {algo_params}")
    
    # Create environment
    env = GridWorld(**env_params)
    
    # Get expert occupancy
    d_E = collect_occupancy(env, expert_policy, n_episodes=1000)
    
    # Setup constraints
    unsafe_idx = np.repeat(env.unsafe_mask, env.nA)
    c_unsafe = unsafe_idx.astype(float)
    tau = algo_params.get('tau', 0.05)
    
    # Initialize dual variables
    lam = np.zeros_like(d_E)
    mu = 0.0
    avg_d = np.zeros_like(d_E)
    
    # Algorithm parameters
    eta_lam = algo_params.get('eta_lam', 1.0)
    eta_mu = algo_params.get('eta_mu', 10.0)
    temp = algo_params.get('temp', 1.0)
    alpha = algo_params.get('alpha', 0.1)
    
    history = defaultdict(list)
    start_time = time.time()
    
    for k in range(1, iterations + 1):
        # Policy player - solve soft Q-learning
        reward = -(lam + mu * c_unsafe)
        Q = np.zeros((env.nS, env.nA))
        
        # Soft Q-learning
        for _ in range(80):
            for s in range(env.nS):
                for a in range(env.nA):
                    s_next = env._next_state(s, a)
                    max_q = np.max(Q[s_next])
                    Q[s, a] = (1-alpha)*Q[s, a] + alpha*(reward[s*env.nA+a] + 0.99 * max_q)
        
        # Softmax policy
        pi = np.exp(Q/temp) / np.sum(np.exp(Q/temp), axis=1, keepdims=True)
        
        # Rollout policy
        d = np.zeros(env.nS * env.nA)
        for _ in range(50):
            s, _ = env.reset()
            t, done = 0, False
            while not done:
                a = np.random.choice(env.nA, p=pi[s])
                s_next, _, done, _, _ = env.step(a)
                d[s*env.nA + a] += (1-0.99)*(0.99**t)
                s, t = s_next, t+1
        d = d / 50
        
        # Dual updates (OGD)
        lam += eta_lam * 2*(d - d_E)
        mu = max(0.0, mu + eta_mu * (np.dot(d, c_unsafe) - tau))
        
        # Running average
        avg_d = ((k-1)*avg_d + d) / k
        
        # Logging
        if k % 100 == 0:
            f_val = np.sum((avg_d - d_E)**2)
            unsafe = np.dot(avg_d, c_unsafe)
            history["f"].append(f_val)
            history["unsafe"].append(unsafe)
            history["iteration"].append(k)
            
            if k % 500 == 0:
                print(f"   iter={k:4d}  f={f_val:.4f}  unsafe={unsafe:.3f}  τ={tau:.3f}")
    
    runtime = time.time() - start_time
    
    # Final metrics
    final_f = history["f"][-1] if history["f"] else float('inf')
    final_unsafe = history["unsafe"][-1] if history["unsafe"] else float('inf')
    constraint_violation = max(0, final_unsafe - tau)
    
    print(f"   ✅ Completed in {runtime:.1f}s")
    print(f"   📊 Final f={final_f:.4f}, unsafe={final_unsafe:.3f}, violation={constraint_violation:.4f}")
    
    return {
        'name': name,
        'history': history,
        'runtime': runtime,
        'final_metrics': {
            'f_value': final_f,
            'unsafe_prob': final_unsafe,
            'constraint_violation': constraint_violation,
            'tau': tau
        },
        'params': {'env': env_params, 'algo': algo_params}
    }

def create_comparison_plots(results, output_dir="quick_wins_plots"):
    """Create comparison plots for the quick wins experiments"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # 1. Learning curves comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Objective function
    ax = axes[0, 0]
    for result in results:
        if result['history']['f']:
            ax.plot(result['history']['iteration'], result['history']['f'], 
                   label=result['name'], linewidth=2, alpha=0.8)
    ax.set_title('Objective Function f(d) = ||d - d_E||²')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('f(d)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Constraint satisfaction
    ax = axes[0, 1]
    for result in results:
        if result['history']['unsafe']:
            tau = result['final_metrics']['tau']
            ax.plot(result['history']['iteration'], result['history']['unsafe'], 
                   label=result['name'], linewidth=2, alpha=0.8)
            ax.axhline(y=tau, color='red', linestyle='--', alpha=0.5, 
                      label=f'τ={tau}' if result == results[0] else "")
    ax.set_title('Safety Constraint: P(unsafe)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Unsafe Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Final performance comparison
    ax = axes[1, 0]
    names = [r['name'] for r in results]
    f_values = [r['final_metrics']['f_value'] for r in results]
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    bars = ax.bar(names, f_values, color=colors, alpha=0.7)
    ax.set_title('Final Objective Values')
    ax.set_ylabel('f(d)')
    ax.set_yscale('log')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, val in zip(bars, f_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Constraint violations
    ax = axes[1, 1]
    violations = [r['final_metrics']['constraint_violation'] for r in results]
    bars = ax.bar(names, violations, color=colors, alpha=0.7)
    ax.set_title('Constraint Violations')
    ax.set_ylabel('max(0, unsafe - τ)')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add value labels
    for bar, val in zip(bars, violations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/quick_wins_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Trade-off analysis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for result in results:
        f_val = result['final_metrics']['f_value']
        unsafe_val = result['final_metrics']['unsafe_prob']
        tau = result['final_metrics']['tau']
        
        # Color based on constraint satisfaction
        color = 'green' if unsafe_val <= tau else 'red'
        size = 100 if unsafe_val <= tau else 150
        
        ax.scatter(f_val, unsafe_val, s=size, alpha=0.7, c=color, 
                  label=result['name'])
        
        # Add text annotation
        ax.annotate(result['name'], (f_val, unsafe_val), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Add constraint line
    tau_line = results[0]['final_metrics']['tau']  # Assuming same tau for visualization
    ax.axhline(y=tau_line, color='red', linestyle='--', alpha=0.5, 
              label=f'Safety threshold τ={tau_line}')
    
    ax.set_xlabel('Objective Value f(d)')
    ax.set_ylabel('Unsafe Probability')
    ax.set_title('Performance Trade-off: Imitation vs Safety')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tradeoff_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📈 Plots saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description="Quick Wins Experiments")
    parser.add_argument("--iterations", type=int, default=3000, help="Number of iterations per experiment")
    parser.add_argument("--output", type=str, default="quick_wins_plots", help="Output directory for plots")
    args = parser.parse_args()
    
    print("🎯 QUICK WINS EXPERIMENTS")
    print("=" * 50)
    
    # Define experiments
    experiments = [
        {
            'name': 'Baseline',
            'env_params': {},  # Default: size=5, unsafe_cells=(12, 17), slip=0.0
            'algo_params': {'tau': 0.05}
        },
        {
            'name': 'Hazard Relocation (6,11)',
            'env_params': {'unsafe_cells': (6, 11)},
            'algo_params': {'tau': 0.05}
        },
        {
            'name': 'High Slip (0.2)',
            'env_params': {'slip': 0.2},
            'algo_params': {'tau': 0.05}
        },
        {
            'name': 'Tight Constraint (τ=0.02)',
            'env_params': {},
            'algo_params': {'tau': 0.02}
        },
        {
            'name': 'Multiple Hazards',
            'env_params': {'unsafe_cells': (6, 11, 16)},
            'algo_params': {'tau': 0.05}
        },
        {
            'name': 'High LR (η_λ=5)',
            'env_params': {},
            'algo_params': {'tau': 0.05, 'eta_lam': 5.0}
        }
    ]
    
    # Run experiments
    results = []
    total_start = time.time()
    
    for exp in experiments:
        result = run_experiment(
            exp['env_params'], 
            exp['algo_params'], 
            exp['name'],
            args.iterations
        )
        results.append(result)
    
    total_time = time.time() - total_start
    print(f"\n🏁 All experiments completed in {total_time:.1f}s")
    
    # Create comparison plots
    print("\n📊 Creating comparison plots...")
    create_comparison_plots(results, args.output)
    
    # Print summary table
    print("\n📋 EXPERIMENT SUMMARY")
    print("-" * 80)
    print(f"{'Experiment':<25} {'Final f':<10} {'Unsafe':<8} {'Violation':<10} {'Runtime':<8}")
    print("-" * 80)
    
    for result in results:
        metrics = result['final_metrics']
        print(f"{result['name']:<25} {metrics['f_value']:<10.4f} {metrics['unsafe_prob']:<8.3f} "
              f"{metrics['constraint_violation']:<10.4f} {result['runtime']:<8.1f}s")
    
    print("-" * 80)
    
    # Identify best performers
    best_f = min(results, key=lambda x: x['final_metrics']['f_value'])
    best_safe = min([r for r in results if r['final_metrics']['constraint_violation'] == 0], 
                   key=lambda x: x['final_metrics']['f_value'], default=None)
    
    print(f"\n🏆 WINNERS:")
    print(f"   Best objective: {best_f['name']} (f={best_f['final_metrics']['f_value']:.4f})")
    if best_safe:
        print(f"   Best safe solution: {best_safe['name']} (f={best_safe['final_metrics']['f_value']:.4f})")
    else:
        print(f"   No fully safe solutions found - consider relaxing τ or increasing iterations")
    
    print(f"\n💡 INSIGHTS:")
    
    # Analyze hazard placement effect
    baseline = next(r for r in results if r['name'] == 'Baseline')
    hazard_exp = next(r for r in results if 'Hazard Relocation' in r['name'])
    
    f_change = hazard_exp['final_metrics']['f_value'] / baseline['final_metrics']['f_value']
    print(f"   • Hazard relocation changed objective by {f_change:.2f}x")
    
    # Analyze slip effect
    slip_exp = next(r for r in results if 'High Slip' in r['name'])
    slip_change = slip_exp['final_metrics']['f_value'] / baseline['final_metrics']['f_value']
    print(f"   • High slip (0.2) changed objective by {slip_change:.2f}x")
    
    # Analyze constraint tightness
    tight_exp = next(r for r in results if 'Tight Constraint' in r['name'])
    if tight_exp['final_metrics']['constraint_violation'] > 0:
        print(f"   • Tight constraint (τ=0.02) was violated - algorithm struggled!")
    else:
        print(f"   • Tight constraint (τ=0.02) was satisfied successfully")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Try even tighter τ values (0.01, 0.005)")
    print(f"   2. Experiment with different learning rates")
    print(f"   3. Test on larger grids (7x7, 10x10)")
    print(f"   4. Add multiple constraints (fairness, entropy)")
    print(f"   5. Compare with other algorithms (FTL, OMD)")

if __name__ == "__main__":
    main() 