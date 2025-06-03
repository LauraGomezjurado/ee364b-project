import argparse
import numpy as np
import json
import time
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import itertools

from envs import GridWorld, MazeWorld, StochasticGridWorld
from dual_loop import DualOptimizer
from utils import collect_occupancy, plot_curves
from advanced_algorithms import FTLOptimizer, OMDOptimizer, AdaptiveOptimizer

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run"""
    name: str
    env_type: str = "gridworld"
    env_params: Dict = None
    algorithm: str = "ogd"  # ogd, ftl, omd, adaptive
    tau: float = 0.05
    eta_lam: float = 1.0
    eta_mu: float = 10.0
    iterations: int = 5000
    constraints: List[str] = None  # ["safety", "fairness", "entropy"]
    slip: float = 0.0
    noise_level: float = 0.0
    seed: int = 42
    
    def __post_init__(self):
        if self.env_params is None:
            self.env_params = {}
        if self.constraints is None:
            self.constraints = ["safety"]

class ExperimentRunner:
    """Advanced experiment runner with parallel execution and analysis"""
    
    def __init__(self, output_dir: str = "experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        
    def create_stress_test_suite(self) -> List[ExperimentConfig]:
        """Create the stress-test experiments from your original plan + creative additions"""
        configs = []
        
        # Original stress tests
        base_config = ExperimentConfig(name="baseline")
        
        # 1. Tight tau progression
        for tau in [0.05, 0.02, 0.01, 0.005, 0.002]:
            configs.append(ExperimentConfig(
                name=f"tight_tau_{tau}",
                tau=tau,
                iterations=8000  # More iterations for tight constraints
            ))
        
        # 2. Hazard placement variations
        hazard_configs = [
            ((6, 11), "path_hazard"),
            ((1, 2, 3), "early_hazards"), 
            ((20, 21, 22, 23), "goal_blocking"),
            ((5, 10, 15, 20), "diagonal_hazards")
        ]
        for hazards, name in hazard_configs:
            configs.append(ExperimentConfig(
                name=f"hazard_{name}",
                env_params={"unsafe_cells": hazards}
            ))
        
        # 3. Slip variations with adaptive learning rates
        for slip in [0.0, 0.1, 0.2, 0.3, 0.5]:
            configs.append(ExperimentConfig(
                name=f"slip_{slip}",
                slip=slip,
                algorithm="adaptive"  # Use adaptive algorithm for noisy environments
            ))
        
        # 4. Multi-constraint experiments (CREATIVE ADDITION)
        configs.append(ExperimentConfig(
            name="multi_constraint_safety_fairness",
            constraints=["safety", "fairness"],
            env_params={"size": 7}  # Larger grid for fairness
        ))
        
        configs.append(ExperimentConfig(
            name="multi_constraint_all",
            constraints=["safety", "fairness", "entropy"],
            iterations=10000
        ))
        
        # 5. Algorithm comparison suite
        for algo in ["ogd", "ftl", "omd", "adaptive"]:
            configs.append(ExperimentConfig(
                name=f"algo_{algo}",
                algorithm=algo,
                iterations=6000
            ))
        
        # 6. Learning rate sensitivity (extended)
        eta_combinations = [(0.1, 1.0), (1.0, 10.0), (10.0, 100.0), (0.01, 0.1)]
        for eta_lam, eta_mu in eta_combinations:
            configs.append(ExperimentConfig(
                name=f"lr_lam{eta_lam}_mu{eta_mu}",
                eta_lam=eta_lam,
                eta_mu=eta_mu
            ))
        
        # 7. CREATIVE: Adversarial environments
        configs.append(ExperimentConfig(
            name="adversarial_maze",
            env_type="maze",
            env_params={"complexity": "high"},
            iterations=12000
        ))
        
        # 8. CREATIVE: Noise robustness
        for noise in [0.0, 0.01, 0.05, 0.1]:
            configs.append(ExperimentConfig(
                name=f"noise_{noise}",
                noise_level=noise,
                algorithm="adaptive"
            ))
        
        return configs
    
    def run_single_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Run a single experiment with the given configuration"""
        np.random.seed(config.seed)
        
        # Create environment
        if config.env_type == "gridworld":
            env = GridWorld(slip=config.slip, **config.env_params)
        elif config.env_type == "maze":
            env = MazeWorld(**config.env_params)
        elif config.env_type == "stochastic":
            env = StochasticGridWorld(noise=config.noise_level, **config.env_params)
        else:
            raise ValueError(f"Unknown environment type: {config.env_type}")
        
        # Create optimizer
        if config.algorithm == "ogd":
            optimizer = DualOptimizer(env, config)
        elif config.algorithm == "ftl":
            optimizer = FTLOptimizer(env, config)
        elif config.algorithm == "omd":
            optimizer = OMDOptimizer(env, config)
        elif config.algorithm == "adaptive":
            optimizer = AdaptiveOptimizer(env, config)
        else:
            raise ValueError(f"Unknown algorithm: {config.algorithm}")
        
        # Run optimization
        start_time = time.time()
        history = optimizer.optimize(config.iterations)
        runtime = time.time() - start_time
        
        # Add metadata
        result = {
            "config": asdict(config),
            "history": history,
            "runtime": runtime,
            "final_metrics": {
                "f_value": history["f"][-1] if history["f"] else float('inf'),
                "constraint_violation": max(0, history["unsafe"][-1] - config.tau) if history["unsafe"] else float('inf'),
                "convergence_iter": self._find_convergence_point(history)
            }
        }
        
        return result
    
    def run_experiment_suite(self, configs: List[ExperimentConfig], 
                           parallel: bool = True, max_workers: int = 4) -> Dict[str, Any]:
        """Run a suite of experiments with optional parallelization"""
        results = {}
        
        if parallel and len(configs) > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_config = {
                    executor.submit(self.run_single_experiment, config): config 
                    for config in configs
                }
                
                for future in tqdm(future_to_config, desc="Running experiments"):
                    config = future_to_config[future]
                    try:
                        result = future.result()
                        results[config.name] = result
                    except Exception as e:
                        print(f"Experiment {config.name} failed: {e}")
                        results[config.name] = {"error": str(e)}
        else:
            for config in tqdm(configs, desc="Running experiments"):
                try:
                    result = self.run_single_experiment(config)
                    results[config.name] = result
                except Exception as e:
                    print(f"Experiment {config.name} failed: {e}")
                    results[config.name] = {"error": str(e)}
        
        self.results = results
        return results
    
    def _find_convergence_point(self, history: Dict[str, List], 
                              tolerance: float = 1e-4, window: int = 100) -> int:
        """Find the iteration where the algorithm converged"""
        if not history.get("f") or len(history["f"]) < window:
            return -1
        
        f_values = np.array(history["f"])
        for i in range(window, len(f_values)):
            recent_std = np.std(f_values[i-window:i])
            if recent_std < tolerance:
                return i
        return -1
    
    def create_comprehensive_analysis(self):
        """Create comprehensive analysis and visualizations"""
        if not self.results:
            print("No results to analyze. Run experiments first.")
            return
        
        # Create analysis directory
        analysis_dir = self.output_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        # 1. Performance comparison heatmap
        self._create_performance_heatmap(analysis_dir)
        
        # 2. Convergence analysis
        self._create_convergence_analysis(analysis_dir)
        
        # 3. Algorithm comparison
        self._create_algorithm_comparison(analysis_dir)
        
        # 4. Parameter sensitivity analysis
        self._create_sensitivity_analysis(analysis_dir)
        
        # 5. Interactive dashboard data
        self._create_dashboard_data(analysis_dir)
        
        print(f"Analysis complete! Check {analysis_dir} for results.")
    
    def _create_performance_heatmap(self, output_dir: Path):
        """Create performance heatmap across different configurations"""
        # Extract performance metrics
        performance_data = []
        for name, result in self.results.items():
            if "error" not in result:
                performance_data.append({
                    "name": name,
                    "f_value": result["final_metrics"]["f_value"],
                    "constraint_violation": result["final_metrics"]["constraint_violation"],
                    "runtime": result["runtime"],
                    "convergence_iter": result["final_metrics"]["convergence_iter"]
                })
        
        if not performance_data:
            return
        
        # Create heatmap
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        names = [d["name"] for d in performance_data]
        metrics = ["f_value", "constraint_violation", "runtime", "convergence_iter"]
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            values = [d[metric] for d in performance_data]
            
            # Create a simple bar plot (can be enhanced to actual heatmap if needed)
            bars = ax.bar(range(len(names)), values)
            ax.set_title(f"{metric.replace('_', ' ').title()}")
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right')
            
            # Color bars based on performance
            if metric in ["f_value", "constraint_violation", "runtime"]:
                # Lower is better
                colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(values)))
            else:
                # Higher convergence iteration might be worse
                colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(values)))
            
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        
        plt.tight_layout()
        plt.savefig(output_dir / "performance_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_convergence_analysis(self, output_dir: Path):
        """Analyze convergence patterns across experiments"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot learning curves for different categories
        categories = {
            "tau_experiments": [name for name in self.results.keys() if "tight_tau" in name],
            "slip_experiments": [name for name in self.results.keys() if "slip_" in name],
            "algorithm_experiments": [name for name in self.results.keys() if "algo_" in name],
            "hazard_experiments": [name for name in self.results.keys() if "hazard_" in name]
        }
        
        for idx, (category, exp_names) in enumerate(categories.items()):
            if idx >= 4:
                break
            ax = axes[idx//2, idx%2]
            
            for name in exp_names[:5]:  # Limit to 5 curves per plot
                if name in self.results and "error" not in self.results[name]:
                    history = self.results[name]["history"]
                    if "f" in history:
                        ax.plot(history["f"], label=name, alpha=0.7)
            
            ax.set_title(category.replace("_", " ").title())
            ax.set_xlabel("Iteration (×100)")
            ax.set_ylabel("Objective Value")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(output_dir / "convergence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_algorithm_comparison(self, output_dir: Path):
        """Compare different algorithms"""
        algo_results = {}
        for name, result in self.results.items():
            if "algo_" in name and "error" not in result:
                algo_name = name.replace("algo_", "")
                algo_results[algo_name] = result
        
        if len(algo_results) < 2:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Convergence curves
        for algo, result in algo_results.items():
            history = result["history"]
            if "f" in history:
                axes[0].plot(history["f"], label=algo, linewidth=2)
        axes[0].set_title("Objective Convergence")
        axes[0].set_xlabel("Iteration (×100)")
        axes[0].set_ylabel("f(d)")
        axes[0].set_yscale('log')
        axes[0].legend()
        
        # Constraint satisfaction
        for algo, result in algo_results.items():
            history = result["history"]
            if "unsafe" in history:
                axes[1].plot(history["unsafe"], label=algo, linewidth=2)
        axes[1].axhline(y=0.05, color='red', linestyle='--', label='τ=0.05')
        axes[1].set_title("Constraint Satisfaction")
        axes[1].set_xlabel("Iteration (×100)")
        axes[1].set_ylabel("Unsafe Probability")
        axes[1].legend()
        
        # Runtime comparison
        algos = list(algo_results.keys())
        runtimes = [algo_results[algo]["runtime"] for algo in algos]
        bars = axes[2].bar(algos, runtimes)
        axes[2].set_title("Runtime Comparison")
        axes[2].set_ylabel("Time (seconds)")
        
        # Color bars
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        plt.savefig(output_dir / "algorithm_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_sensitivity_analysis(self, output_dir: Path):
        """Analyze parameter sensitivity"""
        # Learning rate sensitivity
        lr_results = {}
        for name, result in self.results.items():
            if "lr_" in name and "error" not in result:
                lr_results[name] = result
        
        if lr_results:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for name, result in lr_results.items():
                final_f = result["final_metrics"]["f_value"]
                final_constraint = result["final_metrics"]["constraint_violation"]
                
                # Extract learning rates from name
                parts = name.replace("lr_lam", "").replace("_mu", " ").split()
                if len(parts) >= 2:
                    eta_lam, eta_mu = float(parts[0]), float(parts[1])
                    
                    # Plot as scatter with size indicating constraint violation
                    size = max(10, 100 * final_constraint) if final_constraint > 0 else 10
                    ax.scatter(eta_lam, eta_mu, s=size, alpha=0.7, 
                             c=final_f, cmap='viridis', label=f"f={final_f:.3f}")
            
            ax.set_xlabel("η_λ (Lagrangian step size)")
            ax.set_ylabel("η_μ (Constraint step size)")
            ax.set_title("Learning Rate Sensitivity\n(Color=f value, Size=constraint violation)")
            ax.set_xscale('log')
            ax.set_yscale('log')
            
            plt.colorbar(ax.collections[0], ax=ax, label="Final f value")
            plt.tight_layout()
            plt.savefig(output_dir / "sensitivity_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_dashboard_data(self, output_dir: Path):
        """Create data for interactive dashboard"""
        dashboard_data = {
            "experiments": {},
            "summary": {
                "total_experiments": len(self.results),
                "successful_experiments": len([r for r in self.results.values() if "error" not in r]),
                "best_performance": None,
                "fastest_convergence": None
            }
        }
        
        best_f = float('inf')
        fastest_conv = float('inf')
        
        for name, result in self.results.items():
            if "error" not in result:
                dashboard_data["experiments"][name] = {
                    "config": result["config"],
                    "final_metrics": result["final_metrics"],
                    "runtime": result["runtime"],
                    "history_length": len(result["history"].get("f", []))
                }
                
                # Track best performance
                if result["final_metrics"]["f_value"] < best_f:
                    best_f = result["final_metrics"]["f_value"]
                    dashboard_data["summary"]["best_performance"] = name
                
                # Track fastest convergence
                conv_iter = result["final_metrics"]["convergence_iter"]
                if conv_iter > 0 and conv_iter < fastest_conv:
                    fastest_conv = conv_iter
                    dashboard_data["summary"]["fastest_convergence"] = name
        
        # Save dashboard data
        with open(output_dir / "dashboard_data.json", "w") as f:
            json.dump(dashboard_data, f, indent=2)
    
    def save_results(self, filename: str = "experiment_results.json"):
        """Save all results to file"""
        output_file = self.output_dir / filename
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Advanced Experiment Runner")
    parser.add_argument("--suite", choices=["stress", "scale", "theory", "custom"], 
                       default="stress", help="Experiment suite to run")
    parser.add_argument("--parallel", action="store_true", help="Run experiments in parallel")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--output", type=str, default="experiments", help="Output directory")
    parser.add_argument("--analyze", action="store_true", help="Run analysis after experiments")
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(args.output)
    
    if args.suite == "stress":
        configs = runner.create_stress_test_suite()
    else:
        # Add other suites as needed
        configs = runner.create_stress_test_suite()
    
    print(f"Running {len(configs)} experiments...")
    results = runner.run_experiment_suite(configs, parallel=args.parallel, max_workers=args.workers)
    
    runner.save_results()
    
    if args.analyze:
        runner.create_comprehensive_analysis()

if __name__ == "__main__":
    main() 