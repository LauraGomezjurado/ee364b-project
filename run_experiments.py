#!/usr/bin/env python3
"""
Master Experiment Runner

Simplified experiment suite for dual optimization testing:
1. Stress Tests - Binding constraints, stochastic slippage, multiple constraints
2. Algorithmic Variants - FTL, OMD, FTRL comparisons
3. Theory-Practice - Regret analysis and occupancy error bounds
4. Scaling - MazeWorld, StochasticGridWorld, computational scaling
"""

import argparse
import subprocess
import sys
import os

def run_stress_tests(args_str=""):
    """Run stress testing experiments"""
    print(" Running Stress Tests...")
    cmd = f"python stress_tests.py {args_str}"
    return subprocess.run(cmd, shell=True)

def run_algorithmic_variants(args_str=""):
    """Run algorithmic variant comparisons"""
    print("  Running Algorithmic Variants...")
    cmd = f"python algorithmic_variants.py {args_str}"
    return subprocess.run(cmd, shell=True)

def run_theory_practice(args_str=""):
    """Run theory-meets-practice verification"""
    print(" Running Theory-Practice Verification...")
    cmd = f"python theory_practice.py {args_str}"
    return subprocess.run(cmd, shell=True)

def run_scaling_experiments(args_str=""):
    """Run scaling experiments"""
    print(" Running Scaling Experiments...")
    cmd = f"python scaling_experiments.py {args_str}"
    return subprocess.run(cmd, shell=True)

def run_quick_demo():
    """Run a quick demonstration of all experiments"""
    print(" QUICK DEMO - Running subset of all experiments")
    print("=" * 60)
    
    # Quick stress test
    print("\n1. Quick Stress Test (binding constraints only)")
    subprocess.run("python stress_tests.py --test binding", shell=True)
    
    # Quick algorithm comparison
    print("\n2. Quick Algorithm Comparison (cost players only)")
    subprocess.run("python algorithmic_variants.py --test cost", shell=True)
    
    # Quick theory check
    print("\n3. Quick Theory Check (regret analysis only)")
    subprocess.run("python theory_practice.py --test regret", shell=True)
    
    # Quick scaling test
    print("\n4. Quick Scaling Test (maze only)")
    subprocess.run("python scaling_experiments.py --test maze", shell=True)
    
    print("\n Quick demo completed!")

def main():
    parser = argparse.ArgumentParser(
        description="Master Experiment Runner for Dual Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py --suite all          # Run all experiments
  python run_experiments.py --suite stress       # Run only stress tests
  python run_experiments.py --suite quick        # Quick demo
  python run_experiments.py --suite stress --test binding  # Specific test
        """
    )
    
    parser.add_argument("--suite", 
                       choices=["stress", "algorithms", "theory", "scaling", "all", "quick"],
                       default="quick",
                       help="Which experiment suite to run")
    
    parser.add_argument("--test", 
                       help="Specific test within suite (varies by suite)")
    
    parser.add_argument("--verbose", "-v", 
                       action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Check if required files exist
    required_files = [
        "stress_tests.py",
        "algorithmic_variants.py", 
        "theory_practice.py",
        "scaling_experiments.py",
        "dual_loop.py",
        "envs.py",
        "utils.py"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f" Missing required files: {missing_files}")
        sys.exit(1)
    
    print(" DUAL OPTIMIZATION EXPERIMENT SUITE")
    print("=" * 50)
    
    # Build arguments string for sub-scripts
    sub_args = ""
    if args.test:
        sub_args += f"--test {args.test}"
    if args.verbose:
        sub_args += " --verbose"
    
    # Run experiments based on suite selection
    if args.suite == "quick":
        run_quick_demo()
        
    elif args.suite == "stress":
        result = run_stress_tests(sub_args)
        if result.returncode != 0:
            print(" Stress tests failed")
            sys.exit(1)
            
    elif args.suite == "algorithms":
        result = run_algorithmic_variants(sub_args)
        if result.returncode != 0:
            print(" Algorithmic variants failed")
            sys.exit(1)
            
    elif args.suite == "theory":
        result = run_theory_practice(sub_args)
        if result.returncode != 0:
            print(" Theory-practice verification failed")
            sys.exit(1)
            
    elif args.suite == "scaling":
        result = run_scaling_experiments(sub_args)
        if result.returncode != 0:
            print(" Scaling experiments failed")
            sys.exit(1)
            
    elif args.suite == "all":
        print(" Running complete experiment suite...")
        
        suites = [
            ("Stress Tests", lambda: run_stress_tests(sub_args)),
            ("Algorithmic Variants", lambda: run_algorithmic_variants(sub_args)),
            ("Theory-Practice", lambda: run_theory_practice(sub_args)),
            ("Scaling Experiments", lambda: run_scaling_experiments(sub_args))
        ]
        
        results = {}
        for suite_name, suite_func in suites:
            print(f"\n{'='*20} {suite_name} {'='*20}")
            result = suite_func()
            results[suite_name] = result.returncode == 0
            
            if result.returncode != 0:
                print(f" {suite_name} encountered issues")
            else:
                print(f" {suite_name} completed successfully")
        
        # Summary
        print(f"\n{'='*60}")
        print(" EXPERIMENT SUITE SUMMARY")
        print(f"{'='*60}")
        
        for suite_name, success in results.items():
            status = " PASSED" if success else " FAILED"
            print(f"  {suite_name:<25} {status}")
        
        total_passed = sum(results.values())
        total_suites = len(results)
        print(f"\nOverall: {total_passed}/{total_suites} suites passed")
        
        if total_passed == total_suites:
            print(" All experiments completed successfully!")
        else:
            print(" Some experiments had issues - check logs above")
            sys.exit(1)
    
    print(f"\n Experiment suite '{args.suite}' completed!")

if __name__ == "__main__":
    main() 