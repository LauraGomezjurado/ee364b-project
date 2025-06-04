#!/bin/bash


# Quick demonstration (recommended first run)
python run_experiments.py --suite quick

# Run specific experiment suites
# python run_experiments.py --suite stress
# python run_experiments.py --suite algorithms
python run_experiments.py --suite theory
python run_experiments.py --suite scaling

# Run everything
python run_experiments.py --suite all

# Run specific tests within suites
python run_experiments.py --suite stress --test binding
python run_experiments.py --suite theory --test regret