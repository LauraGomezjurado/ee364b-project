# Advanced Experiment Suite for Dual Optimization

This repository contains a comprehensive experiment suite for testing and analyzing dual optimization algorithms for imitation learning with safety constraints. The suite goes far beyond the original stress-test ideas and includes creative extensions for robust experimentation.

## Quick Start 

Get immediate results with these quick experiments:

```bash
python quick_wins.py

python quick_wins.py --iterations 5000

python interactive_dashboard.py

```

##  New Files Overview

### Core Experiment Infrastructure
- **`experiment_runner.py`** - Advanced experiment orchestrator with parallel execution
- **`advanced_algorithms.py`** - FTL, OMD, and adaptive optimization variants
- **`quick_wins.py`** - Immediate experiments to 
- **`interactive_dashboard.py`** - Real-time web dashboard for monitoring

### Extended Environments
- **`envs.py`** (extended) - New environment types:
  - `MazeWorld` - Complex navigation with walls
  - `StochasticGridWorld` - Noisy transitions and observations
  - `MultiObjectiveGridWorld` - Multiple rewards and penalties
  - `DynamicGridWorld` - Time-varying hazards

## Experiment Categories

### 1. Stress Tests (Enhanced)
Your original ideas plus creative extensions:

```python
# Original stress tests
- Tight τ progression: 0.05 → 0.002
- Hazard placement variations
- Slip parameter testing: 0.0 → 0.5
- Learning rate sensitivity

# Creative additions
- Multi-constraint optimization (safety + fairness + entropy)
- Adversarial maze environments
- Noise robustness testing
- Dynamic hazard adaptation
```

### 2. Algorithm Comparison Suite
Compare different optimization approaches:

```python
algorithms = ["ogd", "ftl", "omd", "adaptive"]
# Each with different convergence properties and robustness
```

### 3. Multi-Constraint Experiments
Beyond just safety constraints:

```python
constraints = [
    "safety",      # P(unsafe) ≤ τ
    "fairness",    # |P(group1) - P(group2)| ≤ ε  
    "entropy"      # H(d) ≥ threshold
]
```

### 4. Environment Complexity Tests
Test on increasingly complex environments:

```python
environments = [
    "GridWorld",           # Original 5x5 grid
    "MazeWorld",          # Complex navigation
    "StochasticGridWorld", # Noisy dynamics
    "DynamicGridWorld"     # Time-varying hazards
]
```

##  Usage Examples

### Basic Experiment Runner

```python
from experiment_runner import ExperimentRunner, ExperimentConfig

# Create experiment configuration
config = ExperimentConfig(
    name="my_experiment",
    tau=0.02,                    # Tight constraint
    slip=0.2,                    # High slip
    constraints=["safety", "fairness"],
    algorithm="adaptive"
)

# Run experiments
runner = ExperimentRunner()
results = runner.run_single_experiment(config)
```

### Parallel Experiment Suite

```python
# Run full stress test suite in parallel
python experiment_runner.py --suite stress --parallel --workers 4 --analyze
```

### Interactive Dashboard

```python
# Launch web dashboard
python interactive_dashboard.py

# Features:
# - Real-time convergence monitoring
# - Performance comparison plots
# - Trade-off analysis
# - Algorithm insights
```

##  Advanced Visualizations

The suite includes sophisticated visualization capabilities:

### 1. Performance Heatmaps
- Compare metrics across all experiments
- Color-coded performance indicators
- Automatic best/worst identification

### 2. Convergence Analysis
- Learning curves by experiment category
- Convergence rate comparison
- Stability analysis

### 3. Trade-off Plots
- Imitation quality vs constraint satisfaction
- Pareto frontier identification
- Multi-objective optimization results

### 4. Algorithm Comparison
- Runtime vs performance
- Convergence speed analysis
- Robustness to hyperparameters

##  Creative Extensions

### 1. Adaptive Learning Rates
The `AdaptiveOptimizer` automatically adjusts learning rates based on performance:

```python
# Increases η when making good progress
# Decreases η when oscillating or stuck
# Maintains stability bounds
```

### 2. Multi-Constraint Optimization
Handle multiple constraints simultaneously:

```python
# Safety: P(unsafe) ≤ τ
# Fairness: |visitation_difference| ≤ ε
# Entropy: H(occupancy) ≥ min_entropy
```

### 3. Noise Robustness
Test algorithm performance under various noise conditions:

```python
noise_types = [
    "observation_noise",    # Noisy state observations
    "transition_noise",     # Stochastic dynamics
    "reward_noise",         # Noisy reward signals
    "action_noise"          # Execution errors
]
```

### 4. Dynamic Environments
Environments that change over time:

```python
# Hazards move every N steps
# Rewards change location
# New obstacles appear
# Goal location shifts
```

##  Analysis Features

### Automatic Insights Generation
The system automatically identifies:
- Best performing configurations
- Constraint violation patterns
- Convergence characteristics
- Parameter sensitivity

### Statistical Analysis
- Confidence intervals
- Significance testing
- Effect size calculations
- Robustness metrics

### Export Capabilities
- JSON results for further analysis
- High-quality plots (PNG, PDF)
- LaTeX tables for papers
- Interactive HTML reports

## 🎛️ Configuration Options

### Environment Parameters
```python
env_params = {
    "size": 7,                    # Grid size
    "unsafe_cells": (6, 11, 16),  # Hazard locations
    "slip": 0.2,                  # Action noise
    "complexity": "high",         # For maze environments
    "noise": 0.05,               # Observation noise
    "change_frequency": 100       # For dynamic environments
}
```

### Algorithm Parameters
```python
algo_params = {
    "tau": 0.02,                 # Constraint threshold
    "eta_lam": 1.0,              # Lagrangian step size
    "eta_mu": 10.0,              # Constraint step size
    "algorithm": "adaptive",      # Optimization method
    "constraints": ["safety", "fairness"]
}
```

##  Getting Started Tonight

1. **Quick Wins** (5 minutes):
   ```bash
   python quick_wins.py
   ```

2. **Interactive Dashboard** (10 minutes):
   ```bash
   python interactive_dashboard.py
   # Open browser to localhost:8050
   ```

3. **Full Stress Test** (30 minutes):
   ```bash
   python experiment_runner.py --suite stress --parallel
   ```

##  Experiment Checklist

### Immediate (Tonight)
- [ ] Run `quick_wins.py` to get baseline results
- [ ] Test hazard relocation experiment
- [ ] Verify slip parameter effects
- [ ] Check tight constraint behavior

### Short-term (This Week)
- [ ] Run full stress test suite
- [ ] Compare different algorithms
- [ ] Test multi-constraint optimization
- [ ] Analyze parameter sensitivity

### Medium-term (Next Week)
- [ ] Scale to larger environments
- [ ] Test noise robustness
- [ ] Implement custom constraints
- [ ] Optimize hyperparameters

##  Dependencies

```bash
# Core dependencies
pip install numpy matplotlib tqdm scipy

# For interactive dashboard
pip install dash plotly pandas

# For advanced analysis
pip install seaborn scikit-learn
```

##  Expected Results

Based on the experiments, you should see:

1. **Tight Constraints**: Harder to satisfy, slower convergence
2. **Hazard Placement**: Affects trade-off between imitation and safety
3. **Slip Effects**: Degrades performance, requires adaptive methods
4. **Algorithm Differences**: FTL faster initially, OGD more stable
5. **Multi-Constraints**: Complex trade-offs, need careful tuning

##  Next Steps

After running these experiments:

1. **Scale Up**: Test on 10x10 grids, continuous control
2. **Real Environments**: CartPole, MountainCar, etc.
3. **Deep Learning**: Replace tabular with neural networks
4. **Theory**: Analyze convergence rates, regret bounds
5. **Applications**: Real-world safety-critical domains

##  Contributing

To add new experiments:

1. Create new `ExperimentConfig` in `experiment_runner.py`
2. Add environment variants in `envs.py`
3. Implement algorithm variants in `advanced_algorithms.py`
4. Update dashboard in `interactive_dashboard.py`

<!-- ##  References

This experiment suite implements and extends ideas from:
- Online convex optimization
- Imitation learning with constraints
- Multi-objective optimization
- Safe reinforcement learning

---
 -->
