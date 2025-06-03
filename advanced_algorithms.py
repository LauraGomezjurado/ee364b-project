import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Any
from tqdm import trange
import scipy.optimize as opt

from utils import collect_occupancy, DEF_GAMMA

class BaseOptimizer(ABC):
    """Base class for dual optimization algorithms"""
    
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.setup_constraints()
        
    def setup_constraints(self):
        """Setup constraint functions based on config"""
        self.constraints = {}
        
        # Safety constraint (always present)
        unsafe_idx = np.repeat(self.env.unsafe_mask, self.env.nA)
        self.constraints["safety"] = {
            "vector": unsafe_idx.astype(float),
            "threshold": self.config.tau,
            "dual_var": 0.0
        }
        
        # Fairness constraint (if specified)
        if "fairness" in self.config.constraints:
            # Create fairness constraint: difference in visitation between "groups"
            # For simplicity, divide states into two groups based on position
            fairness_vec = np.zeros(self.env.nS * self.env.nA)
            for s in range(self.env.nS):
                row, col = divmod(s, self.env.size)
                # Group 1: left half, Group 2: right half
                group_indicator = 1.0 if col < self.env.size // 2 else -1.0
                for a in range(self.env.nA):
                    fairness_vec[s * self.env.nA + a] = group_indicator
            
            self.constraints["fairness"] = {
                "vector": fairness_vec,
                "threshold": 0.1,  # Max allowed difference
                "dual_var": 0.0
            }
        
        # Entropy constraint (if specified)
        if "entropy" in self.config.constraints:
            # This will be handled differently as it's not linear
            self.constraints["entropy"] = {
                "threshold": -2.0,  # Min entropy
                "dual_var": 0.0
            }
    
    def expert_policy(self, s):
        """Expert policy for imitation"""
        row, col = divmod(s, self.env.size)
        if col < self.env.size-1: return 3          # go RIGHT
        if row < self.env.size-1: return 1          # then DOWN
        return 0                                    # arbitrary at goal
    
    def solve_soft_q(self, reward_vec, n_iter=100, temp=1.0):
        """Solve soft Q-learning with given reward"""
        Q = np.zeros((self.env.nS, self.env.nA))
        alpha = 0.1
        
        for _ in range(n_iter):
            for s in range(self.env.nS):
                for a in range(self.env.nA):
                    s_next = self.env._next_state(s, a)
                    max_q = np.max(Q[s_next])
                    Q[s, a] = (1-alpha)*Q[s, a] + alpha*(reward_vec[s*self.env.nA+a] + 
                                                         DEF_GAMMA * max_q)
        
        # Softmax policy
        pi = np.exp(Q/temp) / np.sum(np.exp(Q/temp), axis=1, keepdims=True)
        return pi
    
    def rollout_policy(self, pi, n_episodes=50):
        """Rollout policy to get occupancy"""
        d = np.zeros(self.env.nS * self.env.nA)
        for _ in range(n_episodes):
            s, _ = self.env.reset()
            t, done = 0, False
            while not done:
                a = np.random.choice(self.env.nA, p=pi[s])
                s_next, _, done, _, _ = self.env.step(a)
                d[s*self.env.nA + a] += (1-DEF_GAMMA)*(DEF_GAMMA**t)
                s, t = s_next, t+1
        return d / n_episodes
    
    def compute_entropy(self, d):
        """Compute entropy of occupancy distribution"""
        d_normalized = d / (np.sum(d) + 1e-8)
        d_normalized = np.maximum(d_normalized, 1e-8)  # Avoid log(0)
        return -np.sum(d_normalized * np.log(d_normalized))
    
    @abstractmethod
    def optimize(self, iterations: int) -> Dict[str, List]:
        """Run the optimization algorithm"""
        pass

class DualOptimizer(BaseOptimizer):
    """Original OGD-based dual optimizer"""
    
    def optimize(self, iterations: int) -> Dict[str, List]:
        # Get expert occupancy
        d_E = collect_occupancy(self.env, self.expert_policy, n_episodes=1000)
        
        # Initialize dual variables
        lam = np.zeros_like(d_E)
        dual_vars = {name: 0.0 for name in self.constraints.keys()}
        avg_d = np.zeros_like(d_E)
        
        history = defaultdict(list)
        
        for k in trange(1, iterations+1):
            # Policy player: solve for best response
            reward = -lam.copy()
            
            # Add constraint penalties
            for name, constraint in self.constraints.items():
                if name == "entropy":
                    continue  # Handle entropy separately
                reward -= dual_vars[name] * constraint["vector"]
            
            pi = self.solve_soft_q(reward, n_iter=80)
            d = self.rollout_policy(pi)
            
            # Dual updates (OGD)
            lam += self.config.eta_lam * 2*(d - d_E)
            
            for name, constraint in self.constraints.items():
                if name == "safety":
                    violation = np.dot(d, constraint["vector"]) - constraint["threshold"]
                    dual_vars[name] = max(0.0, dual_vars[name] + self.config.eta_mu * violation)
                elif name == "fairness":
                    violation = abs(np.dot(d, constraint["vector"])) - constraint["threshold"]
                    dual_vars[name] = max(0.0, dual_vars[name] + self.config.eta_mu * violation)
                elif name == "entropy":
                    entropy = self.compute_entropy(d)
                    violation = constraint["threshold"] - entropy  # Want entropy >= threshold
                    dual_vars[name] = max(0.0, dual_vars[name] + self.config.eta_mu * violation)
            
            # Running average
            avg_d = ((k-1)*avg_d + d) / k
            
            # Logging
            if k % 100 == 0:
                f_val = np.sum((avg_d - d_E)**2)
                unsafe = np.dot(avg_d, self.constraints["safety"]["vector"])
                
                history["f"].append(f_val)
                history["unsafe"].append(unsafe)
                history["dual_vars"].append(dual_vars.copy())
                
                if "fairness" in self.constraints:
                    fairness_gap = abs(np.dot(avg_d, self.constraints["fairness"]["vector"]))
                    history["fairness"].append(fairness_gap)
                
                if "entropy" in self.constraints:
                    entropy = self.compute_entropy(avg_d)
                    history["entropy"].append(entropy)
        
        return history

class FTLOptimizer(BaseOptimizer):
    """Follow-the-Leader optimizer"""
    
    def optimize(self, iterations: int) -> Dict[str, List]:
        d_E = collect_occupancy(self.env, self.expert_policy, n_episodes=1000)
        
        # Store all past occupancies for FTL
        past_occupancies = []
        history = defaultdict(list)
        
        for k in trange(1, iterations+1):
            if k == 1:
                # First iteration: use OGD step
                lam = np.zeros_like(d_E)
                dual_vars = {name: 0.0 for name in self.constraints.keys()}
            else:
                # FTL: solve for best dual variables given all past occupancies
                avg_past_d = np.mean(past_occupancies, axis=0)
                
                # Simple FTL update (could be more sophisticated)
                lam = 2 * (avg_past_d - d_E)
                
                for name, constraint in self.constraints.items():
                    if name == "safety":
                        violation = np.dot(avg_past_d, constraint["vector"]) - constraint["threshold"]
                        dual_vars[name] = max(0.0, violation / self.config.eta_mu)
            
            # Solve policy
            reward = -lam.copy()
            for name, constraint in self.constraints.items():
                if name != "entropy":
                    reward -= dual_vars[name] * constraint["vector"]
            
            pi = self.solve_soft_q(reward, n_iter=80)
            d = self.rollout_policy(pi)
            past_occupancies.append(d)
            
            # Logging
            if k % 100 == 0:
                avg_d = np.mean(past_occupancies, axis=0)
                f_val = np.sum((avg_d - d_E)**2)
                unsafe = np.dot(avg_d, self.constraints["safety"]["vector"])
                
                history["f"].append(f_val)
                history["unsafe"].append(unsafe)
        
        return history

class OMDOptimizer(BaseOptimizer):
    """Online Mirror Descent optimizer"""
    
    def __init__(self, env, config):
        super().__init__(env, config)
        self.regularizer_strength = 0.01
    
    def mirror_map(self, x):
        """Mirror map for OMD (using entropy regularization)"""
        return np.log(np.maximum(x, 1e-8))
    
    def inverse_mirror_map(self, y):
        """Inverse mirror map"""
        return np.exp(y)
    
    def optimize(self, iterations: int) -> Dict[str, List]:
        d_E = collect_occupancy(self.env, self.expert_policy, n_episodes=1000)
        
        # Initialize in mirror space
        lam_mirror = np.zeros_like(d_E)
        dual_vars = {name: 0.0 for name in self.constraints.keys()}
        
        history = defaultdict(list)
        
        for k in trange(1, iterations+1):
            # Convert from mirror space
            lam = self.inverse_mirror_map(lam_mirror)
            lam = lam / (np.sum(lam) + 1e-8)  # Normalize
            
            # Solve policy
            reward = -lam * len(d_E)  # Scale back up
            for name, constraint in self.constraints.items():
                if name != "entropy":
                    reward -= dual_vars[name] * constraint["vector"]
            
            pi = self.solve_soft_q(reward, n_iter=80)
            d = self.rollout_policy(pi)
            
            # OMD updates in mirror space
            gradient = 2 * (d - d_E)
            lam_mirror -= self.config.eta_lam * gradient
            
            # Constraint dual updates (standard)
            for name, constraint in self.constraints.items():
                if name == "safety":
                    violation = np.dot(d, constraint["vector"]) - constraint["threshold"]
                    dual_vars[name] = max(0.0, dual_vars[name] + self.config.eta_mu * violation)
            
            # Logging
            if k % 100 == 0:
                # Use running average for evaluation
                if not hasattr(self, 'avg_d'):
                    self.avg_d = d.copy()
                else:
                    self.avg_d = ((k//100-1)*self.avg_d + d) / (k//100)
                
                f_val = np.sum((self.avg_d - d_E)**2)
                unsafe = np.dot(self.avg_d, self.constraints["safety"]["vector"])
                
                history["f"].append(f_val)
                history["unsafe"].append(unsafe)
        
        return history

class AdaptiveOptimizer(BaseOptimizer):
    """Adaptive optimizer that adjusts learning rates based on performance"""
    
    def __init__(self, env, config):
        super().__init__(env, config)
        self.adaptive_window = 200
        self.lr_increase_factor = 1.1
        self.lr_decrease_factor = 0.9
        self.performance_threshold = 0.01
    
    def optimize(self, iterations: int) -> Dict[str, List]:
        d_E = collect_occupancy(self.env, self.expert_policy, n_episodes=1000)
        
        # Adaptive learning rates
        eta_lam = self.config.eta_lam
        eta_mu = self.config.eta_mu
        
        lam = np.zeros_like(d_E)
        dual_vars = {name: 0.0 for name in self.constraints.keys()}
        avg_d = np.zeros_like(d_E)
        
        # Performance tracking
        recent_f_values = []
        recent_violations = []
        
        history = defaultdict(list)
        
        for k in trange(1, iterations+1):
            # Solve policy
            reward = -lam.copy()
            for name, constraint in self.constraints.items():
                if name != "entropy":
                    reward -= dual_vars[name] * constraint["vector"]
            
            # Add noise robustness if specified
            if self.config.noise_level > 0:
                reward += np.random.normal(0, self.config.noise_level, size=reward.shape)
            
            pi = self.solve_soft_q(reward, n_iter=80)
            d = self.rollout_policy(pi)
            
            # Standard dual updates
            lam += eta_lam * 2*(d - d_E)
            
            for name, constraint in self.constraints.items():
                if name == "safety":
                    violation = np.dot(d, constraint["vector"]) - constraint["threshold"]
                    dual_vars[name] = max(0.0, dual_vars[name] + eta_mu * violation)
            
            # Running average
            avg_d = ((k-1)*avg_d + d) / k
            
            # Adaptive learning rate adjustment
            if k % self.adaptive_window == 0 and k > self.adaptive_window:
                current_f = np.sum((avg_d - d_E)**2)
                current_violation = max(0, np.dot(avg_d, self.constraints["safety"]["vector"]) - self.config.tau)
                
                recent_f_values.append(current_f)
                recent_violations.append(current_violation)
                
                if len(recent_f_values) >= 2:
                    # Check if we're making progress
                    f_improvement = recent_f_values[-2] - recent_f_values[-1]
                    violation_improvement = recent_violations[-2] - recent_violations[-1]
                    
                    if f_improvement > self.performance_threshold or violation_improvement > 0:
                        # Good progress, increase learning rates
                        eta_lam *= self.lr_increase_factor
                        eta_mu *= self.lr_increase_factor
                    else:
                        # Poor progress, decrease learning rates
                        eta_lam *= self.lr_decrease_factor
                        eta_mu *= self.lr_decrease_factor
                    
                    # Keep learning rates in reasonable bounds
                    eta_lam = np.clip(eta_lam, 0.001, 100.0)
                    eta_mu = np.clip(eta_mu, 0.001, 1000.0)
            
            # Logging
            if k % 100 == 0:
                f_val = np.sum((avg_d - d_E)**2)
                unsafe = np.dot(avg_d, self.constraints["safety"]["vector"])
                
                history["f"].append(f_val)
                history["unsafe"].append(unsafe)
                history["eta_lam"].append(eta_lam)
                history["eta_mu"].append(eta_mu)
        
        return history 