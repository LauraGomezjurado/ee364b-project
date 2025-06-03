import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GridWorld(gym.Env):
    """
    5×5 grid with deterministic moves and two unsafe cells.

    State  s = row*size + col  ∈ {0,…,nS-1}
    Action a ∈ {0:UP, 1:DOWN, 2:LEFT, 3:RIGHT}
    Terminal when agent reaches bottom-right goal.
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, size: int = 5, unsafe_cells=(12, 17), slip=0.0):
        super().__init__()
        self.size, self.nS, self.nA = size, size * size, 4
        self.action_space  = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.unsafe_mask = np.zeros(self.nS, dtype=bool)
        self.unsafe_mask[list(unsafe_cells)] = True
        self.goal = self.nS - 1
        self.slip = slip
        self._build_transition_matrix()

    #  core Gym API 
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.s = 0                      # start top-left
        return self.s, {}               # obs, info

    def step(self, a):
        s = self.s
        if self.np_random.random() < self.slip:
            a = self.np_random.integers(0, self.nA)      # random slip
        r, terminated, truncated = 0.0, False, False
        s_ = self._next_state(s, a)
        if s_ == self.goal:
            terminated, r = True, 1.0
        self.s = s_
        return s_, r, terminated, truncated, {}


    def _next_state(self, s, a):
        row, col = divmod(s, self.size)
        if   a == 0 and row > 0:            row -= 1       # UP
        elif a == 1 and row < self.size-1:  row += 1       # DOWN
        elif a == 2 and col > 0:            col -= 1       # LEFT
        elif a == 3 and col < self.size-1:  col += 1       # RIGHT
        return row * self.size + col

    def _build_transition_matrix(self):
        # (nS, nA, nS)   for exact occupancy check if wanted
        P = np.zeros((self.nS, self.nA, self.nS))
        for s in range(self.nS):
            for a in range(self.nA):
                s_ = self._next_state(s, a)
                P[s, a, s_] = 1.0
        self.P = P


class MazeWorld(gym.Env):
    """
    More complex maze environment with walls and multiple hazards.
    Designed to test algorithm robustness in complex navigation scenarios.
    """
    metadata = {"render_modes": ["ansi"]}
    
    def __init__(self, size: int = 7, complexity: str = "medium", slip: float = 0.0):
        super().__init__()
        self.size = size
        self.nS = size * size
        self.nA = 4
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)
        self.slip = slip
        
        # Create maze layout based on complexity
        self._create_maze_layout(complexity)
        self.goal = self.nS - 1  # Bottom-right corner
        
    def _create_maze_layout(self, complexity: str):
        """Create maze with walls and hazards based on complexity level"""
        self.walls = set()
        self.unsafe_mask = np.zeros(self.nS, dtype=bool)
        
        if complexity == "low":
            # Simple maze with few walls
            self.walls = {9, 16}  # Just a couple of walls
            unsafe_cells = [10, 20]
        elif complexity == "medium":
            # Moderate complexity
            self.walls = {8, 9, 15, 16, 22, 23, 29, 30}
            unsafe_cells = [10, 17, 24, 31]
        elif complexity == "high":
            # Complex maze with many obstacles
            self.walls = {7, 8, 9, 14, 16, 21, 23, 28, 30, 35, 37, 42}
            unsafe_cells = [10, 11, 17, 18, 24, 25, 31, 32]
        else:
            raise ValueError(f"Unknown complexity: {complexity}")
        
        # Set unsafe cells
        for cell in unsafe_cells:
            if cell < self.nS:
                self.unsafe_mask[cell] = True
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.s = 0  # Start at top-left
        return self.s, {}
    
    def step(self, a):
        s = self.s
        
        # Apply slip
        if self.np_random.random() < self.slip:
            a = self.np_random.integers(0, self.nA)
        
        # Try to move
        s_next = self._next_state(s, a)
        
        # Check if next state is a wall
        if s_next in self.walls:
            s_next = s  # Stay in place if hitting wall
        
        r, terminated, truncated = 0.0, False, False
        if s_next == self.goal:
            terminated, r = True, 1.0
        
        self.s = s_next
        return s_next, r, terminated, truncated, {}
    
    def _next_state(self, s, a):
        """Calculate next state without considering walls"""
        row, col = divmod(s, self.size)
        if   a == 0 and row > 0:            row -= 1       # UP
        elif a == 1 and row < self.size-1:  row += 1       # DOWN
        elif a == 2 and col > 0:            col -= 1       # LEFT
        elif a == 3 and col < self.size-1:  col += 1       # RIGHT
        return row * self.size + col


class StochasticGridWorld(gym.Env):
    """
    GridWorld with stochastic transitions and observation noise.
    Tests algorithm robustness to uncertainty.
    """
    metadata = {"render_modes": ["ansi"]}
    
    def __init__(self, size: int = 5, unsafe_cells=(12, 17), 
                 slip: float = 0.1, noise: float = 0.05, 
                 transition_noise: float = 0.1):
        super().__init__()
        self.size = size
        self.nS = size * size
        self.nA = 4
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)
        
        self.unsafe_mask = np.zeros(self.nS, dtype=bool)
        self.unsafe_mask[list(unsafe_cells)] = True
        self.goal = self.nS - 1
        
        # Noise parameters
        self.slip = slip
        self.noise = noise  # Observation noise
        self.transition_noise = transition_noise  # Transition stochasticity
        
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.s = 0
        return self._add_observation_noise(self.s), {}
    
    def step(self, a):
        s = self.s
        
        # Action noise (slip)
        if self.np_random.random() < self.slip:
            a = self.np_random.integers(0, self.nA)
        
        # Stochastic transitions
        intended_next = self._next_state(s, a)
        
        if self.np_random.random() < self.transition_noise:
            # Random transition to adjacent state
            adjacent_states = self._get_adjacent_states(s)
            s_next = self.np_random.choice(adjacent_states)
        else:
            s_next = intended_next
        
        r, terminated, truncated = 0.0, False, False
        if s_next == self.goal:
            terminated, r = True, 1.0
        
        self.s = s_next
        
        # Add observation noise
        observed_state = self._add_observation_noise(s_next)
        
        return observed_state, r, terminated, truncated, {}
    
    def _next_state(self, s, a):
        """Deterministic next state calculation"""
        row, col = divmod(s, self.size)
        if   a == 0 and row > 0:            row -= 1
        elif a == 1 and row < self.size-1:  row += 1
        elif a == 2 and col > 0:            col -= 1
        elif a == 3 and col < self.size-1:  col += 1
        return row * self.size + col
    
    def _get_adjacent_states(self, s):
        """Get all valid adjacent states"""
        adjacent = []
        row, col = divmod(s, self.size)
        
        # Check all four directions
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.size and 0 <= new_col < self.size:
                adjacent.append(new_row * self.size + new_col)
        
        return adjacent if adjacent else [s]  # Return current state if no adjacent
    
    def _add_observation_noise(self, true_state):
        """Add noise to observations"""
        if self.np_random.random() < self.noise:
            # Return a random nearby state
            adjacent = self._get_adjacent_states(true_state)
            return self.np_random.choice(adjacent)
        return true_state


class MultiObjectiveGridWorld(gym.Env):
    """
    GridWorld with multiple objectives and constraints.
    Useful for testing multi-constraint optimization.
    """
    metadata = {"render_modes": ["ansi"]}
    
    def __init__(self, size: int = 6, unsafe_cells=(10, 15, 20), 
                 reward_cells=(8, 14), penalty_cells=(12, 18)):
        super().__init__()
        self.size = size
        self.nS = size * size
        self.nA = 4
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)
        
        # Different types of cells
        self.unsafe_mask = np.zeros(self.nS, dtype=bool)
        self.unsafe_mask[list(unsafe_cells)] = True
        
        self.reward_mask = np.zeros(self.nS, dtype=bool)
        self.reward_mask[list(reward_cells)] = True
        
        self.penalty_mask = np.zeros(self.nS, dtype=bool)
        self.penalty_mask[list(penalty_cells)] = True
        
        self.goal = self.nS - 1
        
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.s = 0
        return self.s, {}
    
    def step(self, a):
        s = self.s
        s_next = self._next_state(s, a)
        
        # Multi-objective rewards
        r = 0.0
        if s_next == self.goal:
            r += 10.0  # Goal reward
        if self.reward_mask[s_next]:
            r += 1.0   # Bonus reward
        if self.penalty_mask[s_next]:
            r -= 2.0   # Penalty
        
        terminated = (s_next == self.goal)
        truncated = False
        
        self.s = s_next
        return s_next, r, terminated, truncated, {}
    
    def _next_state(self, s, a):
        row, col = divmod(s, self.size)
        if   a == 0 and row > 0:            row -= 1
        elif a == 1 and row < self.size-1:  row += 1
        elif a == 2 and col > 0:            col -= 1
        elif a == 3 and col < self.size-1:  col += 1
        return row * self.size + col


class DynamicGridWorld(gym.Env):
    """
    GridWorld where hazards and rewards change over time.
    Tests adaptation capabilities of algorithms.
    """
    metadata = {"render_modes": ["ansi"]}
    
    def __init__(self, size: int = 5, change_frequency: int = 100):
        super().__init__()
        self.size = size
        self.nS = size * size
        self.nA = 4
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)
        
        self.change_frequency = change_frequency
        self.step_count = 0
        self.goal = self.nS - 1
        
        # Initialize with random hazards
        self._update_hazards()
        
    def _update_hazards(self):
        """Randomly update hazard locations"""
        n_hazards = np.random.randint(2, 5)  # 2-4 hazards
        hazard_locations = np.random.choice(
            range(1, self.nS - 1),  # Exclude start and goal
            size=n_hazards, 
            replace=False
        )
        
        self.unsafe_mask = np.zeros(self.nS, dtype=bool)
        self.unsafe_mask[hazard_locations] = True
        
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.s = 0
        self.step_count = 0
        return self.s, {}
    
    def step(self, a):
        self.step_count += 1
        
        # Update hazards periodically
        if self.step_count % self.change_frequency == 0:
            self._update_hazards()
        
        s = self.s
        s_next = self._next_state(s, a)
        
        r, terminated, truncated = 0.0, False, False
        if s_next == self.goal:
            terminated, r = True, 1.0
        
        self.s = s_next
        return s_next, r, terminated, truncated, {}
    
    def _next_state(self, s, a):
        row, col = divmod(s, self.size)
        if   a == 0 and row > 0:            row -= 1
        elif a == 1 and row < self.size-1:  row += 1
        elif a == 2 and col > 0:            col -= 1
        elif a == 3 and col < self.size-1:  col += 1
        return row * self.size + col
