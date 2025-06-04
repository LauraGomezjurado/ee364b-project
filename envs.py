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
        self.size = size # Store size as an attribute
        self.nS = size * size
        self.nA = 4
        self.action_space  = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.unsafe_mask = np.zeros(self.nS, dtype=bool)
        if unsafe_cells and self.nS > 0: # Ensure unsafe_cells can be applied
            valid_unsafe_cells = [cell for cell in unsafe_cells if cell < self.nS]
            self.unsafe_mask[list(valid_unsafe_cells)] = True
            
        self.goal = self.nS - 1 if self.nS > 0 else 0
        self.slip = slip
        # self._build_transition_matrix() # Not strictly needed if not using self.P

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.s = 0
        return self.s, {}

    def step(self, a):
        s = self.s
        if self.np_random.random() < self.slip:
            a = self.np_random.integers(0, self.nA)
        r, terminated, truncated = 0.0, False, False
        s_ = self._next_state(s, a)
        if s_ == self.goal:
            terminated, r = True, 1.0
        self.s = s_
        return s_, r, terminated, truncated, {}

    def _next_state(self, s, a):
        row, col = divmod(s, self.size)
        if   a == 0 and row > 0:            row -= 1
        elif a == 1 and row < self.size-1:  row += 1
        elif a == 2 and col > 0:            col -= 1
        elif a == 3 and col < self.size-1:  col += 1
        return row * self.size + col

    def _build_transition_matrix(self):
        P = np.zeros((self.nS, self.nA, self.nS))
        for s in range(self.nS):
            for a in range(self.nA):
                s_ = self._next_state(s, a)
                P[s, a, s_] = 1.0
        self.P = P


class MazeWorld(gym.Env):
    metadata = {"render_modes": ["ansi"]}
    
    def __init__(self, size: int = 7, complexity: str = "medium", slip: float = 0.0):
        super().__init__()
        self.size = size
        self.nS = size * size
        self.nA = 4
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)
        self.slip = slip
        
        self._create_maze_layout(complexity)
        self.goal = self.nS - 1
        
    def _create_maze_layout(self, complexity: str):
        self.walls = set()
        self.unsafe_mask = np.zeros(self.nS, dtype=bool)
        
        if self.size == 7: # Examples for a 7x7 grid
            if complexity == "low":
                self.walls = {9, 16, 30, 37}
                unsafe_cells = [10, 24, 38]
            elif complexity == "medium":
                self.walls = {8, 9, 15, 16, 22, 23, 29, 30, 36, 37}
                unsafe_cells = [10, 17, 24, 31, 38]
            elif complexity == "high":
                self.walls = {1, 2, 3, 8, 9, 10, 15, 17, 22, 24, 29, 31, 36, 38, 43, 44, 45}
                unsafe_cells = [16, 23, 30, 37, 12, 19, 26, 33]
            else:
                raise ValueError(f"Unknown complexity: {complexity} for size {self.size}")
        elif self.size == 5: # Fallback for 5x5 if MazeWorld is used with it
             self.walls = {7,8,12,13}
             unsafe_cells = [6, 11, 16]
        else: # Generic, less interesting for other sizes
            base_unsafe = [self.size + 2, 2 * self.size + 2]
            unsafe_cells = [c for c in base_unsafe if c < self.nS -1]


        for cell in unsafe_cells:
            if 0 <= cell < self.nS:
                self.unsafe_mask[cell] = True
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.s = 0
        return self.s, {}
    
    def step(self, a):
        s = self.s
        if self.np_random.random() < self.slip:
            a = self.np_random.integers(0, self.nA)
        
        s_next_candidate = self._next_state(s, a)
        
        if s_next_candidate in self.walls:
            s_next = s
        else:
            s_next = s_next_candidate
        
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


class StochasticGridWorld(gym.Env):
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
        if unsafe_cells and self.nS > 0:
            valid_unsafe_cells = [cell for cell in unsafe_cells if cell < self.nS]
            self.unsafe_mask[list(valid_unsafe_cells)] = True

        self.goal = self.nS - 1 if self.nS > 0 else 0
        
        self.slip = slip
        self.noise = noise
        self.transition_noise = transition_noise
        
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.s = 0
        return self._add_observation_noise(self.s), {}
    
    def step(self, a):
        s = self.s
        if self.np_random.random() < self.slip:
            a = self.np_random.integers(0, self.nA)
        
        intended_next = self._next_state(s, a)
        
        if self.np_random.random() < self.transition_noise:
            adjacent_states = self._get_adjacent_states(s)
            s_next = self.np_random.choice(adjacent_states) if adjacent_states else intended_next
        else:
            s_next = intended_next
        
        r, terminated, truncated = 0.0, False, False
        if s_next == self.goal:
            terminated, r = True, 1.0
        
        self.s = s_next
        observed_state = self._add_observation_noise(s_next)
        return observed_state, r, terminated, truncated, {}
    
    def _next_state(self, s, a):
        row, col = divmod(s, self.size)
        if   a == 0 and row > 0:            row -= 1
        elif a == 1 and row < self.size-1:  row += 1
        elif a == 2 and col > 0:            col -= 1
        elif a == 3 and col < self.size-1:  col += 1
        return row * self.size + col
    
    def _get_adjacent_states(self, s):
        adjacent = []
        row, col = divmod(s, self.size)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.size and 0 <= new_col < self.size:
                adjacent.append(new_row * self.size + new_col)
        return adjacent if adjacent else [s]
    
    def _add_observation_noise(self, true_state):
        if self.np_random.random() < self.noise:
            adjacent = self._get_adjacent_states(true_state)
            return self.np_random.choice(adjacent) if adjacent else true_state
        return true_state


class MultiObjectiveGridWorld(gym.Env):
    # ... (content of MultiObjectiveGridWorld, unchanged) ...
    metadata = {"render_modes": ["ansi"]}
    
    def __init__(self, size: int = 6, unsafe_cells=(10, 15, 20), 
                 reward_cells=(8, 14), penalty_cells=(12, 18)):
        super().__init__()
        self.size = size
        self.nS = size * size
        self.nA = 4
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)
        
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
        
        r = 0.0
        if s_next == self.goal:
            r += 10.0
        if self.reward_mask[s_next]:
            r += 1.0
        if self.penalty_mask[s_next]:
            r -= 2.0
        
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
    GridWorld where hazards change over time.
    MODIFIED for Alternating Hostile States.
    """
    metadata = {"render_modes": ["ansi"]}
    
    def __init__(self, size: int = 5, change_frequency: int = 100,
                 hazard_mode: str = "alternating_hostile"): # New parameter
        super().__init__()
        self.size = size
        self.nS = size * size
        self.nA = 4
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)
        
        self.change_frequency = change_frequency # This is agent steps, not outer loop iterations
        self.total_agent_steps = 0 # Track total agent steps for hazard changes
        self.goal = self.nS - 1 if self.nS > 0 else 0
        
        self.hazard_mode = hazard_mode
        self.current_hazard_config_index = 0 # For alternating mode

        # Define hostile hazard configurations for a 5x5 grid
        # These should ideally conflict with the expert path (R,R,R,R, D,D,D,D)
        # Expert path states for 5x5: 0,1,2,3,4, 9,14,19,24 (goal)
        if self.size == 5:
            self.hostile_config_1 = [6, 11]  # (1,1), (2,1) - near start, forces deviation
            self.hostile_config_2 = [18, 23] # (3,3), (4,3) - near goal, forces late deviation
            # A config directly on the path
            self.hostile_config_3 = [2, 9]   # (0,2) and (1,4) - directly on expert path
        else: # Generic fallback if size is not 5
            self.hostile_config_1 = [self.size + 1, self.size + 2]
            self.hostile_config_2 = [self.nS - self.size - 2, self.nS - self.size - 3]
            self.hostile_config_3 = [self.size//2, self.nS - self.size//2 -1]


        self.unsafe_mask = np.zeros(self.nS, dtype=bool) # Initialized by _update_hazards
        self._update_hazards() # Initial hazard setup
        
    def _update_hazards(self):
        """Update hazard locations based on hazard_mode."""
        self.unsafe_mask.fill(False) # Clear previous hazards

        if self.hazard_mode == "alternating_hostile":
            if self.current_hazard_config_index == 0:
                active_config = self.hostile_config_1
            elif self.current_hazard_config_index == 1:
                active_config = self.hostile_config_2
            else: # self.current_hazard_config_index == 2
                active_config = self.hostile_config_3
            
            valid_cells = [c for c in active_config if 0 <= c < self.nS]
            if valid_cells:
                self.unsafe_mask[valid_cells] = True
            self.current_hazard_config_index = (self.current_hazard_config_index + 1) % 3 # Cycle through 3 configs
            # print(f"Agent step {self.total_agent_steps}: Switched to hazard config {self.current_hazard_config_index}, unsafe: {np.where(self.unsafe_mask)[0]}")

        elif self.hazard_mode == "random": # Original random behavior
            n_hazards = self.np_random.integers(2, 5)
            candidate_locations = [s for s in range(self.nS) if s != 0 and s != self.goal]
            if candidate_locations and len(candidate_locations) >= n_hazards:
                 hazard_locations = self.np_random.choice(
                    candidate_locations,
                    size=n_hazards, 
                    replace=False
                )
                 self.unsafe_mask[hazard_locations] = True
        else:
            raise ValueError(f"Unknown hazard_mode: {self.hazard_mode}")
            
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.s = 0
        # self.total_agent_steps = 0 # Resetting agent steps on env.reset() might be too frequent if outer loop calls reset.
                                 # Let total_agent_steps be persistent across episodes within one run_dynamic_env_experiment call.
                                 # It's reset in run_dynamic_env_experiment itself before the main loop.
        return self.s, {}
    
    def step(self, a):
        self.total_agent_steps += 1
        
        if self.total_agent_steps > 0 and self.total_agent_steps % self.change_frequency == 0:
            self._update_hazards()
        
        s = self.s
        s_next = self._next_state(s, a)
        
        r, terminated, truncated = 0.0, False, False
        if s_next == self.goal:
            terminated, r = True, 1.0
        # No penalty for hitting unsafe state in reward, handled by constraint
        
        self.s = s_next
        return s_next, r, terminated, truncated, {}
    
    def _next_state(self, s, a):
        row, col = divmod(s, self.size)
        if   a == 0 and row > 0:            row -= 1
        elif a == 1 and row < self.size-1:  row += 1
        elif a == 2 and col > 0:            col -= 1
        elif a == 3 and col < self.size-1:  col += 1
        return row * self.size + col