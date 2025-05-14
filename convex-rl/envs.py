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
