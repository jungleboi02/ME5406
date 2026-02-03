# env.py

import random
from config import *

class FrozenLakeEnv:
    def __init__(self):
        self.rows = GRID_ROWS
        self.cols = GRID_COLS
        self.start = START_STATE
        self.goal = GOAL_STATE
        self.holes = set(HOLES)

        self.state = None

        # Total number of states
        self.n_states = self.rows * self.cols
        self.n_actions = len(ACTIONS)

    def reset(self):
        """Reset environment to starting state."""
        self.state = self.start
        return self.state_to_index(self.state)

    def step(self, action):
        """
        Take an action and return:
        next_state_index, reward, done
        """
        r, c = self.state
        dr, dc = ACTION_TO_DELTA[action]

        # Confine robot within grid
        new_r = min(max(r + dr, 0), self.rows - 1)
        new_c = min(max(c + dc, 0), self.cols - 1)
        self.state = (new_r, new_c)

        if self.state == self.goal:
            return self.state_to_index(self.state), 1, True

        if self.state in self.holes:
            return self.state_to_index(self.state), -1, True

        return self.state_to_index(self.state), 0, False

    def state_to_index(self, state):
        """Map (row, col) to integer state index."""
        r, c = state
        return r * self.cols + c

    def index_to_state(self, index):
        """Map integer state index to (row, col)."""
        r = index // self.cols
        c = index % self.cols
        return (r, c)
