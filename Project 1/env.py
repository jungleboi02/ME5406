"""
env.py

Defines the FrozenLakeEnv class.

This environment:
- Creates a grid world
- Tracks agent position
- Returns rewards
- Determines episode termination
"""

from config import *

class FrozenLakeEnv:

    def __init__(self):
        """
        Initialize environment using config parameters.
        """

        self.rows = GRID_ROWS
        self.cols = GRID_COLS

        self.start_state = START_STATE
        self.goal_state = GOAL_STATE
        self.holes = set(HOLES)  # convert to set for fast lookup

        # Current agent state
        self.state = self.start_state

        # Generate visual grid (for plotting)
        self.grid = self._generate_grid()

    def _generate_grid(self):
        """
        Creates 2D list representing the grid.

        F = frozen tile
        S = start
        G = goal
        H = hole
        """

        # Initialize everything as Frozen tiles
        grid = [['F' for _ in range(self.cols)]
                for _ in range(self.rows)]

        # Set start tile
        r, c = self.start_state
        grid[r][c] = 'S'

        # Set goal tile
        r, c = self.goal_state
        grid[r][c] = 'G'

        # Set holes
        for r, c in self.holes:
            grid[r][c] = 'H'

        return grid

    def reset(self):
        """
        Reset environment to start state.
        Returns integer index of start state.
        """

        self.state = self.start_state
        return self.state_to_index(self.state)

    def step(self, action):
        """
        Perform one action in the environment.

        Returns:
            next_state_index
            reward
            done (True if terminal state)
        """

        # Current position
        r, c = self.state

        # Convert action into movement direction
        dr, dc = ACTION_TO_DELTA[action]

        # Compute new position
        new_r = min(max(r + dr, 0), self.rows - 1)
        new_c = min(max(c + dc, 0), self.cols - 1)

        self.state = (new_r, new_c)

        # -------------------------------
        # Check terminal conditions
        # -------------------------------

        # Goal reached
        if self.state == self.goal_state:
            return self.state_to_index(self.state), 1, True

        # Fell into hole
        if self.state in self.holes:
            return self.state_to_index(self.state), -1, True

        # Normal step
        return self.state_to_index(self.state), 0, False

    def state_to_index(self, state):
        """
        Convert (row, col) → integer index.
        """
        r, c = state
        return r * self.cols + c

    def index_to_state(self, index):
        """
        Convert integer index → (row, col).
        """
        r = index // self.cols
        c = index % self.cols
        return (r, c)