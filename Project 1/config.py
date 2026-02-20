"""
config.py

Central configuration file for:
- Environment structure
- Action definitions
- Reinforcement learning hyperparameters

Changing values here modifies the entire experiment.
"""

# --------------------------------------------------
# GRID CONFIGURATION
# --------------------------------------------------

# Size of the Frozen Lake grid
GRID_ROWS = 10
GRID_COLS = 10

# Start position (row, col)
START_STATE = (0, 0)

# Goal position (row, col)
GOAL_STATE = (9, 9)

# Hole locations (terminal states with -1 reward)
HOLES = [
    (1,1), (1,3), (2,3), (3,0), (0,7), (1,5), (2,7),
    (2,6), (3,3), (3,6), (4,2), (4,3), (4,8), (5,8),
    (5,0), (6,2), (6,1), (7,4), (7,0), (7,9),
    (8,6), (8,2), (9,3), (9,0), (9,2)
]

# --------------------------------------------------
# ACTION DEFINITIONS
# --------------------------------------------------

# Action space
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']

# Maps each action to how it changes (row, col)
ACTION_TO_DELTA = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1)
}

# --------------------------------------------------
# RL HYPERPARAMETERS
# --------------------------------------------------

# Discount factor (γ)
# Controls importance of future rewards
DISCOUNT = 0.99

# Learning rate (α)
# Controls how fast Q-values update
ALPHA = 0.05

# Exploration rate (ε)
# Probability of choosing random action
EPSILON = 0.01

# Total number of training episodes
NUM_EPISODES = 250000

# Maximum steps allowed in one episode
MAX_STEPS_PER_EPISODE = 1000