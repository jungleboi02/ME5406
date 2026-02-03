# config.py
# Configuration parameters for the Frozen Lake environment and RL algorithms

# Frozen Lake grid dimensions
GRID_ROWS = 10
GRID_COLS = 10

# Start and goal states
START_STATE = (0, 0)
GOAL_STATE = (9, 9)

# Holes for the Frozen Lake environment
# For 3x3 grid
# HOLES = [(1, 1), (1, 3), (2, 3), (3, 0)]
# For 10x10 grid
HOLES = [(1,1), (1,3), (2,3), (3,0), (0,7), (1,5), (2,7), (2,6), (3,3), (3,6), 
          (4,2), (4,3), (4,8), (5,8), (5,0), (6,2), (6,1), (7,4), (7,0), (7,9), 
          (8,6), (8,2), (9,3), (9,0), (9,2)]         

# Possible actions
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ACTION_TO_DELTA = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1)
}

# Reinforcement Learning parameters
DISCOUNT = 0.99
ALPHA = 0.1
EPSILON = 0.05
NUM_EPISODES = 10000
MAX_STEPS_PER_EPISODE = 250

