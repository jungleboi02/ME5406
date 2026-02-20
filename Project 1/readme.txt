Frozen Lake Reinforcement Learning Project
==========================================

Author: (Your Name)
Course: ME5406 – Deep Learning for Robotics
Description:
This project implements and compares three reinforcement learning (RL) control algorithms
on a custom Frozen Lake grid environment:

    1. Monte Carlo Control
    2. SARSA (on-policy TD control)
    3. Q-Learning (off-policy TD control)

The goal is for an agent to learn an optimal policy that navigates from a start state (S)
to a goal state (G) while avoiding holes (H).

------------------------------------------------------------
PROJECT STRUCTURE
------------------------------------------------------------

main.py
    Entry point of the project.
    - Trains all three algorithms
    - Prints learned policies
    - Compares training time
    - Plots performance metrics
    - Plots final policy paths

config.py
    Central configuration file.
    Contains:
    - Grid dimensions
    - Start/goal positions
    - Hole locations
    - Action definitions
    - RL hyperparameters (alpha, gamma, epsilon, episodes)

env.py
    Defines the FrozenLakeEnv class.
    Implements:
    - reset()
    - step(action)
    - state indexing functions
    - Grid generation

misc.py
    Utility functions:
    - epsilon-greedy policy
    - policy printing
    - plotting performance
    - plotting comparison
    - plotting learned policy path

mc_control.py
    Implements Monte Carlo Control using incremental first-visit updates.

sarsa.py
    Implements SARSA (on-policy Temporal Difference learning).

q_learning.py
    Implements Q-learning (off-policy TD learning).

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------

1. Install required packages:
       pip install matplotlib numpy

2. Run the project:
       python main.py

3. The program will:
   - Train all three algorithms
   - Print their optimal policies
   - Display performance plots
   - Display learned policy paths

------------------------------------------------------------
ENVIRONMENT DETAILS
------------------------------------------------------------

Grid:
    Default: 10x10 grid (configurable in config.py)

States:
    Each grid cell is a state.
    States are internally converted to integer indices.

Rewards:
    +1  → Reaching the goal
    -1  → Falling into a hole
     0  → All other moves

Episode ends when:
    - Agent reaches goal
    - Agent falls into hole
    - Max steps reached

------------------------------------------------------------
ALGORITHM SUMMARY
------------------------------------------------------------

Monte Carlo Control
    - Learns from complete episodes
    - First-visit incremental updates
    - No bootstrapping

SARSA
    - On-policy TD method
    - Updates using the next action actually taken

Q-Learning
    - Off-policy TD method
    - Updates using the greedy next action

------------------------------------------------------------
METRICS TRACKED
------------------------------------------------------------

For each episode:
    - Total reward
    - Number of steps
    - Success (1 if goal reached, else 0)

Plots include:
    - Moving average reward
    - Moving average steps
    - Accuracy (success rate)
    - Success vs failure bars
    - Policy path visualization

------------------------------------------------------------
CUSTOMIZATION
------------------------------------------------------------

To modify the grid:
    Edit config.py:
        GRID_ROWS
        GRID_COLS
        HOLES
        START_STATE
        GOAL_STATE

To adjust learning:
    Modify:
        ALPHA
        DISCOUNT
        EPSILON
        NUM_EPISODES
        MAX_STEPS_PER_EPISODE

------------------------------------------------------------
NOTES
------------------------------------------------------------

- Q-tables are dictionaries:
      Q[state_index][action] = value

- State indices are mapped from (row, col):
      index = row * num_cols + col

- Policy arrows:
      ↑ UP
      ↓ DOWN
      ← LEFT
      → RIGHT

------------------------------------------------------------
END
------------------------------------------------------------