"""
misc.py

Utility functions for:
- Action selection (epsilon-greedy)
- Printing learned policies
- Plotting learning curves
- Comparing algorithms
- Visualizing learned policy paths

This file does NOT contain learning logic.
It only supports training and visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import random

# Import required configuration variables
from config import ACTIONS, ACTION_TO_DELTA, GRID_ROWS, GRID_COLS, HOLES, GOAL_STATE


# ============================================================
# Arrow symbols used to visually display policies
# ============================================================

arrow_map = {
    'LEFT': '←',
    'DOWN': '↓',
    'RIGHT': '→',
    'UP': '↑'
}


# ============================================================
# 1. Epsilon-Greedy Action Selection
# ============================================================

def epsilon_greedy(Q, state, epsilon):
    """
    Select an action using textbook epsilon-greedy policy.

    With probability (1 - epsilon):
        Choose the greedy (best) action.

    With probability epsilon:
        Choose randomly among all actions.

    Implementation detail:
        Greedy action probability = 1 - epsilon + epsilon/|A|
        Other actions probability = epsilon/|A|

    Inputs:
        Q       → Q-table dictionary
        state   → integer state index
        epsilon → exploration probability

    Returns:
        action (string)
    """

    # Retrieve dictionary of action-values for this state
    # Example: Q[state] = {'UP': 0.3, 'DOWN': -0.1, ...}
    q_values = Q[state]

    # Find maximum Q-value among all actions
    max_q = max(q_values.values())

    # There may be ties for maximum Q-value.
    # Collect all actions that share the maximum value.
    greedy_actions = [
        action for action, value in q_values.items()
        if value == max_q
    ]

    # If multiple greedy actions exist, choose randomly among them
    greedy_action = random.choice(greedy_actions)

    num_actions = len(ACTIONS)

    # Build probability distribution over actions
    probs = []

    for action in ACTIONS:

        if action == greedy_action:
            # Greedy action gets majority probability mass
            probs.append(1 - epsilon + epsilon / num_actions)
        else:
            # All other actions share exploration probability
            probs.append(epsilon / num_actions)

    # Randomly sample according to probability weights
    return random.choices(ACTIONS, weights=probs, k=1)[0]


# ============================================================
# 2. Print Policy in Grid Format
# ============================================================

def print_policy(Q, env):
    """
    Print optimal policy derived from Q-table.

    For each grid cell:
        - G for goal
        - H for hole
        - Arrow for best action
    """

    for r in range(GRID_ROWS):

        row = []

        for c in range(GRID_COLS):

            state = (r, c)

            # If goal state
            if state == GOAL_STATE:
                row.append(' G ')

            # If hole
            elif state in HOLES:
                row.append(' H ')

            else:
                # Convert (r,c) to state index
                s_idx = env.state_to_index(state)

                # Select action with highest Q-value
                best_action = max(Q[s_idx], key=Q[s_idx].get)

                row.append(f' {arrow_map[best_action]} ')

        # Print full row as continuous string
        print(''.join(row))

    print()  # blank line after grid


# ============================================================
# 3. Moving Average (Smoothing)
# ============================================================

def moving_average(data, window=500):
    """
    Smooth noisy learning curves using convolution.

    window = number of episodes to average over.

    This helps visualize overall trend instead of noisy spikes.
    """
    return np.convolve(data, np.ones(window)/window, mode='valid')


# ============================================================
# 4. Plot Single Algorithm Performance
# ============================================================

def plot_single_algorithm(metrics, algo_name):
    """
    Plot:
    - Average reward
    - Average steps
    - Success rate
    - Success vs failure bar chart

    metrics is a dictionary:
        {
            "rewards": [...],
            "steps": [...],
            "success": [...]
        }
    """

    rewards = metrics["rewards"]
    steps = metrics["steps"]
    success = metrics["success"]

    # Compute cumulative success rate:
    # accuracy[i] = (# successes up to episode i) / i
    acc = np.cumsum(success) / np.arange(1, len(success) + 1)

    # Create 2x2 subplot layout
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{algo_name} Performance")

    # ------------------------------------------
    # Reward Curve
    # ------------------------------------------
    axs[0, 0].plot(moving_average(rewards))
    axs[0, 0].set_title("Average Reward")

    # ------------------------------------------
    # Steps Curve
    # ------------------------------------------
    axs[0, 1].plot(moving_average(steps))
    axs[0, 1].set_title("Average Steps")

    # ------------------------------------------
    # Accuracy Curve
    # ------------------------------------------
    axs[1, 0].plot(acc)
    axs[1, 0].set_title("Accuracy (Success Rate)")

    # ------------------------------------------
    # Success vs Failure Bar Chart
    # ------------------------------------------

    total_success = sum(success)
    total_fail = len(success) - total_success

    axs[1, 1].bar(
        ["Success", "Failure"],
        [total_success, total_fail],
        color=['tab:blue', 'red'],
        edgecolor='black'
    )

    axs[1, 1].set_title("Success vs Failure")

    plt.tight_layout()
    plt.show()


# ============================================================
# 5. Plot Comparison Between Algorithms
# ============================================================

def plot_comparison(metrics_dict):
    """
    Compare multiple algorithms on same figure.

    metrics_dict example:
    {
        "Monte Carlo": mc_metrics,
        "SARSA": sarsa_metrics,
        "Q-Learning": ql_metrics
    }
    """

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Algorithm Comparison")

    # ------------------------------------------
    # Plot curves for each algorithm
    # ------------------------------------------

    for name, metrics in metrics_dict.items():

        rewards = metrics["rewards"]
        steps = metrics["steps"]
        success = metrics["success"]

        acc = np.cumsum(success) / np.arange(1, len(success) + 1)

        axs[0, 0].plot(moving_average(rewards), label=name)
        axs[0, 1].plot(moving_average(steps), label=name)
        axs[1, 0].plot(acc, label=name)

    axs[0, 0].set_title("Average Reward")
    axs[0, 1].set_title("Average Steps")
    axs[1, 0].set_title("Accuracy")

    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[1, 0].legend()

    # ------------------------------------------
    # Success Comparison Bars
    # ------------------------------------------

    labels = list(metrics_dict.keys())

    success_counts = []
    fail_counts = []

    for metrics in metrics_dict.values():
        s = sum(metrics["success"])
        f = len(metrics["success"]) - s
        success_counts.append(s)
        fail_counts.append(f)

    x = np.arange(len(labels))
    width = 0.35

    axs[1, 1].bar(
        x - width/2, success_counts, width,
        label="Success", color='tab:blue', edgecolor='black'
    )

    axs[1, 1].bar(
        x + width/2, fail_counts, width,
        label="Failure", color='red', edgecolor='black'
    )

    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(labels)
    axs[1, 1].set_title("Success vs Failure Comparison")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# 6. Plot Policy Path Visualization
# ============================================================

def plot_policy_path(Q, env, grid_size, algo_name):
    """
    Visualize:
    - Grid layout
    - Greedy policy arrows
    - Path followed from start to goal
    """

    plt.figure(figsize=(6, 6))
    plt.title(f"{algo_name} Policy & Path")

    # --------------------------------------------------
    # Follow greedy policy starting from start state
    # --------------------------------------------------

    state = env.start_state
    path = [state]   # store visited states
    visited = set()  # detect loops

    while state != env.goal_state:

        s_idx = env.state_to_index(state)

        # If state not learned yet → stop
        if s_idx not in Q:
            break

        # If loop detected → stop
        if state in visited:
            break

        visited.add(state)

        # Choose greedy action
        best_action = max(Q[s_idx], key=Q[s_idx].get)

        # Convert action into movement
        dr, dc = ACTION_TO_DELTA[best_action]
        next_state = (state[0] + dr, state[1] + dc)

        # Stop if invalid move or hole
        if (next_state[0] < 0 or next_state[0] >= grid_size or
            next_state[1] < 0 or next_state[1] >= grid_size or
            next_state in env.holes):
            break

        path.append(next_state)
        state = next_state

    # --------------------------------------------------
    # Draw Grid
    # --------------------------------------------------

    for r in range(grid_size):
        for c in range(grid_size):

            tile = env.grid[r][c]
            color = 'white'

            if tile == 'H':
                color = 'black'
            elif tile == 'G':
                color = 'green'
            elif tile == 'S':
                color = 'blue'

            # Highlight path
            if (r, c) in path and (r, c) not in [env.start_state, env.goal_state]:
                color = '#fff5b1'

            plt.gca().add_patch(
                plt.Rectangle((c, r), 1, 1, color=color, ec='gray')
            )

    # --------------------------------------------------
    # Draw Policy Arrows
    # --------------------------------------------------

    for r in range(grid_size):
        for c in range(grid_size):

            s_idx = env.state_to_index((r, c))

            if (s_idx in Q and
                (r, c) not in env.holes and
                (r, c) != env.goal_state):

                best_action = max(Q[s_idx], key=Q[s_idx].get)

                plt.text(
                    c + 0.5, r + 0.5,
                    arrow_map[best_action],
                    ha='center', va='center',
                    fontsize=16, color='red'
                )

    # Formatting
    plt.xlim(0, grid_size)
    plt.ylim(0, grid_size)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')
    plt.xticks(range(grid_size + 1))
    plt.yticks(range(grid_size + 1))
    plt.grid(which='major')
    plt.show()