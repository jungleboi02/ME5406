# misc.py

import matplotlib.pyplot as plt
import numpy as np
from config import ACTIONS, ACTION_TO_DELTA, GRID_ROWS, GRID_COLS, HOLES, GOAL_STATE
import random

arrow_map = {
    'LEFT': '←',
    'DOWN': '↓',
    'RIGHT': '→',
    'UP': '↑'
}

# 1. Epsilon-greedy action selection policy
def epsilon_greedy(Q, state, epsilon):
    """
    Select an action using the textbook epsilon-greedy policy:
    - Greedy action gets probability: 1 - epsilon + epsilon/|A|
    - All other actions get probability: epsilon/|A|
    """
    q_values = Q[state]
    max_q = max(q_values.values())

    # Identify greedy actions (handle ties)
    greedy_actions = [a for a, q in q_values.items() if q == max_q]

    # Choose one greedy action uniformly if ties exist
    greedy_action = random.choice(greedy_actions)

    num_actions = len(ACTIONS)
    probs = []
    
    # Build probability distribution
    for action in ACTIONS:
        if action == greedy_action:
            probs.append(1 - epsilon + epsilon / num_actions)
        else:
            probs.append(epsilon / num_actions)

    # Sample according to probability distribution
    return random.choices(ACTIONS, weights=probs, k=1)[0]

# 2. Print policy in grid format
def print_policy(Q, env):
    """
    Print the optimal policy derived from Q in grid form.
    """
    for r in range(GRID_ROWS):
        row = []
        for c in range(GRID_COLS):
            state = (r, c)

            if state == GOAL_STATE:
                row.append(' G ')
            elif state in HOLES:
                row.append(' H ')
            else:
                s_idx = env.state_to_index(state)
                best_action = max(Q[s_idx], key=Q[s_idx].get)
                row.append(f' {arrow_map[best_action]} ')
        print(''.join(row))
    print()

def moving_average(data, window=500):
    return np.convolve(data, np.ones(window)/window, mode='valid')

def plot_single_algorithm(metrics, algo_name):

    rewards = metrics["rewards"]
    steps = metrics["steps"]
    success = metrics["success"]

    acc = np.cumsum(success) / np.arange(1, len(success)+1)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{algo_name} Performance")

    # Reward curve
    axs[0, 0].plot(moving_average(rewards))
    axs[0, 0].set_title("Average Reward")

    # Steps curve
    axs[0, 1].plot(moving_average(steps))
    axs[0, 1].set_title("Average Steps")

    # Accuracy curve
    axs[1, 0].plot(acc)
    axs[1, 0].set_title("Accuracy (Success Rate)")

    # Success vs Failure bar
    total_success = sum(success)
    total_fail = len(success) - total_success

    axs[1, 1].bar(
        ["Success", "Failure"], 
        [total_success, total_fail],
        color=['tab:blue', 'red'],  # default matplotlib blue for success
        edgecolor='black'
    )

    axs[1, 1].set_title("Success vs Failure")


    plt.tight_layout()
    plt.show()

def plot_comparison(metrics_dict):
    """
    metrics_dict = {
        "MC": mc_metrics,
        "SARSA": sarsa_metrics,
        "Q-Learning": ql_metrics
    }
    """

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Algorithm Comparison")

    for name, metrics in metrics_dict.items():
        rewards = metrics["rewards"]
        steps = metrics["steps"]
        success = metrics["success"]

        acc = np.cumsum(success) / np.arange(1, len(success)+1)

        axs[0, 0].plot(moving_average(rewards), label=name)
        axs[0, 1].plot(moving_average(steps), label=name)
        axs[1, 0].plot(acc, label=name)

    axs[0, 0].set_title("Average Reward")
    axs[0, 1].set_title("Average Steps")
    axs[1, 0].set_title("Accuracy")

    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[1, 0].legend()

    # Success comparison bar chart
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

def plot_policy_path(Q, env, grid_size, algo_name):
    import matplotlib.pyplot as plt
    arrow_map = {'LEFT':'←','RIGHT':'→','UP':'↑','DOWN':'↓'}

    plt.figure(figsize=(6,6))
    plt.title(f"{algo_name} Policy & Path")

    # Follow greedy policy to get path
    state = env.start_state
    path = [state]
    visited = set()

    while state != env.goal_state:
        s_idx = env.state_to_index(state)

        if s_idx not in Q:
            break
        if state in visited:
            break
        visited.add(state)

        best_action = max(Q[s_idx], key=Q[s_idx].get)
        dr, dc = ACTION_TO_DELTA[best_action]
        next_state = (state[0]+dr, state[1]+dc)

        if (next_state[0] < 0 or next_state[0] >= grid_size or
            next_state[1] < 0 or next_state[1] >= grid_size or
            next_state in env.holes):
            break

        path.append(next_state)
        state = next_state

    # Draw tiles with path highlighted
    for r in range(grid_size):
        for c in range(grid_size):
            tile = env.grid[r][c]
            color = 'white'
            if tile == 'H': color='black'
            elif tile=='G': color='green'
            elif tile=='S': color='blue'
            if (r, c) in path and (r, c) != env.start_state and (r, c) != env.goal_state:
                color = '#fff5b1'  # light yellow
            plt.gca().add_patch(plt.Rectangle((c,r),1,1,color=color,ec='gray'))

    # Draw policy arrows
    for r in range(grid_size):
        for c in range(grid_size):
            s_idx = env.state_to_index((r,c))
            if s_idx in Q and (r,c) not in env.holes and (r,c) != env.goal_state:
                best_action = max(Q[s_idx], key=Q[s_idx].get)
                plt.text(c+0.5, r+0.5, arrow_map[best_action],
                         ha='center', va='center', fontsize=16, color='red')

    plt.xlim(0, grid_size)
    plt.ylim(0, grid_size)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')
    plt.grid(True)

    # Show only major grid lines (integers)
    plt.xticks(range(grid_size + 1))
    plt.yticks(range(grid_size + 1))
    plt.grid(which='major')  # only major ticks
    plt.show()
