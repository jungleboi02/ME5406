# visualization.py

import matplotlib.pyplot as plt
import numpy as np
import config

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

    axs[1, 1].bar(["Success", "Failure"], [total_success, total_fail])
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

    axs[1, 1].bar(x - width/2, success_counts, width, label="Success")
    axs[1, 1].bar(x + width/2, fail_counts, width, label="Failure")
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(labels)
    axs[1, 1].set_title("Success vs Failure Comparison")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()

def plot_policy(Q, grid_size, algo_name):

    import numpy as np
    import matplotlib.pyplot as plt

    policy_grid = np.empty((grid_size, grid_size), dtype=str)

    arrow_map = {
        0: "←",
        1: "↓",
        2: "→",
        3: "↑"
    }

    for state in Q.keys():

        if isinstance(state, tuple):
            row, col = state
        else:
            row = state // grid_size
            col = state % grid_size

        best_action = max(Q[state], key=Q[state].get)
        policy_grid[row, col] = arrow_map.get(best_action, "?")

    plt.figure(figsize=(6, 6))
    plt.title(f"{algo_name} Final Policy")

    plt.imshow(np.zeros((grid_size, grid_size)), cmap="gray")

    for i in range(grid_size):
        for j in range(grid_size):
            if policy_grid[i, j] != "":
                plt.text(j, i, policy_grid[i, j],
                         ha="center", va="center", fontsize=18)

    plt.xticks(range(grid_size))
    plt.yticks(range(grid_size))
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.show()
