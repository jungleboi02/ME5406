# plotting.py

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np


def moving_average(data, window=100):
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_learning_curves(mc_metrics, sarsa_metrics, ql_metrics, window=100):

    mc_rewards = np.array(mc_metrics["rewards"])
    sarsa_rewards = np.array(sarsa_metrics["rewards"])
    ql_rewards = np.array(ql_metrics["rewards"])

    mc_success = np.array(mc_metrics["success"])
    sarsa_success = np.array(sarsa_metrics["success"])
    ql_success = np.array(ql_metrics["success"])

    mc_steps = np.array(mc_metrics["steps"])
    sarsa_steps = np.array(sarsa_metrics["steps"])
    ql_steps = np.array(ql_metrics["steps"])

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    # Success rate
    axs[0].plot(moving_average(mc_success, window), label="Monte Carlo")
    axs[0].plot(moving_average(sarsa_success, window), label="SARSA")
    axs[0].plot(moving_average(ql_success, window), label="Q-learning")
    axs[0].set_title("Success Rate")
    axs[0].legend()
    axs[0].grid(True)

    # Reward
    axs[1].plot(moving_average(mc_rewards, window))
    axs[1].plot(moving_average(sarsa_rewards, window))
    axs[1].plot(moving_average(ql_rewards, window))
    axs[1].set_title("Average Reward")
    axs[1].grid(True)

    # Steps
    axs[2].plot(mc_steps, alpha=0.5)
    axs[2].plot(sarsa_steps, alpha=0.5)
    axs[2].plot(ql_steps, alpha=0.5)
    axs[2].set_title("Steps per Episode")
    axs[2].grid(True)

    plt.suptitle("RL Algorithm Comparison")
    plt.tight_layout()
    plt.show()
