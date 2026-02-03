# plotting.py

import matplotlib.pyplot as plt

def moving_average_success(rewards, window=100):
    """
    Compute moving average success rate.
    Success is defined as reward == +1.
    """
    success_rate = []

    for i in range(len(rewards)):
        start = max(0, i - window)
        window_rewards = rewards[start:i]

        if len(window_rewards) == 0:
            success_rate.append(0)
        else:
            successes = sum(1 for r in window_rewards if r == 1)
            success_rate.append(successes / len(window_rewards))

    return success_rate


def plot_learning_curves(mc_rewards, sarsa_rewards, ql_rewards, window=100):
    """
    Plot learning curves for all three RL methods.
    """
    mc_curve = moving_average_success(mc_rewards, window)
    sarsa_curve = moving_average_success(sarsa_rewards, window)
    ql_curve = moving_average_success(ql_rewards, window)

    plt.figure(figsize=(8, 5))
    plt.plot(mc_curve, label='Monte Carlo')
    plt.plot(sarsa_curve, label='SARSA')
    plt.plot(ql_curve, label='Q-learning')

    plt.xlabel('Episode')
    plt.ylabel('Success Rate (Moving Average)')
    plt.title('Learning Progress Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
