"""
main.py

Main driver script for the Frozen Lake Reinforcement Learning project.

This file:
1. Creates the environment
2. Trains Monte Carlo, SARSA, and Q-learning
3. Measures training time
4. Prints learned policies
5. Plots performance metrics
6. Plots learned policy paths

This is the entry point of the entire project.
"""

import time
from env import FrozenLakeEnv
from mc_control import monte_carlo_control
from sarsa import sarsa
from q_learning import q_learning
from misc import (
    plot_policy_path,
    print_policy,
    plot_single_algorithm,
    plot_comparison,
)

def main():

    # -------------------------------------------------
    # Create environment
    # -------------------------------------------------
    # FrozenLakeEnv contains the grid, reward logic,
    # state transitions, and reset/step functions.
    env = FrozenLakeEnv()

    # Dictionary to store learned Q-tables from each algorithm
    # This allows us to later plot policies for each method.
    q_tables = {}

    # =================================================
    # Train Monte Carlo
    # =================================================
    print("Training Monte Carlo Control...")
    start = time.time()  # record start time

    # Train and receive:
    #   Q_mc        → learned Q-table
    #   mc_metrics  → reward, steps, success history
    Q_mc, mc_metrics = monte_carlo_control(env)

    mc_time = time.time() - start  # compute training time
    q_tables["Monte Carlo"] = Q_mc  # store learned Q-table

    # =================================================
    # Train SARSA
    # =================================================
    print("Training SARSA...")
    start = time.time()

    Q_sarsa, sarsa_metrics = sarsa(env)

    sarsa_time = time.time() - start
    q_tables["SARSA"] = Q_sarsa

    # =================================================
    # Train Q-learning
    # =================================================
    print("Training Q-learning...")
    start = time.time()

    Q_ql, ql_metrics = q_learning(env)

    ql_time = time.time() - start
    q_tables["Q-Learning"] = Q_ql

    # =================================================
    # Print learned optimal policies
    # =================================================
    print("Optimal policy from Monte Carlo Control:")
    print_policy(Q_mc, env)

    print("Optimal policy from SARSA:")
    print_policy(Q_sarsa, env)

    print("Optimal policy from Q-learning:")
    print_policy(Q_ql, env)

    # =================================================
    # Print training time comparison
    # =================================================
    print("\nTraining Time Comparison:")
    print(f"Monte Carlo time: {mc_time:.3f}s")
    print(f"SARSA time: {sarsa_time:.3f}s")
    print(f"Q-learning time: {ql_time:.3f}s")

    # Organize metrics into dictionary for plotting
    metrics_dict = {
        "Monte Carlo": mc_metrics,
        "SARSA": sarsa_metrics,
        "Q-Learning": ql_metrics
    }

    # =================================================
    # Plot results for each algorithm individually
    # =================================================
    for name in metrics_dict:

        # Plot learning curves (reward, steps, success rate)
        plot_single_algorithm(metrics_dict[name], name)

        # Plot final policy and greedy path from start to goal
        plot_policy_path(q_tables[name], env, env.rows, name)

    # =================================================
    # Plot comparison across all algorithms
    # =================================================
    plot_comparison(metrics_dict)


# Only execute main() if this file is run directly
if __name__ == "__main__":
    main()