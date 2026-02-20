# main.py

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
    env = FrozenLakeEnv()

    # Dictionary to store Q-tables
    q_tables = {}

    print("Training Monte Carlo Control...")
    start = time.time()
    Q_mc, mc_metrics = monte_carlo_control(env)
    mc_time = time.time() - start
    q_tables["Monte Carlo"] = Q_mc  # store Q-table

    print("Training SARSA...")
    start = time.time()
    Q_sarsa, sarsa_metrics = sarsa(env)
    sarsa_time = time.time() - start
    q_tables["SARSA"] = Q_sarsa  # store Q-table

    print("Training Q-learning...")
    start = time.time()
    Q_ql, ql_metrics = q_learning(env)
    ql_time = time.time() - start
    q_tables["Q-Learning"] = Q_ql  # store Q-table

    print("Optimal policy from Monte Carlo Control:")
    print_policy(Q_mc, env)

    print("Optimal policy from SARSA:")
    print_policy(Q_sarsa, env)

    print("Optimal policy from Q-learning:")
    print_policy(Q_ql, env)

    print("Training completed.")
    print("\nTraining Time Comparison:")
    print(f"Monte Carlo time: {mc_time:.3f}s")
    print(f"SARSA time: {sarsa_time:.3f}s")
    print(f"Q-learning time: {ql_time:.3f}s")

    metrics_dict = {
        "Monte Carlo": mc_metrics,
        "SARSA": sarsa_metrics,
        "Q-Learning": ql_metrics
    }

    # Plot single algorithm metrics and policy paths
    for name in ["Monte Carlo", "SARSA", "Q-Learning"]:
        plot_single_algorithm(metrics_dict[name], name)
        plot_policy_path(q_tables[name], env, env.rows, name)  # use stored Q-table

    # Compare all algorithms
    plot_comparison(metrics_dict)


if __name__ == "__main__":
    main()

