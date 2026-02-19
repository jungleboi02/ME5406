# main.py

import time
from config import GRID_ROWS
from env import FrozenLakeEnv
from mc_control import monte_carlo_control
from sarsa import sarsa
from q_learning import q_learning
from visualisation import (
    print_policy,
    plot_single_algorithm,
    plot_comparison,
    plot_policy

)

def main():
    env = FrozenLakeEnv()

    print("Training Monte Carlo Control...")
    start = time.time()
    Q_mc, mc_metrics = monte_carlo_control(env)
    mc_time = time.time() - start

    print("Training SARSA...")
    start = time.time()
    Q_sarsa, sarsa_metrics = sarsa(env)
    sarsa_time = time.time() - start

    print("Training Q-learning...")
    start = time.time()
    Q_ql, ql_metrics = q_learning(env)
    ql_time = time.time() - start

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

    plot_single_algorithm(mc_metrics, "Monte Carlo")
    plot_policy(Q_mc, env, env.rows, "Monte Carlo")

    plot_single_algorithm(sarsa_metrics, "SARSA")
    plot_policy(Q_sarsa, env, env.rows, "SARSA")

    plot_single_algorithm(ql_metrics, "Q-Learning")
    plot_policy(Q_ql, env, env.rows, "Q-Learning")

    plot_comparison(metrics_dict)


if __name__ == "__main__":
    main()

