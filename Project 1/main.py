# main.py

from env import FrozenLakeEnv
from mc_control import monte_carlo_control
from plotting import plot_learning_curves
from sarsa import sarsa
from q_learning import q_learning
from utilities import print_policy, moving_average
import time

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
    print(f"Monte Carlo time: {mc_time:.2f}s")
    print(f"SARSA time: {sarsa_time:.2f}s")
    print(f"Q-learning time: {ql_time:.2f}s")

    plot_learning_curves(mc_metrics, sarsa_metrics, ql_metrics)
    print("\nTraining Time Comparison:")

    print(f"Monte Carlo: {mc_time:.3f} sec")
    print(f"SARSA: {sarsa_time:.3f} sec")
    print(f"Q-learning: {ql_time:.3f} sec")


if __name__ == "__main__":
    main()

