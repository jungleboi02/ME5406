# main.py

from env import FrozenLakeEnv
from mc_control import monte_carlo_control
from plotting import plot_learning_curves
from sarsa import sarsa
from q_learning import q_learning
from utilities import print_policy, moving_average

def main():
    env = FrozenLakeEnv()

    print("Training Monte Carlo Control...")
    Q_mc, mc_rewards = monte_carlo_control(env)

    print("Training SARSA...")
    Q_sarsa, sarsa_rewards = sarsa(env)

    print("Training Q-learning...")
    Q_ql, ql_rewards = q_learning(env)

    print("Optimal policy from Monte Carlo Control:")
    print_policy(Q_mc, env)

    print("Optimal policy from SARSA:")
    print_policy(Q_sarsa, env)

    print("Optimal policy from Q-learning:")
    print_policy(Q_ql, env)

    print("Training completed.")

    plot_learning_curves(mc_rewards, sarsa_rewards, ql_rewards)

if __name__ == "__main__":
    main()

