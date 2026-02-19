# sarsa.py

from collections import defaultdict
from config import *
from misc import epsilon_greedy

def sarsa(env):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    episode_rewards = []
    episode_steps = []
    episode_success = []

    for episode in range(NUM_EPISODES):

        state = env.reset()
        epsilon = max(EPSILON, 1.0 - episode / 50000)
        action = epsilon_greedy(Q, state, epsilon)

        total_reward = 0

        for step in range(MAX_STEPS_PER_EPISODE):

            next_state, reward, done = env.step(action)
            total_reward += reward

            if done:
                # Terminal update (NO BOOTSTRAP)
                Q[state][action] += ALPHA * (
                    reward - Q[state][action]
                )

                episode_rewards.append(total_reward)
                episode_steps.append(step + 1)
                episode_success.append(1 if reward == 1 else 0)
                break

            next_action = epsilon_greedy(Q, next_state, epsilon)

            Q[state][action] += ALPHA * (
                reward + DISCOUNT * Q[next_state][next_action]
                - Q[state][action]
            )

            state = next_state
            action = next_action

        else:
            # Episode did not terminate
            episode_rewards.append(total_reward)
            episode_steps.append(MAX_STEPS_PER_EPISODE)
            episode_success.append(0)

    metrics = {
        "rewards": episode_rewards,
        "steps": episode_steps,
        "success": episode_success
    }

    return Q, metrics
