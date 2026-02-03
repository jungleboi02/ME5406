# q_learning.py

from collections import defaultdict
from config import *
from utilities import epsilon_greedy

def q_learning(env):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    episode_rewards = []
    episode_steps = []
    episode_success = []

    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            action = epsilon_greedy(Q, state, EPSILON)
            next_state, reward, done = env.step(action)

            best_next_q = max(Q[next_state].values())

            Q[state][action] += ALPHA * (
                reward + DISCOUNT * best_next_q - Q[state][action]
            )

            total_reward += reward
            state = next_state

            if done:
                episode_rewards.append(total_reward)
                episode_steps.append(step + 1)
                episode_success.append(1 if reward == 1 else 0)
                break

    metrics = {
        "rewards": episode_rewards,
        "steps": episode_steps,
        "success": episode_success
    }

    return Q, metrics
