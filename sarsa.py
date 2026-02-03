# sarsa.py

from collections import defaultdict
from config import *
from utilities import epsilon_greedy

def sarsa(env):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    episode_rewards = []

    for episode in range(NUM_EPISODES):
        state = env.reset()
        action = epsilon_greedy(Q, state, EPSILON)

        for _ in range(MAX_STEPS_PER_EPISODE):
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy(Q, next_state, EPSILON)

            # SARSA update
            Q[state][action] += ALPHA * (
                reward + DISCOUNT * Q[next_state][next_action] - Q[state][action]
            )

            state = next_state
            action = next_action

            if done:
                episode_rewards.append(reward)
                break

    return Q, episode_rewards
