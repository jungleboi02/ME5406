# q_learning.py

from collections import defaultdict
from config import *
from utilities import epsilon_greedy

def q_learning(env):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    episode_rewards = []

    for episode in range(NUM_EPISODES):
        state = env.reset()

        for _ in range(MAX_STEPS_PER_EPISODE):
            action = epsilon_greedy(Q, state, EPSILON)
            next_state, reward, done = env.step(action)

            best_next_q = max(Q[next_state].values())

            # Q-learning update
            Q[state][action] += ALPHA * (
                reward + DISCOUNT * best_next_q - Q[state][action]
            )

            state = next_state

            if done:
                episode_rewards.append(reward)
                break

    return Q, episode_rewards
