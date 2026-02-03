# mc_control.py

from collections import defaultdict
import random
from config import *
from utilities import epsilon_greedy

def monte_carlo_control(env):
    # Q(s,a) initialised to 0
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    episode_rewards = []
    
    for episode in range(NUM_EPISODES):
        episode_data = []
        state = env.reset()

        for _ in range(MAX_STEPS_PER_EPISODE):
            action = epsilon_greedy(Q, state, EPSILON)
            next_state, reward, done = env.step(action)

            episode_data.append((state, action, reward))
            state = next_state

            if done:
                episode_rewards.append(reward)
                break

        G = 0
        visited = set()

        # Traverse episode backward
        for state, action, reward in reversed(episode_data):
            G = DISCOUNT * G + reward

            if (state, action) not in visited:
                visited.add((state, action))
                returns_sum[(state, action)] += G
                returns_count[(state, action)] += 1
                Q[state][action] = returns_sum[(state, action)] / returns_count[(state, action)]

    return Q, episode_rewards
