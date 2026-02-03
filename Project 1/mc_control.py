# mc_control.py

from collections import defaultdict
import random
from config import *
from utilities import epsilon_greedy

def monte_carlo_control(env):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)

    episode_rewards = []
    episode_steps = []
    episode_success = []

    for episode in range(NUM_EPISODES):
        episode_data = []
        state = env.reset()
        total_reward = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            action = epsilon_greedy(Q, state, EPSILON)
            next_state, reward, done = env.step(action)

            episode_data.append((state, action, reward))
            total_reward += reward
            state = next_state

            if done:
                episode_rewards.append(total_reward)
                episode_steps.append(step + 1)
                episode_success.append(1 if reward == 1 else 0)
                break
        else:
            # No break happened
            episode_rewards.append(total_reward)
            episode_steps.append(MAX_STEPS_PER_EPISODE)
            episode_success.append(0)
            
        G = 0
        visited = set()

        for state, action, reward in reversed(episode_data):
            G = DISCOUNT * G + reward
            if (state, action) not in visited:
                visited.add((state, action))
                returns_sum[(state, action)] += G
                returns_count[(state, action)] += 1
                Q[state][action] = returns_sum[(state, action)] / returns_count[(state, action)]

    metrics = {
        "rewards": episode_rewards,
        "steps": episode_steps,
        "success": episode_success
    }

    return Q, metrics
