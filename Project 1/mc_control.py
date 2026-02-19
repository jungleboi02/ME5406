from collections import defaultdict
import random
from config import *
from utilities import epsilon_greedy

def monte_carlo_control(env):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    episode_rewards = []
    episode_steps = []
    episode_success = []

    alpha = 0.01  # incremental MC step size

    for episode in range(NUM_EPISODES):
        # if episode % 1000 == 0:
        #     print("Episode:", episode)

        episode_data = []
        state = env.reset()
        total_reward = 0
        epsilon = max(0.05, 1.0 - episode / 50000)

        for step in range(MAX_STEPS_PER_EPISODE):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done = env.step(action)
            episode_data.append((state, action, reward))
            total_reward += reward
            state = next_state

            if done:
                break
        else:
            # Episode reached max steps
            pass

        # Incremental first-visit MC update
        G = 0
        visited = set()
        for state, action, reward in reversed(episode_data):
            G = DISCOUNT * G + reward
            if (state, action) not in visited:
                visited.add((state, action))
                Q[state][action] += alpha * (G - Q[state][action])

        episode_rewards.append(total_reward)
        episode_steps.append(len(episode_data))
        episode_success.append(1 if total_reward > 0 else 0)

    metrics = {
        "rewards": episode_rewards,
        "steps": episode_steps,
        "success": episode_success
    }

    return Q, metrics
