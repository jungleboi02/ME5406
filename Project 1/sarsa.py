"""
SARSA (On-policy TD Control)

Update rule:
Q(s,a) ← Q(s,a) + α [ r + γ Q(s',a') - Q(s,a) ]
"""

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

        # Select first action BEFORE loop
        action = epsilon_greedy(Q, state, EPSILON)

        total_reward = 0

        for step in range(MAX_STEPS_PER_EPISODE):

            # Take action
            next_state, reward, done = env.step(action)
            total_reward += reward

            if done:
                # Terminal update (no bootstrap)
                Q[state][action] += ALPHA * (reward - Q[state][action])
                episode_rewards.append(total_reward)
                episode_steps.append(step + 1)
                episode_success.append(1 if reward == 1 else 0)
                break

            # Choose next action (on-policy)
            next_action = epsilon_greedy(Q, next_state, EPSILON)

            # SARSA TD update
            Q[state][action] += ALPHA * (
                reward +
                DISCOUNT * Q[next_state][next_action] -
                Q[state][action]
            )

            # Move forward
            state = next_state
            action = next_action

    metrics = {
        "rewards": episode_rewards,
        "steps": episode_steps,
        "success": episode_success
    }

    return Q, metrics