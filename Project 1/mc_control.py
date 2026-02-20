"""
Monte Carlo Control (First-Visit, Incremental)

Key Idea:
- Learn from complete episodes.
- Update Q-values after episode ends.
- No bootstrapping.
"""

from collections import defaultdict
from config import *
from misc import epsilon_greedy

def monte_carlo_control(env):

    # Initialize Q-table:
    # For every new state encountered,
    # create a dictionary mapping each action → 0.0
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    episode_rewards = []
    episode_steps = []
    episode_success = []

    alpha = 0.01  # incremental learning rate

    # ==========================================
    # Main training loop over episodes
    # ==========================================
    for episode in range(NUM_EPISODES):

        episode_data = []  # will store (state, action, reward)
        state = env.reset()
        total_reward = 0

        # --------------------------------------
        # Generate one full episode
        # --------------------------------------
        for step in range(MAX_STEPS_PER_EPISODE):

            # Choose action via epsilon-greedy
            action = epsilon_greedy(Q, state, EPSILON)

            # Take action in environment
            next_state, reward, done = env.step(action)

            # Store transition
            episode_data.append((state, action, reward))

            total_reward += reward
            state = next_state

            if done:
                break

        # --------------------------------------
        # Backward return computation
        # --------------------------------------
        G = 0
        visited = set()

        # Loop backward through episode
        for state, action, reward in reversed(episode_data):

            # Compute return
            G = DISCOUNT * G + reward

            # First-visit condition
            if (state, action) not in visited:
                visited.add((state, action))

                # Incremental MC update
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