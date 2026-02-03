# utilities.py

from config import ACTIONS, GRID_ROWS, GRID_COLS, HOLES, GOAL_STATE
import random

# 1. Epsilon-greedy action selection policy
def epsilon_greedy(Q, state, epsilon):
    """
    Select an action using the textbook epsilon-greedy policy:
    - Greedy action gets probability: 1 - epsilon + epsilon/|A|
    - All other actions get probability: epsilon/|A|
    """
    q_values = Q[state]
    max_q = max(q_values.values())

    # Identify greedy actions (handle ties)
    greedy_actions = [a for a, q in q_values.items() if q == max_q]

    # Choose one greedy action uniformly if ties exist
    greedy_action = random.choice(greedy_actions)

    num_actions = len(ACTIONS)
    probs = []
    
    # Build probability distribution
    for action in ACTIONS:
        if action == greedy_action:
            probs.append(1 - epsilon + epsilon / num_actions)
        else:
            probs.append(epsilon / num_actions)

    # Sample according to probability distribution
    return random.choices(ACTIONS, weights=probs, k=1)[0]


# 2. Print policy in grid format
ACTION_TO_ARROW = {
    'UP': '↑',
    'DOWN': '↓',
    'LEFT': '←',
    'RIGHT': '→'
}

def print_policy(Q, env):
    """
    Print the optimal policy derived from Q in grid form.
    """
    for r in range(GRID_ROWS):
        row = []
        for c in range(GRID_COLS):
            state = (r, c)

            if state == GOAL_STATE:
                row.append(' G ')
            elif state in HOLES:
                row.append(' H ')
            else:
                s_idx = env.state_to_index(state)
                best_action = max(Q[s_idx], key=Q[s_idx].get)
                row.append(f' {ACTION_TO_ARROW[best_action]} ')
        print(''.join(row))
    print()


# 3. Compute moving average
def moving_average(data, window=100):
    return [
        sum(1 for x in data[max(0, i-window):i] if x == 1) / max(1, len(data[max(0, i-window):i]))
        for i in range(len(data))
    ]
