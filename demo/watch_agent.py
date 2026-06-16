"""
Q-Learning FrozenLake — Visual Demo (Windows / local only, requires a display)
--------------------------------------------------------------------------
Watch a trained Q-learning agent play FrozenLake with live Q-values shown on
each tile. This script trains a fresh agent quickly (same algorithm as the
main notebook) and then renders it playing, for both the Non-Slippery and
Slippery settings.

NOT for Colab — pygame's "human" render mode needs a real display window.

Run locally with:
    pip install gymnasium[toy-text] numpy pygame
    python watch_agent.py
"""

import numpy as np
from frozen_lake_qviz import FrozenLakeQVizEnv

N_STATES = 16   # 4x4 grid
N_ACTIONS = 4   # left, down, right, up


def train_q_table(is_slippery: bool, n_episodes: int, alpha: float, gamma: float,
                   eps_start: float, eps_end: float, eps_decay_episodes: int, seed: int = 42):
    """Train a tabular Q-learning agent (headless) and return the learned Q-table."""
    env = FrozenLakeQVizEnv(is_slippery=is_slippery, render_mode=None)
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES, N_ACTIONS))

    for ep in range(n_episodes):
        frac = min(1.0, ep / eps_decay_episodes)
        epsilon = eps_start + (eps_end - eps_start) * frac
        state, _ = env.reset(seed=seed + ep)
        done = False
        steps = 0
        while not done and steps < 200:
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            best_next = np.max(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])
            state = next_state
            steps += 1

    env.close()
    return Q


def watch_agent(Q, is_slippery: bool, agent_label: str, n_episodes: int = 5):
    """Render the trained agent playing using the greedy policy."""
    env = FrozenLakeQVizEnv(is_slippery=is_slippery, render_mode="human", agent_label=agent_label)
    env.set_q(Q)

    for ep in range(1, n_episodes + 1):
        env.set_episode(ep)
        state, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < 200:
            action = int(np.argmax(Q[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

    env.close()


if __name__ == "__main__":
    print("Training Non-Slippery agent...")
    Q_non_slippery = train_q_table(
        is_slippery=False, n_episodes=5000, alpha=0.8, gamma=0.95,
        eps_start=1.0, eps_end=0.05, eps_decay_episodes=500,
    )
    print("Training Slippery agent...")
    Q_slippery = train_q_table(
        is_slippery=True, n_episodes=20000, alpha=0.1, gamma=0.99,
        eps_start=1.0, eps_end=0.01, eps_decay_episodes=9000,
    )

    print("Launching demo window — Non-Slippery Agent (close window or press ESC to advance)")
    watch_agent(Q_non_slippery, is_slippery=False, agent_label="Non-Slippery Agent", n_episodes=5)

    print("Launching demo window — Slippery Agent")
    watch_agent(Q_slippery, is_slippery=True, agent_label="Slippery Agent", n_episodes=5)

    print("Demo complete.")