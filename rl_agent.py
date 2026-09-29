"""
Q-Learning agent for the grid environment, following the class reference
approach: Q(s, a) is approximated with one scikit-learn SGDRegressor per
action, trained incrementally with partial_fit() on observed transitions
(instead of a plain lookup table).

TODO (Juan Pablo): the hyperparameters below are a reasonable starting
point, not a tuned result. Please retrain with a few different values,
compare the success rate / average reward you get from `train()`, pick the
configuration you consider best, and explain that choice (and what you
tried) in the report.
"""

import random
import numpy as np
from sklearn.linear_model import SGDRegressor

from rl_environment import GridEnvironment, ACTIONS

# Hyperparameters -------------------------------------------------------
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
GAMMA = 0.95
N_EPISODES = 800
MAX_STEPS_PER_EPISODE = 150
MAX_STEPS_EVALUATION = 100


def featurize(env, state):
    """One-hot encode (row, col) over all cells in the grid.

    This lets a linear model (SGDRegressor) learn an independent value per
    state per action, similar in spirit to a tabular Q-table, while still
    using the predict()/partial_fit() workflow required by the assignment.
    """
    r, c = state
    vec = np.zeros(env.rows * env.cols, dtype=float)
    vec[r * env.cols + c] = 1.0
    return vec.reshape(1, -1)


class QLearningAgent:
    def __init__(self, env):
        self.env = env
        self.regressors = {a: SGDRegressor(max_iter=1, learning_rate="constant", eta0=0.05) for a in ACTIONS}
        self.fitted = {a: False for a in ACTIONS}

    def predict_q(self, state, action):
        if not self.fitted[action]:
            return 0.0
        x = featurize(self.env, state)
        return float(self.regressors[action].predict(x)[0])

    def q_values(self, state):
        return {a: self.predict_q(state, a) for a in ACTIONS}

    def best_action(self, state):
        qvals = self.q_values(state)
        best_value = max(qvals.values())
        best_actions = [a for a, v in qvals.items() if v == best_value]
        return random.choice(best_actions)

    def update(self, state, action, target):
        x = featurize(self.env, state)
        self.regressors[action].partial_fit(x, [target])
        self.fitted[action] = True

    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.choice(ACTIONS)
        return self.best_action(state)


def train(env=None, episodes=N_EPISODES, max_steps=MAX_STEPS_PER_EPISODE,
          gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_min=EPSILON_MIN,
          epsilon_decay=EPSILON_DECAY, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    env = env or GridEnvironment()
    agent = QLearningAgent(env)
    epsilon = epsilon_start
    history = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done and steps < max_steps:
            action = agent.choose_action(state, epsilon)
            next_state, reward, done, _cell_type = env.step(state, action)

            best_next_q = max(agent.predict_q(next_state, a) for a in ACTIONS)
            target = reward + (0.0 if done else gamma * best_next_q)
            agent.update(state, action, target)

            state = next_state
            total_reward += reward
            steps += 1

        success = state == env.target
        history.append({
            "episode": episode,
            "total_reward": round(total_reward, 2),
            "steps": steps,
            "success": success,
            "epsilon": round(epsilon, 4),
        })

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return agent, history


def evaluate(env, agent, max_steps=MAX_STEPS_EVALUATION):
    """Run the greedy (epsilon = 0) policy once and log every step."""
    state = env.reset()
    steps_log = []
    total_reward = 0.0
    done = False
    step_num = 0
    visited_state_actions = set()
    looped = False

    while not done and step_num < max_steps:
        action = agent.best_action(state)

        state_action = (state, action)
        if state_action in visited_state_actions:
            looped = True
            break
        visited_state_actions.add(state_action)

        next_state, reward, done, cell_type = env.step(state, action)
        step_num += 1
        steps_log.append({
            "step": step_num,
            "state": state,
            "action": action,
            "next_state": next_state,
            "cell_type": cell_type,
            "reward": reward,
        })
        total_reward += reward
        state = next_state

    return {
        "steps_log": steps_log,
        "total_reward": round(total_reward, 2),
        "steps": step_num,
        "goal_reached": state == env.target,
        "looped": looped,
        "path": [row["state"] for row in steps_log] + ([state] if steps_log else [env.start]),
    }


def get_q_table(env, agent):
    """Q-values for every non-wall cell, for display in the UI/report."""
    rows = []
    for r in range(env.rows):
        for c in range(env.cols):
            if env.layout[r][c] == "#":
                continue
            qvals = agent.q_values((r, c))
            rows.append({
                "state": (r, c),
                "cell_type": env.cell_type((r, c)),
                "up": round(qvals["Up"], 2),
                "down": round(qvals["Down"], 2),
                "left": round(qvals["Left"], 2),
                "right": round(qvals["Right"], 2),
            })
    return rows


def training_summary(history):
    total_episodes = len(history)
    successful = sum(1 for h in history if h["success"])
    avg_reward = sum(h["total_reward"] for h in history) / total_episodes if total_episodes else 0.0
    return {
        "total_episodes": total_episodes,
        "successful_episodes": successful,
        "success_rate": round(successful / total_episodes * 100, 2) if total_episodes else 0.0,
        "average_reward": round(avg_reward, 2),
        "final_epsilon": history[-1]["epsilon"] if history else None,
    }


if __name__ == "__main__":
    env = GridEnvironment()
    agent, history = train(env)
    summary = training_summary(history)
    print("Training summary:", summary)

    result = evaluate(env, agent)
    print("Evaluation goal_reached:", result["goal_reached"], "steps:", result["steps"], "total_reward:", result["total_reward"])
    print("Path:", result["path"])
