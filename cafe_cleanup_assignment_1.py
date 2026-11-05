"""Cafe Closing Cleanup - CENG 567 Assignment 1.
Actions: 0 = clean, 1 = rest, 2 = wait; r = restart, q = quit.
"""

import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CafeCleanupEnv(gym.Env):
    """State = (dirty tables, energy, time left); actions = clean/rest/wait."""

    metadata = {"render_modes": ["human", "ansi"]}
    ACTIONS = ("clean", "rest", "wait")
    REWARDS = ("sparse", "efficient", "progress")

    def __init__(self, reward_mode="sparse", render_mode=None):
        if reward_mode not in self.REWARDS:
            raise ValueError(f"Choose one of {self.REWARDS}")
        if render_mode not in (None, "human", "ansi"):
            raise ValueError("render_mode must be None, 'human', or 'ansi'")
        self.reward_mode = reward_mode
        self.render_mode = render_mode
        self.observation_space = spaces.MultiDiscrete([4, 3, 6])
        self.action_space = spaces.Discrete(3)
        self.state = None

    @staticmethod
    def status(state):
        tables, _, time_left = state
        if tables == 0:
            return "success"
        if time_left == 0:
            return "timeout"
        return "ongoing"

    def transition(self, state, action):
        """Pure deterministic model, also useful for tabular algorithms later."""
        if not self.action_space.contains(action):
            raise ValueError("Action must be 0, 1, or 2")
        if not self.observation_space.contains(np.asarray(state)):
            raise ValueError("State is outside the observation space")
        state = tuple(int(x) for x in state)
        if self.status(state) != "ongoing":
            return state, 0.0, True, {
                "status": self.status(state), "cleaned": False, "invalid": False,
            }

        tables, energy, time_left = state
        cleaned = False
        invalid = False
        if action == 0:                    
            if energy > 0:
                tables -= 1
                energy -= 1
                cleaned = True
            else:
                invalid = True
        elif action == 1: 
            if energy < 2:
                energy += 1
            else:
                invalid = True

        time_left -= 1                
        next_state = (tables, energy, time_left)
        status = self.status(next_state)
        done = status != "ongoing"
        reward = 10.0 if status == "success" else (-10.0 if done else 0.0)
        if self.reward_mode in ("efficient", "progress"):
            reward -= 1.0 + 2.0 * invalid
        if self.reward_mode == "progress":
            reward += 2.0 * cleaned
        return next_state, float(reward), done, {
            "status": status, "cleaned": cleaned, "invalid": invalid,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = (3, 2, 5)
        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.int64), {
            "status": "ongoing", "cleaned": False, "invalid": False,
        }

    def step(self, action):
        if self.state is None or self.status(self.state) != "ongoing":
            raise gym.error.ResetNeeded("Call reset() before a new episode")
        self.state, reward, done, info = self.transition(self.state, action)
        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.int64), reward, done, False, info

    def render(self):
        if self.state is None:
            raise gym.error.ResetNeeded("Call reset() first")
        tables, energy, time_left = self.state
        message = (f"Dirty tables: {tables} | Energy: {energy}/2 | "
                   f"Time left: {time_left} | {self.status(self.state)}")
        if self.render_mode == "human":
            print(message)
        elif self.render_mode == "ansi":
            return message


SCENARIOS = {
    "quick_success": [0, 0, 1, 0],
    "delayed_success": [2, 0, 0, 1, 0],
    "invalid_clean": [0, 0, 0, 1, 0],
    "partial_failure": [0, 0, 2, 2, 2],
    "waiting_failure": [2, 2, 2, 2, 2],
}


def episode(reward_mode, actions):
    env = CafeCleanupEnv(reward_mode)
    observation, _ = env.reset(seed=42)
    rows = []
    for turn, action in enumerate(actions, start=1):
        before = observation.tolist()
        observation, reward, terminated, truncated, info = env.step(action)
        rows.append({
            "turn": turn, "state": before, "action": env.ACTIONS[action],
            "next_state": observation.tolist(), "reward": reward,
            "terminated": terminated, "truncated": truncated, **info,
        })
        if terminated or truncated:
            break
    env.close()
    return rows


def write_examples():
    output = Path(__file__).resolve().parent / "results"
    output.mkdir(exist_ok=True)
    summary = {}
    for scenario, actions in SCENARIOS.items():
        summary[scenario] = {}
        for mode in CafeCleanupEnv.REWARDS:
            rows = episode(mode, actions)
            with (output / f"{scenario}_{mode}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            summary[scenario][mode] = {
                "turns": len(rows), "status": rows[-1]["status"],
                "total_return": sum(row["reward"] for row in rows),
            }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def check_environment():
    """Optional verification; no separate test script is required."""
    from itertools import product
    from gymnasium.utils.env_checker import check_env

    envs = [CafeCleanupEnv(mode) for mode in CafeCleanupEnv.REWARDS]
    for env in envs:
        check_env(env, skip_render_check=True)
    for state in product(range(4), range(3), range(6)):
        for action in range(3):
            results = [env.transition(state, action) for env in envs]
            next_state, _, done, info = results[0]
            assert envs[0].observation_space.contains(np.array(next_state))
            for result in results[1:]:
                assert (next_state, done, info) == (result[0], result[2], result[3])
            if envs[0].status(state) == "ongoing":
                assert next_state[2] == state[2] - 1
            else:
                assert next_state == state and done
                assert all(result[1] == 0.0 for result in results)
    for env in envs:
        env.close()
    print("Gymnasium checks passed for all rewards; 648 model evaluations passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward", choices=CafeCleanupEnv.REWARDS, default="sparse")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--examples", action="store_true", help="generate report trajectories")
    mode.add_argument("--check", action="store_true", help="check the environment API and dynamics")
    args = parser.parse_args()
    if args.examples:
        write_examples()
        return
    if args.check:
        check_environment()
        return
    env = CafeCleanupEnv(args.reward, render_mode="ansi")
    observation, _ = env.reset(seed=42)
    print("0 = clean, 1 = rest, 2 = wait; r = reset, q = quit")
    print("State:", observation.tolist(), env.render())
    try:
        while True:
            command = input("Action> ").strip().lower()
            if command == "q":
                break
            if command == "r":
                observation, _ = env.reset()
                print("State:", observation.tolist(), env.render())
                continue
            if command not in ("0", "1", "2"):
                print("Enter 0, 1, 2, r, or q.")
                continue
            if env.status(env.state) != "ongoing":
                print("Episode ended. Press r to restart or q to quit.")
                continue
            before = observation.tolist()
            observation, reward, terminated, truncated, info = env.step(int(command))
            print(f"{before} -> {env.ACTIONS[int(command)]} -> {observation.tolist()}")
            print(f"Reward: {reward}; terminated={terminated}; truncated={truncated}")
            print(env.render(), "| invalid command:", info["invalid"])
    except (EOFError, KeyboardInterrupt):
        print("\nSession closed.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
