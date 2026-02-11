import argparse
import csv
import os
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

import rl_scape


class RewardStepCounter(BaseCallback):
    def __init__(self):
        super().__init__()
        self.reward_steps = 0
        self.total_steps = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        if rewards is not None:
            self.reward_steps += int((rewards > 0).sum())
            self.total_steps += len(rewards)
        return True


class ProgressLogger(BaseCallback):
    def __init__(self, log_every=10_000, log_dir="runs", run_name="run"):
        super().__init__()
        self.log_every = int(log_every)
        self.step_rewards = []
        self.total_steps = 0
        self.rows = []
        self._next_log = self.log_every
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, f"{run_name}_progress_{stamp}.csv")
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timesteps", "avg_reward"])
            writer.writeheader()

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        if rewards is not None:
            self.step_rewards.extend(list(rewards))
            self.total_steps += len(rewards)
        if self.total_steps >= self._next_log:
            avg_reward = sum(self.step_rewards) / max(1, len(self.step_rewards))
            row = {
                "timesteps": self.total_steps,
                "avg_reward": avg_reward,
            }
            self.rows.append(row)
            print(f"[progress] steps={self.total_steps} avg_reward={avg_reward:.6f}")
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["timesteps", "avg_reward"])
                writer.writerow(row)
            self.step_rewards = []
            self._next_log += self.log_every
        return True


class CheckpointSaver(BaseCallback):
    def __init__(self, save_path, save_every=100_000):
        super().__init__()
        self.save_path = save_path
        self.save_every = int(save_every)
        self._next_save = self.save_every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_save:
            path = f"{self.save_path}_step_{self.num_timesteps}"
            self.model.save(path)
            print(f"[checkpoint] saved {path}")
            self._next_save += self.save_every
        return True


def make_env(name):
    def _thunk():
        print("[startup] creating env", flush=True)
        env = rl_scape.make(name=name, render_mode="rgb_array", log_tick_sync=False)
        env.unwrapped.action_space = env.unwrapped.sb3_action_space
        print("[startup] env created", flush=True)
        return Monitor(env)

    return _thunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="agent")
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", default="models/ppo_rlscape")
    parser.add_argument("--log-dir", default="runs")
    parser.add_argument("--log-every", type=int, default=10_000)
    parser.add_argument("--save-every", type=int, default=100_000)
    args = parser.parse_args()

    print("[startup] building vec env", flush=True)
    vec_env = DummyVecEnv([make_env(args.name)])
    vec_env = VecTransposeImage(vec_env)
    print("[startup] vec env ready", flush=True)

    print("[startup] building model", flush=True)
    model = PPO(
        "CnnPolicy",
        vec_env,
        verbose=1,
        seed=args.seed,
        n_steps=512,
        batch_size=64,
        ent_coef=0.01,
    )
    print("[startup] model ready", flush=True)
    reward_counter = RewardStepCounter()
    progress_logger = ProgressLogger(log_every=args.log_every, log_dir=args.log_dir, run_name=args.name)
    checkpoint_saver = CheckpointSaver(args.save_path, save_every=args.save_every)
    try:
        model.learn(total_timesteps=args.total_steps, callback=[reward_counter, progress_logger, checkpoint_saver])
        model.save(args.save_path)
    finally:
        vec_env.close()
        print(f"Agent got reward in {reward_counter.reward_steps} steps out of {reward_counter.total_steps}.")
        _write_progress_plot(progress_logger.rows, progress_logger.csv_path)


def _write_progress_plot(rows, csv_path):
    if not rows:
        print("No progress logs collected.")
        return
    png_path = csv_path.replace("_progress_", "_avg_reward_").replace(".csv", ".png")
    try:
        import matplotlib.pyplot as plt

        xs = [e["timesteps"] for e in rows]
        ys = [e["avg_reward"] for e in rows]
        plt.figure(figsize=(10, 4))
        plt.plot(xs, ys, linewidth=1)
        plt.title("Avg Reward vs Timesteps")
        plt.xlabel("Timesteps")
        plt.ylabel("Avg Reward")
        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()
        print(f"Saved progress plot to {png_path}")
    except Exception as exc:
        print(f"Saved progress CSV to {csv_path}; plot failed ({exc}).")


if __name__ == "__main__":
    main()
