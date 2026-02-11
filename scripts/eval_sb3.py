import argparse
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import rl_scape


def make_env(name, render_scale=1, render_fps=30):
    def _thunk():
        env = rl_scape.make(
            name=name,
            render_mode="human",
            render_scale=render_scale,
            render_fps=render_fps,
        )
        env.unwrapped.action_space = env.unwrapped.sb3_action_space
        return env

    return _thunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="agent")
    parser.add_argument("--model-path", default="experiments/checkpoints/ppo_rlscape")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--render-scale", type=int, default=1)
    parser.add_argument("--render-fps", type=int, default=30)
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args()

    vec_env = DummyVecEnv([make_env(args.name, args.render_scale, args.render_fps)])
    vec_env = VecTransposeImage(vec_env)

    model = PPO.load(args.model_path, env=vec_env)

    obs = vec_env.reset()
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    coord_samples = []
    for step in range(1, args.steps + 1):
        action, _ = model.predict(obs, deterministic=True)
        if action is not None:
            try:
                action_type = int(action[0][0])
                x = int(action[0][1])
                y = int(action[0][2])
                action_counts[action_type] = action_counts.get(action_type, 0) + 1
                coord_samples.append((x, y))
            except Exception:
                pass
        obs, _, dones, _ = vec_env.step(action)
        if dones[0]:
            obs = vec_env.reset()
        if args.log_every > 0 and step % args.log_every == 0:
            if coord_samples:
                xs = [c[0] for c in coord_samples]
                ys = [c[1] for c in coord_samples]
                coord_info = f" x[{min(xs)},{max(xs)}] y[{min(ys)},{max(ys)}]"
            else:
                coord_info = ""
            print(f"[eval] step={step} action_counts={action_counts}{coord_info}")
            coord_samples = []

    vec_env.close()


if __name__ == "__main__":
    main()
