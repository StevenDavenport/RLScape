import argparse
import os
import sys

import numpy as np
import pygame

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import rl_scape


def _parse_sizes(text):
    sizes = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "x" not in part:
            raise ValueError(f"Invalid size '{part}', expected WxH")
        w_s, h_s = part.split("x", 1)
        sizes.append((int(w_s), int(h_s)))
    return sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="agent")
    parser.add_argument(
        "--sizes",
        default="765x503,640x420,512x336,384x252,256x168",
        help="Comma-separated WxH list",
    )
    parser.add_argument("--warmup-steps", type=int, default=1200)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--out-dir", default="assets/sample_frames")
    args = parser.parse_args()

    sizes = _parse_sizes(args.sizes)
    os.makedirs(args.out_dir, exist_ok=True)

    env = rl_scape.make(name=args.name)
    try:
        obs, _ = env.reset()
        steps = 0
        click_plan = {
            30: (455, 275),  # "Existing User" button (title screen)
            60: (380, 300),  # "Login" button (login screen)
            90: (380, 300),  # retry login
        }
        while steps < args.max_steps:
            if steps in click_plan:
                x, y = click_plan[steps]
                obs, _, _, _, _ = env.step({"type": 2, "x": x, "y": y})
            else:
                obs, _, _, _, _ = env.step({"type": 0, "x": 0, "y": 0})
            steps += 1
            if steps > 30:
                mini = obs[10:150, 560:730]
                chat = obs[340:503, 0:512]
                if mini.std() > 12.0 and chat.std() > 12.0:
                    break

        pygame.init()
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        raw_path = os.path.join(args.out_dir, "frame_raw.png")
        pygame.image.save(surf, raw_path)

        for w, h in sizes:
            resized = pygame.transform.smoothscale(surf, (w, h))
            out_path = os.path.join(args.out_dir, f"frame_{w}x{h}.png")
            pygame.image.save(resized, out_path)
        print(f"Saved frames to {args.out_dir}")
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
