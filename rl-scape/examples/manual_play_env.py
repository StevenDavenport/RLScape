import argparse
import json
import os
import sys

import numpy as np
import pygame

import rl_scape


ACTION_NOOP = 0
ACTION_MOVE = 1
ACTION_LEFT_CLICK = 2
ACTION_RIGHT_CLICK = 3

SKILL_NAMES = [
    "Attack",
    "Defence",
    "Strength",
    "Hitpoints",
    "Ranged",
    "Prayer",
    "Magic",
    "Cooking",
    "Woodcutting",
    "Fletching",
    "Fishing",
    "Firemaking",
    "Crafting",
    "Smithing",
    "Mining",
    "Herblore",
    "Agility",
    "Thieving",
    "Slayer",
    "Farming",
    "Runecrafting",
]


def main(name="agent", scale=1, tick_fps=50, human_speed=False):
    env = rl_scape.make(name=name, render_mode="rgb_array")
    if human_speed and getattr(env, "unwrapped", None) is not None:
        launcher = getattr(env.unwrapped, "_launcher", None)
        if launcher is not None:
            launcher.auto_tune = False
            config_path = os.path.join(launcher.server_dir, launcher.server_config)
            if os.path.isfile(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["cycle_time_ms"] = 600
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, sort_keys=False)
    obs, _ = env.reset()

    pygame.init()
    pygame.display.set_caption("rl-scape env manual play")

    height, width, _ = obs.shape
    win_w = width * scale
    win_h = height * scale
    screen = pygame.display.set_mode((win_w, win_h))
    clock = pygame.time.Clock()

    running = True
    last_action = {"type": ACTION_NOOP, "x": 0, "y": 0}
    while running:
        action = {"type": ACTION_NOOP, "x": last_action["x"], "y": last_action["y"]}
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.MOUSEMOTION:
                x, y = event.pos
                x //= scale
                y //= scale
                action = {"type": ACTION_MOVE, "x": int(x), "y": int(y)}
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                x //= scale
                y //= scale
                if event.button == 1:
                    action = {"type": ACTION_LEFT_CLICK, "x": int(x), "y": int(y)}
                elif event.button == 3:
                    action = {"type": ACTION_RIGHT_CLICK, "x": int(x), "y": int(y)}

        obs, reward, terminated, truncated, info = env.step(action)
        last_action = action

        frame = obs
        surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        if scale != 1:
            surface = pygame.transform.scale(surface, (win_w, win_h))
        screen.blit(surface, (0, 0))
        pygame.display.flip()

        if reward > 0 and (info.get("reward_xp", 0) > 0 or info.get("reward_level", 0) > 0):
            parts = []
            if info.get("reward_xp", 0) > 0:
                skill_idx = int(info.get("skill_index", -1))
                skill_delta = int(info.get("skill_delta", 0))
                skill_name = SKILL_NAMES[skill_idx] if 0 <= skill_idx < len(SKILL_NAMES) else f"skill_{skill_idx}"
                if skill_idx >= 0:
                    parts.append(f"xp={info['reward_xp']:.4f} {skill_name}+{skill_delta}")
                else:
                    parts.append(f"xp={info['reward_xp']:.4f} total_xp={info.get('total_xp')}")
            if info.get("reward_level", 0) > 0:
                parts.append(f"lvl={info['reward_level']:.2f} total_lvl={info.get('total_levels')}")
            detail = " ".join(parts)
            print(f"[RL-REWARD] total={reward:.4f} {detail}")

        if terminated or truncated:
            obs, _ = env.reset()

        clock.tick(tick_fps)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="agent")
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--human-speed", action="store_true")
    args = parser.parse_args()
    main(name=args.name, scale=args.scale, tick_fps=args.fps, human_speed=args.human_speed)
