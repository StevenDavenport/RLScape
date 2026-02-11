import json
import os
import sys
import time

import numpy as np
import pygame

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from rl_scape.bridge import RLBridgeClient
from rl_scape.launcher import RLScapeLauncher


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


def _compute_reward(prev_state, state, action_type):
    if prev_state is None or state is None:
        return 0.0, {}
    reward_xp_scale = 0.01
    reward_level_bonus = 10.0
    xp_delta = max(0, state["total_xp"] - prev_state["total_xp"])
    level_delta = max(0, state["total_levels"] - prev_state["total_levels"])

    reward_xp = xp_delta * reward_xp_scale
    reward_level = level_delta * reward_level_bonus

    reward = reward_xp + reward_level
    info = {
        "reward_xp": reward_xp,
        "reward_level": reward_level,
        "xp_delta": xp_delta,
        "level_delta": level_delta,
        "skill_index": state.get("skill_index", -1),
        "skill_delta": state.get("skill_delta", 0),
    }
    return reward, info


def _default_paths():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    server_dir = os.path.join(base, "third_party", "2006scape", "2006Scape Server")
    client_dir = os.path.join(base, "third_party", "2006scape", "2006Scape Client")
    java_home = os.environ.get("JAVA_HOME", "/home/staff/steven/local/jdk8/jdk8u482-b08")
    return server_dir, client_dir, java_home


def main(
    host="127.0.0.1",
    port=5656,
    scale=1,
    launch=None,
    username="agent",
    normal_speed=True,
):
    launcher = None
    if launch:
        server_dir, client_dir, java_home = launch
        if normal_speed:
            config_path = os.path.join(server_dir, "ServerConfig.json")
            if os.path.isfile(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["cycle_time_ms"] = 600
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, sort_keys=False)
        launcher = RLScapeLauncher(
            server_dir=server_dir,
            client_dir=client_dir,
            java_home=java_home,
            port=port,
            username=username,
            local=True,
            auto_tune=not normal_speed,
        )
        launcher.start()
        time.sleep(2.0)

    client = RLBridgeClient(host=host, port=port, timeout=10.0)
    client.connect()

    pygame.init()
    pygame.display.set_caption("rl-scape manual play")

    # Prime a frame to get size
    width, height, channels, data = client.frame()
    if channels != 3:
        raise RuntimeError(f"Unexpected channels: {channels}")

    win_w = width * scale
    win_h = height * scale
    screen = pygame.display.set_mode((win_w, win_h))
    clock = pygame.time.Clock()

    running = True
    middle_down = False
    last_mouse_pos = None
    last_action_type = ACTION_NOOP
    prev_state = client.state()
    stable_reads = 0
    last = prev_state
    while stable_reads < 3:
        time.sleep(0.1)
        current = client.state()
        if current["total_xp"] == last["total_xp"] and current["total_levels"] == last["total_levels"]:
            stable_reads += 1
        else:
            stable_reads = 0
            last = current
        prev_state = current
    while running:
        last_action_type = ACTION_NOOP
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEMOTION:
                x, y = event.pos
                x //= scale
                y //= scale
                client.move(x, y)
                last_action_type = ACTION_MOVE
                if middle_down and last_mouse_pos is not None:
                    dx = x - last_mouse_pos[0]
                    dy = y - last_mouse_pos[1]
                    if dx != 0 or dy != 0:
                        client.drag(dx, dy)
                last_mouse_pos = (x, y)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                x //= scale
                y //= scale
                client.move(x, y)
                if event.button == 1:
                    client.down(1)
                    last_action_type = ACTION_LEFT_CLICK
                elif event.button == 2:
                    middle_down = True
                    client.down(2)
                elif event.button == 3:
                    client.down(3)
                    last_action_type = ACTION_RIGHT_CLICK
            elif event.type == pygame.MOUSEBUTTONUP:
                x, y = event.pos
                x //= scale
                y //= scale
                client.move(x, y)
                if event.button == 1:
                    client.up(1)
                elif event.button == 2:
                    middle_down = False
                    client.up(2)
                elif event.button == 3:
                    client.up(3)

        width, height, channels, data = client.step()
        frame = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        if scale != 1:
            surface = pygame.transform.scale(surface, (win_w, win_h))
        screen.blit(surface, (0, 0))
        pygame.display.flip()

        state = client.state()
        reward, info = _compute_reward(prev_state, state, last_action_type)
        prev_state = state
        if reward > 0 and (info["reward_xp"] > 0 or info["reward_level"] > 0):
            parts = []
            if info["reward_xp"] > 0:
                skill_idx = int(info.get("skill_index", -1))
                skill_delta = int(info.get("skill_delta", 0))
                skill_name = SKILL_NAMES[skill_idx] if 0 <= skill_idx < len(SKILL_NAMES) else f"skill_{skill_idx}"
                if skill_idx >= 0:
                    parts.append(f"xp={info['reward_xp']:.4f} {skill_name}+{skill_delta}")
                else:
                    parts.append(f"xp={info['reward_xp']:.4f}(Δ{info['xp_delta']})")
            if info["reward_level"] > 0:
                parts.append(f"lvl={info['reward_level']:.2f}(Δ{info['level_delta']})")
            detail = " ".join(parts)
            print(f"[RL-REWARD] total={reward:.4f} {detail}")

        clock.tick(50)

    client.close()
    pygame.quit()
    if launcher:
        launcher.stop()


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 5656
    scale = 1
    launch = None
    java_home_override = None
    username = "agent"
    normal_speed = True
    args = list(sys.argv[1:])
    no_launch = False
    if "--no-launch" in args:
        no_launch = True
        args.remove("--no-launch")
    if "--java-home" in args:
        idx = args.index("--java-home")
        try:
            java_home_override = args[idx + 1]
        except IndexError:
            print("--java-home requires a path")
            sys.exit(1)
        del args[idx:idx + 2]
    if "--name" in args:
        idx = args.index("--name")
        try:
            username = args[idx + 1]
        except IndexError:
            print("--name requires a value")
            sys.exit(1)
        del args[idx:idx + 2]

    if "--fast-ticks" in args:
        normal_speed = False
        args.remove("--fast-ticks")

    if "--launch" in args:
        idx = args.index("--launch")
        # optional: --launch <server_dir> <client_dir> <java_home>
        if len(args) >= idx + 4:
            server_dir = args[idx + 1]
            client_dir = args[idx + 2]
            java_home = args[idx + 3]
            launch = (server_dir, client_dir, java_home)
            del args[idx:idx + 4]
        else:
            server_dir, client_dir, java_home = _default_paths()
            if java_home_override:
                java_home = java_home_override
            launch = (server_dir, client_dir, java_home)
            del args[idx:idx + 1]
    elif not no_launch:
        server_dir, client_dir, java_home = _default_paths()
        if java_home_override:
            java_home = java_home_override
        launch = (server_dir, client_dir, java_home)

    if len(args) > 0:
        host = args[0]
    if len(args) > 1:
        port = int(args[1])
    if len(args) > 2:
        scale = int(args[2])

    main(host=host, port=port, scale=scale, launch=launch, username=username, normal_speed=normal_speed)
