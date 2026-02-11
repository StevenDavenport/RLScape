import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .bridge import RLBridgeClient
from .launcher import RLScapeLauncher


ACTION_NOOP = 0
ACTION_MOVE = 1
ACTION_LEFT_CLICK = 2
ACTION_RIGHT_CLICK = 3


class RLScapeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(
        self,
        host="127.0.0.1",
        port=5656,
        timeout=10.0,
        launch=True,
        server_dir=None,
        client_dir=None,
        java_home=None,
        username="agent",
        name=None,
        local=True,
        resize=(384, 252),
        render_mode="rgb_array",
        render_scale=1,
        render_fps=50,
        episode_length=10_000,
        reward_xp_scale=0.01,
        reward_level_bonus=10.0,
        sync_to_tick=True,
        tick_divisor=1,
        auto_calibrate_tick=True,
        calibrate_every=5000,
        calibrate_window_sec=1.5,
        target_tick_seconds=None,
        log_tick_sync=False,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.headless = True
        self._pygame = None
        self._screen = None
        self._clock = None
        self.render_scale = max(1, int(render_scale))
        self.render_fps = int(render_fps)
        self._client = RLBridgeClient(host=host, port=port, timeout=timeout)
        self._connected = False
        self._launcher = None
        self._launch_enabled = launch
        if name is not None:
            username = name
        if launch:
            self._launcher = RLScapeLauncher(
                server_dir=server_dir,
                client_dir=client_dir,
                java_home=java_home,
                port=port,
                username=username,
                local=local,
                headless=self.headless,
            )

        # Full client frame size: 765x503
        self.raw_width = 765
        self.raw_height = 503
        self.resize = resize
        if resize is None:
            self.width = self.raw_width
            self.height = self.raw_height
        else:
            self.width = int(resize[0])
            self.height = int(resize[1])

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 3),
            dtype=np.uint8,
        )

        # Minimal action: (type, x, y)
        # 0 noop, 1 move, 2 left click, 3 right click
        self.action_space = spaces.Dict(
            {
                "type": spaces.Discrete(4),
                "x": spaces.Box(low=0, high=self.width - 1, shape=(), dtype=np.int32),
                "y": spaces.Box(low=0, high=self.height - 1, shape=(), dtype=np.int32),
            }
        )
        self.sb3_action_space = spaces.MultiDiscrete([4, self.width, self.height])

        self._last_obs = None
        self._prev_state = None
        self.episode_length = int(episode_length)
        self._step_count = 0
        self.reward_xp_scale = float(reward_xp_scale)
        self.reward_level_bonus = float(reward_level_bonus)
        self.sync_to_tick = bool(sync_to_tick)
        self.tick_divisor = max(1, int(tick_divisor))
        self.auto_calibrate_tick = bool(auto_calibrate_tick)
        self.calibrate_every = int(calibrate_every)
        self.calibrate_window_sec = float(calibrate_window_sec)
        self.target_tick_seconds = target_tick_seconds
        self.log_tick_sync = bool(log_tick_sync)
        self._last_tick = None

    def _ensure_connected(self):
        if not self._connected:
            for _ in range(30):
                try:
                    self._client.connect()
                    self._connected = True
                    break
                except Exception:
                    time.sleep(0.1)
            if not self._connected:
                raise RuntimeError("Failed to connect to RL bridge")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._launcher is not None:
            self._launcher.start()
        self._ensure_connected()
        try:
            obs = self._read_frame()
            self._wait_for_ready()
            self._prev_state = self._wait_for_stable_state()
        except Exception:
            self.close()
            raise
        if self.auto_calibrate_tick:
            self._calibrate_tick_divisor()
        if self._prev_state is not None:
            self._last_tick = self._prev_state["loop_cycle"] // self.tick_divisor
        self._last_obs = obs
        self._step_count = 0
        info = {}
        return obs, info

    def step(self, action):
        self._ensure_connected()
        if isinstance(action, dict):
            action_type = int(action.get("type", ACTION_NOOP))
            x = int(action.get("x", 0))
            y = int(action.get("y", 0))
        else:
            action_type = int(action[0])
            x = int(action[1])
            y = int(action[2])

        x_raw, y_raw = self._to_raw_coords(x, y)

        tick_before = None
        if self.sync_to_tick:
            state_before = self._read_state()
            tick_before = state_before["loop_cycle"] // self.tick_divisor
            self._last_tick = tick_before

        if action_type == ACTION_MOVE:
            self._client.move(x_raw, y_raw)
        elif action_type == ACTION_LEFT_CLICK:
            self._client.move(x_raw, y_raw)
            self._client.down(1)
            self._client.up(1)
        elif action_type == ACTION_RIGHT_CLICK:
            self._client.move(x_raw, y_raw)
            self._client.down(3)
            self._client.up(3)

        obs = None
        state = None
        if self.sync_to_tick and tick_before is not None:
            while True:
                obs = self._read_frame(step=True)
                state = self._read_state()
                tick_after = state["loop_cycle"] // self.tick_divisor
                if tick_after > tick_before:
                    self._last_tick = tick_after
                    if self.log_tick_sync:
                        print(f"[rl-scape] tick {tick_before} -> {tick_after} action={action_type}")
                    break
        else:
            obs = self._read_frame(step=True)
            state = self._read_state()

        self._last_obs = obs
        if self.render_mode == "human":
            self.render()

        reward, reward_info = self._compute_reward(self._prev_state, state, action_type)
        self._prev_state = state
        terminated = False
        self._step_count += 1
        truncated = self._step_count >= self.episode_length
        info = {"step_count": self._step_count}
        info.update(reward_info)
        if self.auto_calibrate_tick and self.calibrate_every > 0:
            if self._step_count % self.calibrate_every == 0:
                self._calibrate_tick_divisor()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._last_obs is None:
                return None
            if self._pygame is None:
                import pygame

                self._pygame = pygame
                pygame.init()
                self._clock = pygame.time.Clock()
            frame = self._last_obs
            h, w, _ = frame.shape
            if self._screen is None:
                self._screen = self._pygame.display.set_mode((w * self.render_scale, h * self.render_scale))
                self._pygame.display.set_caption("rl-scape")
            surface = self._pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            if self.render_scale != 1:
                surface = self._pygame.transform.scale(surface, (w * self.render_scale, h * self.render_scale))
            self._screen.blit(surface, (0, 0))
            self._pygame.display.flip()
            self._pygame.event.pump()
            self._clock.tick(self.render_fps)
            return None
        return self._last_obs

    def close(self):
        if self._connected:
            try:
                self._client.close()
            finally:
                self._connected = False
        if self._launcher is not None:
            self._launcher.stop()
        if self._pygame is not None:
            self._pygame.quit()
            self._pygame = None
            self._screen = None
            self._clock = None

    def _read_frame(self, step=False):
        if step:
            width, height, channels, data = self._client.step()
        else:
            width, height, channels, data = self._client.frame()

        if channels != 3:
            raise RuntimeError(f"Unexpected channels: {channels}")
        if width != self.raw_width or height != self.raw_height:
            self.raw_width = width
            self.raw_height = height
            if self.resize is None:
                self.width = width
                self.height = height

        arr = np.frombuffer(data, dtype=np.uint8)
        arr = arr.reshape((height, width, 3))
        if self.resize is None:
            return arr
        return self._resize_nearest(arr, self.width, self.height)

    def _read_state(self):
        return self._client.state()

    def _wait_for_ready(self, poll_s=0.1, timeout_s=60.0):
        start = time.time()
        last_print = 0.0
        while True:
            ready = self._client.ready()
            if ready:
                return
            elapsed = time.time() - start
            if elapsed > timeout_s:
                raise RuntimeError("RL env READY timeout (60s). Login likely failed or UI not initialized.")
            if elapsed - last_print >= 5.0:
                print("[startup] waiting for READY...", flush=True)
                last_print = elapsed
            time.sleep(poll_s)

    def _wait_for_stable_state(self, stable_reads_required=3, poll_s=0.1):
        stable_reads = 0
        last = self._read_state()
        while stable_reads < stable_reads_required:
            time.sleep(poll_s)
            current = self._read_state()
            if current["total_xp"] == last["total_xp"] and current["total_levels"] == last["total_levels"]:
                stable_reads += 1
                print(f"[startup] stable_reads={stable_reads}", flush=True)
            else:
                stable_reads = 0
                last = current
        return last

    def _calibrate_tick_divisor(self):
        if not self.sync_to_tick:
            return
        t0 = time.time()
        state0 = self._read_state()
        start_cycle = state0["loop_cycle"]
        time.sleep(self.calibrate_window_sec)
        t1 = time.time()
        state1 = self._read_state()
        end_cycle = state1["loop_cycle"]
        dt = max(1e-6, t1 - t0)
        cycles_per_sec = max(1.0, (end_cycle - start_cycle) / dt)
        target = self._get_target_tick_seconds()
        if target <= 0:
            return
        new_divisor = max(1, int(round(cycles_per_sec * target)))
        if new_divisor != self.tick_divisor:
            self.tick_divisor = new_divisor
            print(f"[rl-scape] tick_divisor calibrated to {self.tick_divisor} (cycles/sec={cycles_per_sec:.2f}, target={target:.3f}s)")

    def _get_target_tick_seconds(self):
        if self.target_tick_seconds is not None:
            return float(self.target_tick_seconds)
        if self._launcher is None:
            return 0.6
        try:
            config_path = os.path.join(self._launcher.server_dir, self._launcher.server_config)
            with open(config_path, "r", encoding="utf-8") as f:
                data = __import__("json").load(f)
            cycle_ms = float(data.get("cycle_time_ms", 600.0))
            return max(0.01, cycle_ms / 1000.0)
        except Exception:
            return 0.6

    def _compute_reward(self, prev_state, state, action_type):
        if prev_state is None or state is None:
            return 0.0, {}
        xp_delta = max(0, state["total_xp"] - prev_state["total_xp"])
        level_delta = max(0, state["total_levels"] - prev_state["total_levels"])
        reward_xp = xp_delta * self.reward_xp_scale
        reward_level = level_delta * self.reward_level_bonus
        reward = reward_xp + reward_level
        info = {
            "reward_xp": reward_xp,
            "reward_level": reward_level,
            "total_xp": state["total_xp"],
            "total_levels": state["total_levels"],
            "skill_index": state.get("skill_index", -1),
            "skill_delta": state.get("skill_delta", 0),
        }
        return reward, info

    def _to_raw_coords(self, x, y):
        if self.resize is None:
            return x, y
        x_raw = int(x * self.raw_width / max(1, self.width))
        y_raw = int(y * self.raw_height / max(1, self.height))
        x_raw = max(0, min(self.raw_width - 1, x_raw))
        y_raw = max(0, min(self.raw_height - 1, y_raw))
        return x_raw, y_raw

    @staticmethod
    def _resize_nearest(img, out_w, out_h):
        h, w, _ = img.shape
        ys = (np.linspace(0, h - 1, out_h)).astype(np.int32)
        xs = (np.linspace(0, w - 1, out_w)).astype(np.int32)
        return img[ys][:, xs]
