import time
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from rl_scape import RLScapeEnv


env = RLScapeEnv(host="127.0.0.1", port=5656)
obs, info = env.reset()
print("obs shape:", obs.shape)

# Move cursor to center and left-click
w = obs.shape[1]
h = obs.shape[0]
center_x = w // 2
center_y = h // 2

env.step({"type": 1, "x": center_x, "y": center_y})
env.step({"type": 2, "x": center_x, "y": center_y})
env.step({"type": 3, "x": center_x, "y": center_y})

# Fetch a few frames
for _ in range(5):
    obs, _, _, _, _ = env.step({"type": 0, "x": 0, "y": 0})
    time.sleep(0.05)

env.close()
