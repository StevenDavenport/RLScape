# rl-scape

Minimal Gymnasium wrapper for a headless 2006Scape client.

## Quickstart

1) Start the server
2) Start the client in headless + RL mode
3) Run the Python env

### Start headless client

```bash
java -jar "../2006Scape/2006Scape Client/target/client-1.0-jar-with-dependencies.jar" \
  -headless -rl -rl-port 5656 -local -u agent -p agent
```

### Python usage

```python
from rl_scape import RLScapeEnv

env = RLScapeEnv(host="127.0.0.1", port=5656)
obs, info = env.reset()

# move to center
action = {"type": 1, "x": 382, "y": 251, "button": 0}
obs, reward, terminated, truncated, info = env.step(action)
```

Auto-launch server + client from Gym (default behavior, uses env defaults if not set):

```python
env = RLScapeEnv(username="Agent2")
```

Environment variables for defaults:

```bash
export RL_SCAPE_SERVER_DIR="/home/staff/steven/RuneScape/2006Scape/2006Scape Server"
export RL_SCAPE_CLIENT_DIR="/home/staff/steven/RuneScape/2006Scape/2006Scape Client"
export RL_SCAPE_JAVA_HOME="/home/staff/steven/local/jdk8/jdk8u482-b08"
export RL_SCAPE_USERNAME="Agent2"
```

## Action format

Dict with:
- `type`: 0 noop, 1 move, 2 left click, 3 right click
- `x`, `y`: mouse position

## Manual play (via Python)

Requires `pygame`:

```bash
pip install pygame
```

Run:

```bash
python examples/manual_play.py 127.0.0.1 5656 1
```

Arguments: `host port scale` (scale is an integer multiplier for window size).

## Launcher (start server + client from Python)

Quick start (auto-launches by default, auto-detects paths):

```bash
python examples/manual_play.py
```

Optional explicit paths:

```bash
python examples/manual_play.py --launch \
  "/home/staff/steven/RuneScape/2006Scape/2006Scape Server" \
  "/home/staff/steven/RuneScape/2006Scape/2006Scape Client" \
  "/home/staff/steven/local/jdk8/jdk8u482-b08"
```

Choose account name:

```bash
python examples/manual_play.py --name Agent2
```

To connect to an already running server/client:

```bash
python examples/manual_play.py --no-launch 127.0.0.1 5656 1
```
