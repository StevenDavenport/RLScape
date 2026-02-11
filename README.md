# RLScape Monorepo

This repository combines the upstream 2006Scape codebase with RL environment and training tooling.

## Repository layout

- `third_party/2006scape/`: upstream game client/server source
- `src/rl_scape/`: Python RL environment package
- `scripts/`: runnable scripts (manual play, train, eval, capture)
- `assets/sample_frames/`: sample captured frames
- `experiments/`: experiment outputs and checkpoints
- `docs/`: project docs
- `configs/`: config placeholders for env/training/eval

## Python setup

```bash
pip install -e .
```

Then run scripts from the repo root, for example:

```bash
python scripts/manual_play.py
python scripts/train_sb3.py
```
