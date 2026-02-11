from .env import RLScapeEnv

try:
    import gymnasium as gym
    from gymnasium.envs.registration import register

    if "RLScape-v0" not in gym.envs.registry:
        register(id="RLScape-v0", entry_point="rl_scape.env:RLScapeEnv")
except Exception:
    # Optional dependency or registry errors shouldn't break import.
    pass


def make(**kwargs):
    import gymnasium as gym

    return gym.make("RLScape-v0", **kwargs)


__all__ = ["RLScapeEnv", "make"]
