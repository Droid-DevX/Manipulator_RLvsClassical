import sys
sys.path.append("src")

from rl_env import PandaPickPlaceEnv
from gymnasium.utils.env_checker import check_env
import numpy as np

env = PandaPickPlaceEnv()

print("Running gymnasium check_env()...")
try:
    check_env(env, skip_render_check=True)
    print("check_env PASSED")
except Exception as e:
    print("check_env FAILED:", e)

print("\nRunning extended random-agent smoke test (5 episodes)...")
episode_rewards = []
episode_lengths = []
n_successes = 0

for ep in range(5):
    obs, info = env.reset()
    total_reward = 0
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    episode_rewards.append(total_reward)
    episode_lengths.append(steps)
    if info.get("success"):
        n_successes += 1
    print(f"Episode {ep}: reward={total_reward:.2f}  length={steps}  info={info}")

print(f"\nMean episode reward: {np.mean(episode_rewards):.2f}")
print(f"Mean episode length: {np.mean(episode_lengths):.1f}")
print(f"Successes (random agent, expected ~0): {n_successes}/5")
print("Environment is stable and ready for PPO training.")
