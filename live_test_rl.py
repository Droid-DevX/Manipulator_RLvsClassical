import sys
sys.path.append("src")

import time
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from rl_env import PandaPickPlaceEnv

# Update these to whichever checkpoint you want to watch.
MODEL_PATH = "models/ppo_panda_pid_final.zip"
VECNORM_PATH = "models/ppo_panda_pid_vecnormalize.pkl"

N_EPISODES = 5


def make_venv():
    venv = DummyVecEnv([lambda: PandaPickPlaceEnv()])
    venv = VecNormalize.load(VECNORM_PATH, venv)
    venv.training = False
    venv.norm_reward = False
    return venv


if __name__ == "__main__":
    model = PPO.load(MODEL_PATH)
    venv = make_venv()

    # DummyVecEnv wraps a single PandaPickPlaceEnv -- grab it directly so
    # the viewer can render its actual mujoco model/data (VecNormalize only
    # transforms observations, it doesn't expose the sim itself).
    raw_env = venv.venv.envs[0]

    # The current environment advances 20 MuJoCo physics steps per PPO
    # action. Pace against that actual simulated duration.
    sim_dt_per_action = raw_env.model.opt.timestep * 20

    with mujoco.viewer.launch_passive(raw_env.model, raw_env.data) as viewer:
        # Let PandaPickPlaceEnv sync this same viewer during physics
        # substeps. This gives smoother motion than one sync per PPO action.
        raw_env.render_mode = "human"
        raw_env.viewer = viewer

        for ep in range(N_EPISODES):
            if not viewer.is_running():
                break

            obs = venv.reset()
            done = False
            ep_reward = 0.0
            steps = 0
            info = {}

            print(f"\n--- Episode {ep} ---")
            while not done and viewer.is_running():
                action, _ = model.predict(obs, deterministic=True)
                step_t0 = time.perf_counter()

                obs, reward, done_arr, infos = venv.step(action)
                done = bool(done_arr[0])
                info = infos[0]
                ep_reward += float(reward[0])
                steps += 1

                viewer.sync()

                # Only sleep for the remainder of the simulated action time.
                # This avoids the extra fixed delay that made playback feel
                # unnecessarily slow.
                elapsed = time.perf_counter() - step_t0
                time.sleep(max(0.0, sim_dt_per_action - elapsed))

            print(f"steps={steps}  total_reward={ep_reward:.1f}  "
                  f"success={info.get('success')}  "
                  f"final_dist={info.get('distance_obj_to_target', float('nan'))*1000:.1f}mm  "
                  f"max_obj_height={info.get('obj_height', float('nan'))*1000:.1f}mm  "
                  f"is_holding={info.get('is_holding')}")

        print(f"\nFinished {N_EPISODES} episodes. Viewer staying open — close window to exit.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)
