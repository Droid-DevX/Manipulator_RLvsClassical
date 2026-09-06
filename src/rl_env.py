import sys
sys.path.append(".")

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

MODEL_PATH = "mujoco_menagerie/franka_emika_panda/pick_scene.xml"


class PandaPickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)

        self.site_id = self.model.site("attachment_site").id
        self.object_body_id = self.model.body("object").id
        self.gripper_actuator = 7

        self.n_arm_joints = 7
        self.max_delta = 0.05
        self.max_episode_steps = 300
        self.current_step = 0

        self.target_pos = np.array([0.45, -0.3, 0.02])

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        # Bounded observation space using real joint limits + sensible physical bounds
        joint_low = self.model.jnt_range[:7, 0]
        joint_high = self.model.jnt_range[:7, 1]

        vel_bound = 10.0     # rad/s, generous bound for joint velocities
        pos_bound = 2.0      # meters, generous bound for any 3D position in the scene
        gripper_low, gripper_high = 0.0, 0.045

        obs_low = np.concatenate([
            joint_low, -vel_bound * np.ones(7),
            -pos_bound * np.ones(3), -pos_bound * np.ones(3), -pos_bound * np.ones(3),
            [gripper_low]
        ]).astype(np.float32)

        obs_high = np.concatenate([
            joint_high, vel_bound * np.ones(7),
            pos_bound * np.ones(3), pos_bound * np.ones(3), pos_bound * np.ones(3),
            [gripper_high]
        ]).astype(np.float32)

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.render_mode = render_mode
        self.viewer = None

        self.home_qpos = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79])
        self.gripper_closed = False
        self.has_grasped_once = False
        self.prev_ee_to_obj = None
        self.prev_obj_to_target = None

    def _get_obs(self):
        joint_pos = self.data.qpos[:7].copy()
        joint_vel = self.data.qvel[:7].copy()
        ee_pos = self.data.site_xpos[self.site_id].copy()
        obj_pos = self.data.xpos[self.object_body_id].copy()
        gripper_state = np.array([self.data.qpos[7]])

        obs = np.concatenate([
            joint_pos, joint_vel, ee_pos, obj_pos, self.target_pos, gripper_state
        ]).astype(np.float32)
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:7] = self.home_qpos
        self.data.qvel[:] = 0
        self.data.ctrl[:7] = self.home_qpos
        self.data.ctrl[self.gripper_actuator] = 255

        obj_x = self.np_random.uniform(0.35, 0.55)
        obj_y = self.np_random.uniform(-0.15, 0.15)

        object_jnt_adr = self.model.body("object").jntadr[0]
        qpos_adr = self.model.jnt_qposadr[object_jnt_adr]
        self.data.qpos[qpos_adr:qpos_adr + 3] = [obj_x, obj_y, 0.05]
        self.data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]

        mujoco.mj_forward(self.model, self.data)

        self.current_step = 0
        self.gripper_closed = False
        self.has_grasped_once = False
        self.prev_ee_to_obj = None
        self.prev_obj_to_target = None

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        joint_deltas = action[:7] * self.max_delta
        gripper_cmd = action[7]

        target_qpos = self.data.qpos[:7] + joint_deltas
        self.data.ctrl[:7] = target_qpos

        self.gripper_closed = gripper_cmd > 0
        self.data.ctrl[self.gripper_actuator] = 0 if self.gripper_closed else 255

        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, terminated, info = self._compute_reward()

        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps

        if self.render_mode == "human" and self.viewer is not None:
            self.viewer.sync()

        return obs, reward, terminated, truncated, info

    def _compute_reward(self):
        ee_pos = self.data.site_xpos[self.site_id]
        obj_pos = self.data.xpos[self.object_body_id]

        ee_to_obj = np.linalg.norm(ee_pos - obj_pos)
        obj_to_target = np.linalg.norm(obj_pos[:2] - self.target_pos[:2])
        obj_height = obj_pos[2]

        reward = 0.0
        terminated = False
        info = {}

        reward -= 0.01

        if self.prev_ee_to_obj is not None:
            reward += 2.0 * (self.prev_ee_to_obj - ee_to_obj)
        self.prev_ee_to_obj = ee_to_obj

        is_lifted = obj_height > 0.08
        if self.gripper_closed and ee_to_obj < 0.05 and is_lifted:
            if not self.has_grasped_once:
                reward += 10.0
                self.has_grasped_once = True

            if self.prev_obj_to_target is not None:
                reward += 5.0 * (self.prev_obj_to_target - obj_to_target)
            self.prev_obj_to_target = obj_to_target

            if obj_to_target < 0.08 and obj_height < 0.05 and not self.gripper_closed:
                reward += 50.0
                terminated = True
                info["success"] = True

        if obj_pos[2] < -0.1 or np.linalg.norm(obj_pos[:2]) > 2.0:
            reward -= 10.0
            terminated = True
            info["success"] = False

        return reward, terminated, info

    def render(self):
        if self.render_mode == "human":
            import mujoco.viewer
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
