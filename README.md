# Vision-Free Collision-Aware Manipulation: RL vs Classical Control

Comparing classical Jacobian-IK+PID pick-and-place against a PPO-trained policy on a simulated Franka Panda in MuJoCo.

## Setup

```bash
python3 -m venv ~/mujoco_env
source ~/mujoco_env/bin/activate
pip install mujoco gymnasium stable-baselines3[extra] numpy scipy matplotlib mink

git clone https://github.com/google-deepmind/mujoco_menagerie.git
```

Note: `panda.xml` in the menagerie has been modified to add a named `attachment_site` inside the `hand` body, used as the end-effector reference for FK/IK.

## Progress
- Week 1: MuJoCo setup, FK verification
- Week 2: Jacobian-based IK, damped least-squares solver (`src/kinematics.py`) — 50/50 convergence, 0.032mm mean error
- Week 3: Classical pick-and-place pipeline (`test_pid.py`), 20-trial evaluation (`live_test_pid.py`) — 20/20 success rate, 27.1mm mean error, 0.07s mean completion time

- Week 4: Implemented reinforcement learning for the pick-and-place task and trained a PPO (Proximal Policy Optimization) algorithm to perform vision-free manipulation.
