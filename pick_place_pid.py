import sys
sys.path.append("src")

import mujoco
import mujoco.viewer
import numpy as np
import time
from kinematics import damped_least_squares_ik
from pid_controller import PIDController

MODEL_PATH = "mujoco_menagerie/franka_emika_panda/pick_scene_pid.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

site_id = model.site("attachment_site").id
object_body_id = model.body("object").id

GRIPPER_OPEN = 255
GRIPPER_CLOSED = 0
GRIPPER_ACTUATOR = 7

DT = model.opt.timestep

# PID gains per joint (tuned roughly; joints 1-4 heavier -> higher gains)
KP = [400, 400, 400, 400, 180, 100, 40]
KI = [10, 10, 10, 10, 5, 5, 2]
KD = [60, 60, 60, 60, 25, 18, 8]

pid = PIDController(KP, KI, KD, n_joints=7)


def set_gripper(width):
    data.ctrl[GRIPPER_ACTUATOR] = width


def move_arm_to_qpos_pid(target_qpos, max_steps=1500, tol=0.005, vel_tol=0.05,
                          viewer=None, settle_steps=150, n_waypoints=20):
    """
    Smoothly interpolate through n_waypoints intermediate targets between the
    current qpos and target_qpos (cubic ease-in/ease-out), so the PID never
    has to chase a large single jump. This avoids high accelerations that
    can shake a grasped object loose.
    """
    start_qpos = data.qpos[:7].copy()

    for wp in range(1, n_waypoints + 1):
        alpha = wp / n_waypoints
        # smoothstep (cubic ease-in/ease-out) instead of linear
        smooth_alpha = 3 * alpha**2 - 2 * alpha**3
        waypoint_qpos = start_qpos + smooth_alpha * (target_qpos - start_qpos)

        pid.reset()
        converged_count = 0
        steps_per_waypoint = max_steps // n_waypoints
        for step in range(steps_per_waypoint):
            current_qpos = data.qpos[:7]
            current_qvel = data.qvel[:7]

            torque = pid.compute(waypoint_qpos, current_qpos, current_qvel, DT)
            gravity_comp = data.qfrc_bias[:7]
            data.ctrl[:7] = torque + gravity_comp

            mujoco.mj_step(model, data)
            if viewer:
                viewer.sync()
                time.sleep(0.001)

            pos_err = np.linalg.norm(waypoint_qpos - data.qpos[:7])
            vel_norm = np.linalg.norm(data.qvel[:7])

            if pos_err < tol and vel_norm < vel_tol:
                converged_count += 1
                if converged_count > 5:
                    break
            else:
                converged_count = 0

    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()
            time.sleep(0.002)


def solve_ik_for(target_pos, seed_qpos=None):
    if seed_qpos is not None:
        data.qpos[:7] = seed_qpos
        mujoco.mj_forward(model, data)
    qpos_sol, err, n_iters, converged = damped_least_squares_ik(model, data, target_pos)
    return qpos_sol, err, converged


def get_object_pos():
    return data.xpos[object_body_id].copy()


def run_pick_and_place_pid(target_drop_pos, viewer=None):
    log = {"success": False, "steps": []}

    home_qpos = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79])
    data.qpos[:7] = home_qpos
    data.qvel[:] = 0
    set_gripper(GRIPPER_OPEN)
    mujoco.mj_forward(model, data)
    if viewer:
        viewer.sync()

    for _ in range(200):
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()
            time.sleep(0.002)

    obj_pos = get_object_pos()

    above_obj = obj_pos + np.array([0, 0, 0.15])
    qpos_sol, err, ok = solve_ik_for(above_obj, seed_qpos=home_qpos)
    move_arm_to_qpos_pid(qpos_sol, viewer=viewer)
    log["steps"].append(("approach", err, ok))

    obj_pos = get_object_pos()
    grasp_pos = obj_pos + np.array([0, 0, 0.02])
    qpos_sol, err, ok = solve_ik_for(grasp_pos, seed_qpos=qpos_sol)
    move_arm_to_qpos_pid(qpos_sol, viewer=viewer)
    log["steps"].append(("descend", err, ok))

    set_gripper(GRIPPER_CLOSED)
    for _ in range(150):
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()
            time.sleep(0.002)

    lift_pos = grasp_pos + np.array([0, 0, 0.2])
    qpos_sol, err, ok = solve_ik_for(lift_pos, seed_qpos=qpos_sol)
    move_arm_to_qpos_pid(qpos_sol, viewer=viewer)
    log["steps"].append(("lift", err, ok))

    site_id_dbg = model.site("attachment_site").id
    obj_after_lift = get_object_pos()
    grip_offset_after_lift = np.linalg.norm(obj_after_lift - data.site_xpos[site_id_dbg])
    log["grip_offset_after_lift"] = grip_offset_after_lift

    lift_height = get_object_pos()[2] - obj_pos[2]
    grasped = lift_height > 0.05

    above_target = target_drop_pos + np.array([0, 0, 0.15])
    qpos_sol, err, ok = solve_ik_for(above_target, seed_qpos=qpos_sol)
    move_arm_to_qpos_pid(qpos_sol, viewer=viewer)
    log["steps"].append(("transport", err, ok))

    obj_after_transport = get_object_pos()
    grip_offset_after_transport = np.linalg.norm(obj_after_transport - data.site_xpos[site_id_dbg])
    log["grip_offset_after_transport"] = grip_offset_after_transport

    release_pos = target_drop_pos + np.array([0, 0, 0.03])
    qpos_sol, err, ok = solve_ik_for(release_pos, seed_qpos=qpos_sol)
    move_arm_to_qpos_pid(qpos_sol, viewer=viewer)
    log["steps"].append(("place_descend", err, ok))

    set_gripper(GRIPPER_OPEN)
    for _ in range(150):
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()
            time.sleep(0.002)

    retreat_pos = release_pos + np.array([0, 0, 0.2])
    qpos_sol, err, ok = solve_ik_for(retreat_pos, seed_qpos=qpos_sol)
    move_arm_to_qpos_pid(qpos_sol, viewer=viewer)
    log["steps"].append(("retreat", err, ok))

    final_obj_pos = get_object_pos()
    dist_to_target = np.linalg.norm(final_obj_pos[:2] - target_drop_pos[:2])
    log["success"] = grasped and dist_to_target < 0.08
    log["final_dist"] = dist_to_target
    log["grasped"] = grasped

    return log


if __name__ == "__main__":
    target_drop = np.array([0.45, -0.3, 0.02])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        result = run_pick_and_place_pid(target_drop, viewer=viewer)
        print("\n--- PID Pipeline Result ---")
        for name, err, ok in result["steps"]:
            print(f"{name:15s} IK err={err*1000:.3f}mm converged={ok}")
        print(f"Grasped: {result['grasped']}")
        print(f"Success: {result['success']}  final_dist={result['final_dist']*1000:.1f}mm")

        print("\nViewer staying open — close window to exit.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)
