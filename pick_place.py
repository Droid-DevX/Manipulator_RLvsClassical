import sys
sys.path.append("src")

import mujoco
import mujoco.viewer
import numpy as np
import time
from kinematics import damped_least_squares_ik

MODEL_PATH = "mujoco_menagerie/franka_emika_panda/pick_scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

site_id = model.site("attachment_site").id
object_body_id = model.body("object").id

GRIPPER_OPEN = 255
GRIPPER_CLOSED = 0
GRIPPER_ACTUATOR = 7

SLEEP = 0.004  # slow enough to watch


def set_gripper(width):
    data.ctrl[GRIPPER_ACTUATOR] = width


def move_arm_to_qpos(target_qpos, steps=400, settle_steps=150, viewer=None):
    start_ctrl = data.ctrl[:7].copy()
    for i in range(steps):
        alpha = (i + 1) / steps
        data.ctrl[:7] = (1 - alpha) * start_ctrl + alpha * target_qpos
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()
            time.sleep(SLEEP)
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()
            time.sleep(SLEEP)


def solve_ik_for(target_pos, seed_qpos=None):
    if seed_qpos is not None:
        data.qpos[:7] = seed_qpos
    qpos_sol, err, n_iters, converged = damped_least_squares_ik(model, data, target_pos)
    return qpos_sol, err, converged


def get_object_pos():
    return data.xpos[object_body_id].copy()


def run_pick_and_place(target_drop_pos, viewer=None):
    log = {"success": False, "steps": []}

    home_qpos = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79])
    data.qpos[:7] = home_qpos
    data.ctrl[:7] = home_qpos
    set_gripper(GRIPPER_OPEN)
    mujoco.mj_forward(model, data)
    if viewer:
        viewer.sync()

    # Let the box settle under gravity FIRST, before planning any waypoints
    for _ in range(200):
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()
            time.sleep(SLEEP)

    obj_pos = get_object_pos()
    print("Object settled at:", obj_pos)

    above_obj = obj_pos + np.array([0, 0, 0.15])
    qpos_sol, err, ok = solve_ik_for(above_obj, seed_qpos=home_qpos)
    move_arm_to_qpos(qpos_sol, viewer=viewer)
    log["steps"].append(("approach", err, ok))

    # Re-read object position again right before descending (in case it moved)
    obj_pos = get_object_pos()
    grasp_pos = obj_pos + np.array([0, 0, 0.02])
    qpos_sol, err, ok = solve_ik_for(grasp_pos, seed_qpos=qpos_sol)
    move_arm_to_qpos(qpos_sol, viewer=viewer)
    log["steps"].append(("descend", err, ok))

    set_gripper(GRIPPER_CLOSED)
    for _ in range(150):
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()
            time.sleep(SLEEP)

    obj_pos_after_grasp = get_object_pos()
    print("Object pos after grasp attempt:", obj_pos_after_grasp)

    lift_pos = grasp_pos + np.array([0, 0, 0.2])
    qpos_sol, err, ok = solve_ik_for(lift_pos, seed_qpos=qpos_sol)
    move_arm_to_qpos(qpos_sol, viewer=viewer)
    log["steps"].append(("lift", err, ok))

    obj_pos_after_lift = get_object_pos()
    lift_height = obj_pos_after_lift[2] - obj_pos[2]
    print(f"Object height after lift: {obj_pos_after_lift[2]:.3f}m  (delta={lift_height*1000:.1f}mm)")
    grasped = lift_height > 0.05  # object rose at least 5cm -> actually grasped

    above_target = target_drop_pos + np.array([0, 0, 0.15])
    qpos_sol, err, ok = solve_ik_for(above_target, seed_qpos=qpos_sol)
    move_arm_to_qpos(qpos_sol, viewer=viewer)
    log["steps"].append(("transport", err, ok))

    release_pos = target_drop_pos + np.array([0, 0, 0.03])
    qpos_sol, err, ok = solve_ik_for(release_pos, seed_qpos=qpos_sol)
    move_arm_to_qpos(qpos_sol, viewer=viewer)
    log["steps"].append(("place_descend", err, ok))

    set_gripper(GRIPPER_OPEN)
    for _ in range(150):
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()
            time.sleep(SLEEP)

    retreat_pos = release_pos + np.array([0, 0, 0.2])
    qpos_sol, err, ok = solve_ik_for(retreat_pos, seed_qpos=qpos_sol)
    move_arm_to_qpos(qpos_sol, viewer=viewer)
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
        result = run_pick_and_place(target_drop, viewer=viewer)
        print("\n--- Result ---")
        for name, err, ok in result["steps"]:
            print(f"{name:15s} IK err={err*1000:.3f}mm converged={ok}")
        print(f"Grasped: {result['grasped']}")
        print(f"Success: {result['success']}  final_dist={result['final_dist']*1000:.1f}mm")

        print("\nViewer staying open — close window to exit.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)
