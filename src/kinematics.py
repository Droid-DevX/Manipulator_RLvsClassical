import mujoco
import numpy as np


def forward_kinematics(model, data, qpos, site_name="attachment_site"):
    site_id = model.site(site_name).id
    data.qpos[:len(qpos)] = qpos
    mujoco.mj_forward(model, data)
    return data.site_xpos[site_id].copy()


def get_jacobian(model, data, site_name="attachment_site"):
    site_id = model.site(site_name).id
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return jacp


def damped_least_squares_ik(
    model, data, target_pos,
    site_name="attachment_site",
    n_joints=7,
    damping=0.05,
    max_iters=200,
    tol=1e-4,
    step_scale=1.0,
):
    """
    Iteratively solve for qpos such that the EE site reaches target_pos.
    Solves on a SCRATCH copy of data, so the live simulation (and any
    grasped object held via contact) is never disturbed by the solver.
    Returns (qpos, final_error, n_iters, converged).
    """
    site_id = model.site(site_name).id

    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos[:]
    scratch.qvel[:] = 0

    for it in range(max_iters):
        mujoco.mj_forward(model, scratch)
        ee_pos = scratch.site_xpos[site_id]
        error = target_pos - ee_pos
        err_norm = np.linalg.norm(error)

        if err_norm < tol:
            return scratch.qpos[:n_joints].copy(), err_norm, it, True

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, scratch, jacp, jacr, site_id)
        J = jacp[:, :n_joints]

        JJt = J @ J.T
        lam_sq = damping ** 2
        dq = J.T @ np.linalg.solve(JJt + lam_sq * np.eye(3), error)

        scratch.qpos[:n_joints] += step_scale * dq

    mujoco.mj_forward(model, scratch)
    final_err = np.linalg.norm(target_pos - scratch.site_xpos[site_id])
    return scratch.qpos[:n_joints].copy(), final_err, max_iters, False
