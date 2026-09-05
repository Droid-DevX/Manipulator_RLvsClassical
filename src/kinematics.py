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
    site_id = model.site(site_name).id

    for it in range(max_iters):
        mujoco.mj_forward(model, data)
        ee_pos = data.site_xpos[site_id]
        error = target_pos - ee_pos
        err_norm = np.linalg.norm(error)

        if err_norm < tol:
            return data.qpos[:n_joints].copy(), err_norm, it, True

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = jacp[:, :n_joints]

        JJt = J @ J.T
        lam_sq = damping ** 2
        dq = J.T @ np.linalg.solve(JJt + lam_sq * np.eye(3), error)

        data.qpos[:n_joints] += step_scale * dq

    mujoco.mj_forward(model, data)
    final_err = np.linalg.norm(target_pos - data.site_xpos[site_id])
    return data.qpos[:n_joints].copy(), final_err, max_iters, False
