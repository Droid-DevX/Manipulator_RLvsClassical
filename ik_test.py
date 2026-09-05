import sys
sys.path.append("src")

import mujoco
import numpy as np
from kinematics import damped_least_squares_ik, forward_kinematics

np.random.seed(42)

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/panda.xml")
data = mujoco.MjData(model)

X_RANGE = (0.3, 0.6)
Y_RANGE = (-0.3, 0.3)
Z_RANGE = (0.2, 0.6)

N_TARGETS = 50
results = []

for i in range(N_TARGETS):
    target = np.array([
        np.random.uniform(*X_RANGE),
        np.random.uniform(*Y_RANGE),
        np.random.uniform(*Z_RANGE),
    ])

    data.qpos[:] = 0
    data.qpos[:7] = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79])

    qpos_sol, err, n_iters, converged = damped_least_squares_ik(
        model, data, target
    )
    results.append((i, target, err, n_iters, converged))

    status = "OK " if converged else "FAIL"
    print(f"[{status}] target={target.round(3)}  err={err:.6f}  iters={n_iters}")

n_converged = sum(1 for r in results if r[4])
mean_iters = np.mean([r[3] for r in results if r[4]]) if n_converged > 0 else float("nan")
mean_err = np.mean([r[2] for r in results])

print(f"\nConverged: {n_converged}/{N_TARGETS}")
print(f"Mean iterations (converged only): {mean_iters:.1f}")
print(f"Mean final position error: {mean_err*1000:.3f} mm")
