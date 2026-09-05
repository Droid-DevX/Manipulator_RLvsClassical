import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/panda.xml")
data = mujoco.MjData(model)

site_id = model.site("attachment_site").id

# Test 1: all-zero joint configuration
data.qpos[:] = 0
mujoco.mj_forward(model, data)
print("Zero config EE pos:", data.site_xpos[site_id])

# Test 2: a known non-zero configuration
test_qpos = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79, 0.0, 0.0])
data.qpos[:len(test_qpos)] = test_qpos
mujoco.mj_forward(model, data)
print("Test config EE pos:", data.site_xpos[site_id])

# Test 3: sanity check against viewer — set this pose and visually confirm
print("\nJoint names and order:")
for i in range(model.njnt):
    print(i, model.joint(i).name)
