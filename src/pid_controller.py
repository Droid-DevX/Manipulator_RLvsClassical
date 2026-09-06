import numpy as np


class PIDController:
    """
    Simple per-joint PID controller.
    Computes torque = Kp*error + Ki*integral(error) + Kd*d(error)/dt
    """

    def __init__(self, kp, ki, kd, n_joints=7, integral_limit=10.0):
        self.kp = np.array(kp, dtype=float)
        self.ki = np.array(ki, dtype=float)
        self.kd = np.array(kd, dtype=float)
        self.n_joints = n_joints
        self.integral_limit = integral_limit

        self.integral = np.zeros(n_joints)
        self.prev_error = np.zeros(n_joints)

    def reset(self):
        self.integral[:] = 0
        self.prev_error[:] = 0

    def compute(self, target_qpos, current_qpos, current_qvel, dt):
        error = target_qpos - current_qpos

        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)

        derivative = -current_qvel  # d(error)/dt = d(target - current)/dt = -d(current)/dt (target fixed each step)

        torque = self.kp * error + self.ki * self.integral + self.kd * derivative
        return torque
