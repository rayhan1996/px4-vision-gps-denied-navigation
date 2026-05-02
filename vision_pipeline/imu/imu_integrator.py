import numpy as np


def skew(w):
    """
    Convert a vector to a skew-symmetric matrix.
    """
    return np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])


def integrate_imu(timestamps, gyro, accel, p, v, R):
    """
    Integrate IMU data over time.

    Args:
        timestamps: array of time (seconds)
        gyro: angular velocity (rad/s)
        accel: linear acceleration (m/s^2)
        p: position (3,)
        v: velocity (3,)
        R: rotation matrix (3x3)

    Returns:
        updated p, v, R
    """

    if len(timestamps) < 2:
        return p, v, R

    # Gravity (world frame)
    g = np.array([0.0, 0.0, -9.81])

    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i - 1]

        # Safety check
        if dt <= 0:
            continue

        omega = gyro[i]
        a = accel[i]

        # --- 1. Update rotation ---
        dR = np.eye(3) + skew(omega * dt)
        R = R @ dR

        # Optional: re-orthonormalize rotation matrix
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt

        # --- 2. Transform acceleration to world frame ---
        a_world = R @ a + g

        # --- 3. Update velocity ---
        v = v + a_world * dt

        # --- 4. Update position ---
        p = p + v * dt + 0.5 * a_world * dt**2

    return p, v, R
