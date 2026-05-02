import pandas as pd
import numpy as np


def load_imu_data(path):
    """
    Load EuRoC IMU data.

    Args:
        path (str): path to imu0/data.csv

    Returns:
        timestamps (np.array): seconds
        gyro (np.array): angular velocity (rad/s)
        accel (np.array): linear acceleration (m/s^2)
    """
    df = pd.read_csv(path)

    # EuRoC timestamps are in nanoseconds → convert to seconds
    timestamps = df.iloc[:, 0].values.astype(np.float64) * 1e-9

    # Gyroscope: wx, wy, wz
    gyro = df.iloc[:, 1:4].values.astype(np.float64)

    # Accelerometer: ax, ay, az
    accel = df.iloc[:, 4:7].values.astype(np.float64)

    return timestamps, gyro, accel


def get_imu_segment(timestamps, gyro, accel, t0, t1):
    """
    Get IMU measurements between two timestamps.

    Args:
        timestamps: IMU timestamps
        gyro: gyro data
        accel: accel data
        t0: start time
        t1: end time

    Returns:
        segment timestamps, gyro, accel
    """
    mask = (timestamps >= t0) & (timestamps <= t1)

    return (
        timestamps[mask],
        gyro[mask],
        accel[mask]
    )
