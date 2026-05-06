import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from vision_pipeline.imu.imu_loader import load_imu_data
from vision_pipeline.imu.attitude_estimator import ComplementaryFilter

imu_path = "data/euroc/MH01/imu0/data.csv"

timestamps, gyro, accel = load_imu_data(imu_path)

filter = ComplementaryFilter(alpha=0.98)

for i in range(1, len(timestamps)):
    dt = timestamps[i] - timestamps[i-1]

    R = filter.update(gyro[i], accel[i], dt)

    roll = np.arctan2(R[2,1], R[2,2])
    pitch = np.arcsin(-R[2,0])
    yaw = np.arctan2(R[1,0], R[0,0])

    if i % 100 == 0:
        print(f"RPY: {roll:.2f}, {pitch:.2f}, {yaw:.2f}")

print("✅ Orientation test done")
