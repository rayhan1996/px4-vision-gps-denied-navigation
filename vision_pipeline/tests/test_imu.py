import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from vision_pipeline.imu.imu_loader import load_imu_data
from vision_pipeline.imu.imu_integrator import integrate_imu

# 🔥 Adjust this path
imu_path = "data/euroc/MH01/imu0/data.csv"

timestamps, gyro, accel = load_imu_data(imu_path)

p = np.zeros(3)
v = np.zeros(3)
R = np.eye(3)

trajectory = []

for i in range(1, len(timestamps)):
    ts = timestamps[i-1:i+1]
    g = gyro[i-1:i+1]
    a = accel[i-1:i+1]

    p, v, R = integrate_imu(ts, g, a, p, v, R)
    trajectory.append(p.copy())

    if i % 100 == 0:
        print("IMU Position:", p)

np.save("imu_trajectory.npy", np.array(trajectory))
print("✅ IMU test done")
