import time
import numpy as np

from vision_pipeline.fusion.fusion_manager import (
    FusionManager
)


def generate_fake_imu(n_samples=50):
    """
    Fake IMU stream.
    """

    timestamps = np.linspace(
        0,
        0.5,
        n_samples
    )

    gyro = np.zeros((n_samples, 3))
    accel = np.zeros((n_samples, 3))

    # Small yaw rotation
    gyro[:, 2] = 0.02

    # Small forward acceleration
    accel[:, 0] = 0.1

    # Body-frame accel (no fake gravity)
    accel[:, 2] = 0.0

    return timestamps, gyro, accel


def generate_fake_vo(frame_id):
    """
    Fake visual odometry trajectory.
    """

    position = np.array([
        0.05 * frame_id,
        0.0,
        0.0
    ])

    yaw = 0.01 * frame_id

    c = np.cos(yaw)
    s = np.sin(yaw)

    rotation = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

    return position, rotation


def generate_fake_altitude(frame_id):
    """
    Fake altitude profile.
    """

    return 120.0 + 0.1 * frame_id


def print_state(frame_id, state):

    print("=" * 60)
    print(f"Frame {frame_id}")

    print(
        f"Position: {np.round(state.position, 3)}"
    )

    print(
        f"Velocity: {np.round(state.velocity, 3)}"
    )

    print(
        f"Altitude: {round(state.altitude, 3)}"
    )

    print(
        f"Wind: {np.round(state.wind_vector, 3)}"
    )

    print("=" * 60)


def check_rotation_matrix(R):

    should_be_I = R.T @ R

    return np.allclose(
        should_be_I,
        np.eye(3),
        atol=1e-3
    )


def main():

    fusion = FusionManager()

    n_frames = 20

    print("\nStarting fusion integration test...\n")

    for frame_id in range(n_frames):

        imu_t, gyro, accel = (
            generate_fake_imu()
        )

        vo_position, vo_rotation = (
            generate_fake_vo(frame_id)
        )

        altitude = (
            generate_fake_altitude(frame_id)
        )

        airspeed = 15.0

        wind_vector = np.array([
            1.0,
            0.2,
            0.0
        ])

        # Prediction
        fusion.predict_from_imu(
            imu_t,
            gyro,
            accel
        )

        # Correction
        fusion.correct_with_vo(
            vo_position,
            vo_rotation
        )

        fusion.correct_with_altitude(
            altitude
        )

        fusion.correct_with_airdata(
            airspeed,
            wind_vector
        )

        state = fusion.get_state()

        print_state(frame_id, state)

        # Better sanity checks
        assert np.all(
            np.isfinite(state.position)
        ), "Invalid position"

        assert np.all(
            np.isfinite(state.velocity)
        ), "Invalid velocity"

        assert np.isfinite(
            state.altitude
        ), "Invalid altitude"

        assert check_rotation_matrix(
            state.rotation
        ), "Invalid rotation matrix"

        time.sleep(0.1)

    print(
        "\nFusion integration test completed successfully.\n"
    )


if __name__ == "__main__":
    main()
