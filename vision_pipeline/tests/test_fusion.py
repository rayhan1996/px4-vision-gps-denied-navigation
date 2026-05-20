import time
import numpy as np

from vision_pipeline.fusion.fusion_manager import (
    FusionManager
)

from vision_pipeline.fusion.state import (
    NavigationState
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

    # Gravity-compensated body accel
    accel[:, 2] = 9.81

    return timestamps, gyro, accel


def generate_fake_vo():
    """
    Fake visual odometry delta.
    """

    position = np.array([
        0.05,
        0.0,
        0.0
    ])

    rotation = np.eye(3)

    return position, rotation


def generate_fake_altitude():
    """
    Fake altitude estimate.
    """

    return 120.0


def print_state(
    frame_id,
    state
):
    """
    Pretty print navigation state.
    """

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


def main():

    fusion = FusionManager()

    n_frames = 20

    print("\nStarting fusion integration test...\n")

    for frame_id in range(n_frames):

        # ---------------------------------
        # Fake sensor data
        # ---------------------------------
        imu_t, gyro, accel = (
            generate_fake_imu()
        )

        vo_position, vo_rotation = (
            generate_fake_vo()
        )

        altitude = (
            generate_fake_altitude()
        )

        airspeed = 15.0

        wind_vector = np.array([
            1.0,
            0.2,
            0.0
        ])

        # ---------------------------------
        # Prediction
        # ---------------------------------
        fusion.predict_from_imu(
            imu_t,
            gyro,
            accel
        )

        # ---------------------------------
        # Corrections
        # ---------------------------------
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

        # ---------------------------------
        # Read state
        # ---------------------------------
        state = fusion.get_state()

        print_state(
            frame_id,
            state
        )

        # ---------------------------------
        # Sanity checks
        # ---------------------------------
        assert not np.any(
            np.isnan(state.position)
        ), "NaN in position"

        assert not np.any(
            np.isnan(state.velocity)
        ), "NaN in velocity"

        assert not np.isnan(
            state.altitude
        ), "NaN in altitude"

        time.sleep(0.1)

    print(
        "\nFusion integration test completed successfully.\n"
    )


if __name__ == "__main__":
    main()
