from vision_pipeline.fusion.state import (
    NavigationState
)

from vision_pipeline.fusion.prediction import (
    StatePredictor
)

from vision_pipeline.fusion.correction import (
    StateCorrector
)


class FusionManager:
    """
    Main sensor fusion manager.

    Combines:
    - IMU prediction
    - VO correction
    - altitude correction
    - airspeed correction
    """

    def __init__(self):

        self.state = NavigationState()

        self.predictor = StatePredictor()

        self.corrector = StateCorrector()

    # ==========================================
    # Prediction Step
    # ==========================================

    def predict_imu(
        self,
        timestamps,
        gyro,
        accel
    ):

        self.state = self.predictor.predict(
            self.state,
            timestamps,
            gyro,
            accel
        )

        return self.state

    # ==========================================
    # VO Correction
    # ==========================================

    def correct_vo(
        self,
        R_vo,
        t_vo
    ):

        self.state = self.corrector.correct_vo(
            self.state,
            R_vo,
            t_vo
        )

        return self.state

    # ==========================================
    # Altitude Correction
    # ==========================================

    def correct_altitude(
        self,
        altitude
    ):

        self.state = self.corrector.correct_altitude(
            self.state,
            altitude
        )

        return self.state

    # ==========================================
    # Airspeed Correction
    # ==========================================

    def correct_airspeed(
        self,
        airspeed
    ):

        self.state = self.corrector.correct_airspeed(
            self.state,
            airspeed
        )

        return self.state

    # ==========================================
    # Wind Correction
    # ==========================================

    def correct_wind(
        self,
        wind
    ):

        self.state = self.corrector.correct_wind(
            self.state,
            wind
        )

        return self.state
