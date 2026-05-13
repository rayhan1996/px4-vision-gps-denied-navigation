from vision_pipeline.airdata.pitot_model import (
    pressure_to_airspeed
)


class AirspeedEstimator:

    def __init__(self):

        self.airspeed = 0.0

    def update(
        self,
        total_pressure,
        static_pressure
    ):

        self.airspeed = pressure_to_airspeed(
            total_pressure,
            static_pressure
        )

        return self.airspeed
