import numpy as np


AIR_DENSITY = 1.225  # kg/m^3


def pressure_to_airspeed(
    total_pressure,
    static_pressure,
    rho=AIR_DENSITY
):
    """
    Convert pitot pressure to airspeed.

    Args:
        total_pressure: pitot total pressure
        static_pressure: atmospheric pressure
        rho: air density

    Returns:
        airspeed (m/s)
    """

    dp = total_pressure - static_pressure

    if dp < 0:
        dp = 0

    airspeed = np.sqrt((2.0 * dp) / rho)

    return airspeed
