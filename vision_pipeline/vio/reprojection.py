import numpy as np


def reprojection_error(
    observed_point,
    projected_point
):
    """
    Compute reprojection error.

    Args:
        observed_point: measured feature position
        projected_point: predicted feature position

    Returns:
        pixel error
    """

    observed_point = np.array(observed_point)
    projected_point = np.array(projected_point)

    error = np.linalg.norm(
        observed_point - projected_point
    )

    return error
