import numpy as np

def ang_in_mpi_ppi(angle):
    """
    Convert the angle to the range [-pi, pi).
    Args:
        angle (float): angle in radians.
    Returns:
        float: equivalent angle in [-pi, pi).
    """

    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle