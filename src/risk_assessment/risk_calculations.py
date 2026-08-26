import numpy as np
from src.utils.zmf import zmf
from src.config.paper_parameters import (
    RISK_DCPA_LOWER_M,
    RISK_DCPA_UPPER_M,
    RISK_DISTANCE_LOWER_M,
    RISK_DISTANCE_UPPER_M,
    RISK_TCPA_LOWER_S,
    RISK_TCPA_UPPER_S,
)


def risk_calculations(dcpa, tcpa, distance_ob, v_rel):
    """
    Calculate collision risk based on DCPA, TCPA, and current distance.

    Units are intentionally the same as the simulation state:
    DCPA and distance are in meters, TCPA is in seconds. Negative TCPA
    means the closest-approach point is already behind the own ship and
    should not increase future collision risk.

    Parameters:
    dcpa (array-like): Distance at Closest Point of Approach
    tcpa (array-like): Time to Closest Point of Approach
    distance_ob (array-like): Distance to obstacles
    v_rel (array-like): Relative velocity (not used in this function)

    Returns:
    numpy.array: Calculated risk for each obstacle
    """
    dcpa = np.asarray(dcpa)
    tcpa = np.asarray(tcpa)
    distance_ob = np.asarray(distance_ob)
    
    a_dcpa = RISK_DCPA_LOWER_M
    b_dcpa = RISK_DCPA_UPPER_M

    a_tcpa = RISK_TCPA_LOWER_S
    b_tcpa = RISK_TCPA_UPPER_S

    a_dist = RISK_DISTANCE_LOWER_M
    b_dist = RISK_DISTANCE_UPPER_M

    tcpa_for_risk = np.where(tcpa < 0, np.inf, tcpa)
 
    risk_dcpa = zmf(np.abs(dcpa), a_dcpa, b_dcpa)
    risk_tcpa = zmf(tcpa_for_risk, a_tcpa, b_tcpa)
    risk_dist = zmf(distance_ob, a_dist, b_dist)

    risk = (risk_dcpa + risk_tcpa + risk_dist) / 3

    return risk
