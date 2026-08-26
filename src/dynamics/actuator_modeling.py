import numpy as np
from src.config.paper_parameters import ACTUATOR_SATURATION


def actuator_modeling(tau_c, sat_amp_s=ACTUATOR_SATURATION):
    tau_ac = tau_c

    if abs(tau_c) > sat_amp_s:
        tau_ac = np.sign(tau_c) * sat_amp_s

    return tau_ac
