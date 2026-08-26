import numpy as np
from src.config.paper_parameters import (
    DYNAMICS_BIAS_NOISE_STD,
    DYNAMICS_BIAS_TIME_FACTOR,
    DYNAMICS_K_PSI,
    DYNAMICS_K_SPEED,
    DYNAMICS_RATE_NOISE_STD,
    DYNAMICS_T_PSI_S,
    DYNAMICS_T_SPEED_S,
)


def vessel_dynamics(x_0, inputs):
    """
    Calculate the vessel dynamics.

    Parameters:
    x_0 (numpy.array): Initial state [x, y, psi, r, b, u]
    inputs (numpy.array): Control inputs [tau_c, u_c]

    Returns:
    numpy.array: State derivatives [x_dot, y_dot, psi_dot, r_dot, b_dot, u_dot]
    """
    x, y, psi, r, b, u = x_0
    tau_c, u_c = inputs

    k_psi = DYNAMICS_K_PSI
    t_psi = DYNAMICS_T_PSI_S

    k_v = DYNAMICS_K_SPEED
    t_v = DYNAMICS_T_SPEED_S

    t_b = DYNAMICS_BIAS_TIME_FACTOR * t_psi

    x_dot = u_c * np.cos(psi)
    y_dot = u_c * np.sin(psi)
    psi_dot = r

    w_r = (
        DYNAMICS_RATE_NOISE_STD * np.random.randn()
        if DYNAMICS_RATE_NOISE_STD
        else 0.0
    )
    w_b = DYNAMICS_BIAS_NOISE_STD * np.random.randn()

    # Nomoto model
    r_dot = -(1/t_psi) * r + (1/t_psi) * k_psi * (tau_c - b) + w_r
    b_dot = -(1/t_b) * b + w_b
    u_dot = -(1/t_v) * u + (1/t_v) * k_v * u_c

    x_dot = np.array([x_dot, y_dot, psi_dot, r_dot, b_dot, u_dot])

    return x_dot
