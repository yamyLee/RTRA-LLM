"""Canonical numerical settings for the experiments reported in the paper.

Values are expressed in SI units unless a name says otherwise.  The module is
the single source of truth for runtime defaults; ``config/simulation_config.yaml``
mirrors these values for inspection and documentation.
"""

# Simulation grid
SIMULATION_TIME_S = 450.0
SIMULATION_DT_S = 0.1
RANDOM_SEED = 42
PAPER_RANDOM_SEEDS = (42, 43, 44, 45, 46)

# Own ship and target ships
OWN_SHIP_LENGTH_M = 30.0
OWN_SHIP_BEAM_M = 16.0
OWN_SHIP_SPEED_MPS = 43.3
OWN_SHIP_CPA_MULTIPLIER = 2.0
TARGET_SHIP_LENGTH_M = 80.0
TARGET_SHIP_BEAM_M = 30.0
TARGET_SHIP_SPEED_MPS = 18.52
TARGET_SHIP_CPA_MULTIPLIER = 1.0

# Waypoint guidance and local reactive avoidance
WAYPOINT_SWITCH_DISTANCE_M = 200.0
GUIDANCE_RHO_M = 2200.0
AVOIDANCE_DISTANCE_INNER_M = 600.0
AVOIDANCE_DISTANCE_OUTER_M = 1200.0
AVOIDANCE_BEARING_SIGMA_DEG = 80.0
AVOIDANCE_HEADING_GAIN = 2.0

# Traditional VO baseline.  The paper does not specify a separate VO horizon;
# use the upper TCPA risk horizon and the inner local-avoidance safety domain.
VO_TIME_HORIZON_S = 360.0
VO_SAFETY_RADIUS_M = 600.0
VO_HEADING_SEARCH_DEG = 90.0
VO_HEADING_STEP_DEG = 1.0

# First-order Nomoto-like vessel model
DYNAMICS_K_PSI = 0.01
DYNAMICS_T_PSI_S = 30.0
DYNAMICS_K_SPEED = 1.0
DYNAMICS_T_SPEED_S = 50.0
DYNAMICS_BIAS_TIME_FACTOR = 20.0
DYNAMICS_BIAS_NOISE_STD = 0.5
DYNAMICS_RATE_NOISE_STD = 0.0

# Yaw controller and actuator
CONTROL_KP_YAW = 100.0
CONTROL_KD_YAW = -500.0
CONTROL_KI_YAW = 0.0
ACTUATOR_SATURATION = 20.0

# Z-shaped risk membership functions: (lower, upper), in metres/seconds.
RISK_DCPA_LOWER_M = 443.0
RISK_DCPA_UPPER_M = 926.0
RISK_TCPA_LOWER_S = 180.0
RISK_TCPA_UPPER_S = 360.0
RISK_DISTANCE_LOWER_M = 148.16
RISK_DISTANCE_UPPER_M = 463.0

# RTRA-LLM experiment settings
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 500
LLM_RISK_THRESHOLD = 0.30
LLM_FIXED_INTERVAL_STEPS = 250
LLM_HIGH_LEVEL_UPDATE_STEPS = 20

PAPER_MODEL_BY_PROVIDER = {
    "qwen": "qwen3.5-plus",
    "deepseek": "deepseek-v3.2",
    "minimax": "MiniMax-M2.5",
}

PAPER_MODEL_DISPLAY_BY_PROVIDER = {
    "qwen": "Qwen3.5-Plus",
    "deepseek": "DeepSeek-V3.2",
    "minimax": "MiniMax-M2.5",
}

# The manuscript specifies a shared low-level planner across compared methods.
# VO is the named traditional baseline, so paper experiment runners use it for
# both the baseline and LLM conditions.
PAPER_LOW_LEVEL_PLANNER = "vo"
