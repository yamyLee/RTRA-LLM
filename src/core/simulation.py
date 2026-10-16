import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from src.navigation.planning import waypoint_selection, planning
from src.dynamics.vessel_dynamics import vessel_dynamics
from src.core.integration import integration
from src.navigation.encounters import relative_metrics, wrap_angle
from src.core.experiment_metrics import trajectory_metrics
from src.core.run_archive import save_run
from src.core.decision_delivery import DecisionDelivery
from src.navigation.obstacle_sim import obstacle_sim
from src.dynamics.controller import controller
from src.dynamics.actuator_modeling import actuator_modeling
from src.risk_assessment.risk_calculations import risk_calculations
from src.navigation.reactive_avoidance import reactive_avoidance
from src.navigation.velocity_obstacle import plan_velocity_obstacle
from src.visualization.animate import animate_step
from src.utils.imazu_cases import nautical_to_meters, get_obstacle_data
from src.config.paper_parameters import (
    ACTUATOR_SATURATION,
    LLM_FIXED_INTERVAL_STEPS,
    LLM_HIGH_LEVEL_UPDATE_STEPS,
    LLM_RISK_THRESHOLD,
    PAPER_LOW_LEVEL_PLANNER,
    OWN_SHIP_BEAM_M,
    OWN_SHIP_CPA_MULTIPLIER,
    OWN_SHIP_LENGTH_M,
    OWN_SHIP_SPEED_MPS,
    SIMULATION_DT_S,
    SIMULATION_TIME_S,
    RANDOM_SEED,
    TARGET_SHIP_BEAM_M,
    TARGET_SHIP_CPA_MULTIPLIER,
    TARGET_SHIP_LENGTH_M,
)
# Optional LLM imports
try:
    from src.decision_making.rtra_llm_supervisor import Action, RiskTriggeredLLMSupervisor
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    RiskTriggeredLLMSupervisor = None

METERS_TO_NMI = 1 / 1852

def save_figure_with_type(fig, base_path, file_type, case_number, dpi=None):
    """Save figure with proper directory structure based on file type."""
    # Create directory structure
    img_dir = Path("./img")
    file_type_dir = img_dir / file_type
    file_type_dir.mkdir(parents=True, exist_ok=True)

    # Save figure
    if file_type == 'eps':
        filepath = file_type_dir / f"{base_path}_{case_number}.eps"
        fig.savefig(filepath, format='eps', bbox_inches='tight')
    elif file_type == 'png':
        filepath = file_type_dir / f"{base_path}_{case_number}.png"
        fig.savefig(filepath, bbox_inches='tight', dpi=dpi)
    elif file_type == 'gif':
        filepath = file_type_dir / f"{base_path}_{case_number}.gif"
        # For GIF, we need to use the animation writer
        # This function is called outside the animation context
        # So we'll just return the path
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    return filepath

def save_gif_with_type(output_path, file_type, case_number):
    """Save GIF with proper directory structure."""
    # Create directory structure
    img_dir = Path("./img")
    file_type_dir = img_dir / file_type
    file_type_dir.mkdir(parents=True, exist_ok=True)

    # Extract base name and create new path
    base_name = os.path.basename(output_path)
    new_path = file_type_dir / base_name

    # If the file exists, move it
    if os.path.exists(output_path):
        os.makedirs(file_type_dir, exist_ok=True)
        os.rename(output_path, new_path)

    return new_path

def load_env_file():
    """Load environment variables from .env and config/api_keys.json if they exist."""
    env_file = Path(__file__).parent.parent.parent / '.env'

    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value

    api_keys_file = Path(__file__).parent.parent.parent / 'config' / 'api_keys.json'
    if not api_keys_file.exists():
        return

    try:
        with open(api_keys_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as exc:
        print(f"Warning: Could not load config/api_keys.json: {exc}")
        return

    if not isinstance(config, dict):
        return

    provider_hint = config.get("llm_provider") or config.get("provider") or config.get("default_provider")
    if provider_hint:
        os.environ["LLM_PROVIDER"] = str(provider_hint)

    for key, value in config.items():
        if key.isupper() and not isinstance(value, dict):
            os.environ[key] = str(value)

    field_map = {
        "api_key": "API_KEY",
        "key": "API_KEY",
        "model": "MODEL",
        "base_url": "BASE_URL",
        "url": "BASE_URL",
        "temperature": "TEMPERATURE",
        "max_tokens": "MAX_TOKENS",
    }
    provider_entries = list(config.items())
    for section_name in ("providers", "llm_providers", "api_keys"):
        section = config.get(section_name)
        if isinstance(section, dict):
            provider_entries.extend(section.items())

    for provider_name, provider_config in provider_entries:
        if provider_name in {"providers", "llm_providers", "api_keys"}:
            continue
        prefix = provider_name.upper()
        if isinstance(provider_config, str):
            os.environ[f"{prefix}_API_KEY"] = provider_config
            continue
        if not isinstance(provider_config, dict):
            continue
        for source_key, env_suffix in field_map.items():
            if source_key in provider_config and provider_config[source_key] is not None:
                os.environ[f"{prefix}_{env_suffix}"] = str(provider_config[source_key])

def should_show_debug():
    """Check if debug output is enabled."""
    return os.getenv("SHOW_LLM_DEBUG", "false").lower() == "true"

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Marine Vehicle Simulation')
    parser.add_argument('--case_number', type=int, default=1, help='Simulation case number')
    parser.add_argument('--all_cases', action='store_true', help='Run all supported Imazu cases')
    parser.add_argument('--sim_time', type=float, default=SIMULATION_TIME_S, help='Simulation time in seconds')
    parser.add_argument('--dt', type=float, default=SIMULATION_DT_S, help='Time step size')
    parser.add_argument('--no_animation', action='store_true', help='Disable animation')
    parser.add_argument('--output_dir', type=str, default='img/', help='Output directory for results')
    parser.add_argument('--llm', type=int, default=0, help='Use LLM for decision making (0=off, 1=on)')
    parser.add_argument('--llm_provider', type=str, default=None,
                       help='OpenAI-compatible LLM provider to use (for example: openai, zhipu, qwen, deepseek, kimi). If not specified, uses LLM_PROVIDER from .env')
    parser.add_argument('--compare', action='store_true',
                       help='Run comparison between LLM and baseline simulation')
    parser.add_argument('--llm_trigger_mode', choices=['risk', 'fixed', 'always', 'high_level_always'], default='risk',
                       help='LLM trigger strategy')
    parser.add_argument('--llm_risk_threshold', type=float, default=LLM_RISK_THRESHOLD,
                       help='Risk threshold q for triggering LLM supervision')
    parser.add_argument('--llm_fixed_interval', type=int, default=LLM_FIXED_INTERVAL_STEPS,
                       help='Fixed trigger interval in simulation steps when --llm_trigger_mode fixed is used')
    parser.add_argument('--llm_high_level_update_steps', type=int, default=LLM_HIGH_LEVEL_UPDATE_STEPS,
                       help='High-level decision period for the w/o Risk Trigger ablation')
    parser.add_argument('--low_level_planner', choices=['reactive', 'vo'], default=PAPER_LOW_LEVEL_PLANNER,
                       help='Low-level planner used by the paper experiments')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                       help='Random seed for the stochastic vessel bias disturbance')
    parser.add_argument('--disable_memory', action='store_true',
                       help='Disable maneuver memory for ablation experiments')
    parser.add_argument('--disable_rule_validator', action='store_true',
                       help='Disable COLREGs-oriented rule validation for ablation experiments')
    parser.add_argument('--rule_baseline', action='store_true',
                       help='Use the deterministic Rule-trigger baseline instead of an LLM')
    parser.add_argument('--method', choices=['auto', 'ppo_lstm'], default='auto',
                       help='Simulation backend; ppo_lstm runs the Sawada-style learning baseline')
    parser.add_argument('--ppo_config', default=None,
                       help='PPO-LSTM JSON config (default: config/ppo_lstm.json)')
    parser.add_argument('--ppo_checkpoint', default=None,
                       help='Trained PPO-LSTM checkpoint (.pt)')
    parser.add_argument('--ppo_device', default='auto',
                       help='PyTorch device for PPO-LSTM evaluation (auto, cpu, cuda, ...)')
    parser.add_argument('--ppo_stochastic', action='store_true',
                       help='Sample PPO-LSTM actions instead of using the deterministic policy mean')
    parser.add_argument('--ppo_save_traces', action='store_true',
                       help='Save PPO-LSTM per-decision trajectory and risk traces')
    parser.add_argument('--vo_prediction_model', choices=['nomoto','constant'], default='nomoto')
    parser.add_argument('--vo_response_dt_s', type=float, default=.5)
    parser.add_argument('--llm_latency_mode', choices=['zero','fixed','random','measured'], default='zero')
    parser.add_argument('--llm_latency_s', type=float, default=0.)
    parser.add_argument('--llm_latency_jitter_s', type=float, default=0.)
    parser.add_argument('--llm_timeout_s', type=float, default=10.)
    parser.add_argument('--llm_max_decision_age_s', type=float, default=15.)
    parser.add_argument('--llm_failure_probability', type=float, default=0.)
    parser.add_argument('--llm_retry_interval_s', type=float, default=1.)
    return parser.parse_args(argv)

def ensure_arg_defaults(args):
    """Fill new optional arguments when run_simulation is called programmatically."""
    defaults = {
        'compare': False,
        'all_cases': False,
        'llm_trigger_mode': 'risk',
        'llm_risk_threshold': LLM_RISK_THRESHOLD,
        'llm_fixed_interval': LLM_FIXED_INTERVAL_STEPS,
        'llm_high_level_update_steps': LLM_HIGH_LEVEL_UPDATE_STEPS,
        'disable_memory': False,
        'disable_rule_validator': False,
        'rule_baseline': False,
        'llm': 0,
        'llm_provider': None,
        'output_dir': 'img/',
        'no_animation': False,
        'low_level_planner': PAPER_LOW_LEVEL_PLANNER,
        'seed': RANDOM_SEED,
        'method': 'auto',
        'ppo_config': None,
        'ppo_checkpoint': None,
        'ppo_device': 'auto',
        'ppo_stochastic': False,
        'ppo_save_traces': False,
        'save_runs': True,
        'vo_prediction_model': 'nomoto', 'vo_response_dt_s': .5,
        'llm_latency_mode': 'zero', 'llm_latency_s': 0., 'llm_latency_jitter_s': 0.,
        'llm_timeout_s': 10., 'llm_max_decision_age_s': 15.,
        'llm_failure_probability': 0., 'llm_retry_interval_s': 1.,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args

def initialize_llm_supervisor(args):
    """Create the risk-triggered LLM supervisor if LLM mode is enabled."""
    if args.llm != 1:
        return None
    if args.rule_baseline:
        return RiskTriggeredLLMSupervisor(
            provider=None,
            risk_threshold=args.llm_risk_threshold,
            trigger_mode=args.llm_trigger_mode,
            fixed_interval=args.llm_fixed_interval,
            high_level_update_steps=args.llm_high_level_update_steps,
            enable_memory=not args.disable_memory,
            enable_validator=not args.disable_rule_validator,
            rule_only=True,
        )
    if not LLM_AVAILABLE:
        raise RuntimeError("LLM requested but its dependencies are unavailable")

    supervisor = RiskTriggeredLLMSupervisor(
        provider=args.llm_provider,
        risk_threshold=args.llm_risk_threshold,
        trigger_mode=args.llm_trigger_mode,
        fixed_interval=args.llm_fixed_interval,
        high_level_update_steps=args.llm_high_level_update_steps,
        enable_memory=not args.disable_memory,
        enable_validator=not args.disable_rule_validator,
    )
    if not supervisor.available:
        raise RuntimeError("LLM requested but provider is not configured; refusing a mislabeled baseline run")
    return supervisor

def update_llm_supervisor(
    supervisor,
    risk,
    distance,
    bearing_rad,
    dcpa_m,
    tcpa_s,
    step,
    time_value,
    distance_in_meters=True,
    target_heading_rad=None,
    own_heading_rad=None,
    target_speed_mps=None,
):
    """Run one RTRA-LLM supervisor update and print compact trigger information."""
    if supervisor is None:
        return None

    distance_nmi = np.asarray(distance) * METERS_TO_NMI if distance_in_meters else np.asarray(distance)

    result = supervisor.update(
        risk=risk,
        distance=distance_nmi,
        bearing=bearing_rad,
        dcpa=np.asarray(dcpa_m) * METERS_TO_NMI,
        tcpa=tcpa_s,
        step=step,
        target_heading=target_heading_rad,
        own_heading=own_heading_rad,
        target_speed=target_speed_mps,
    )

    if result.triggered:
        reasons = ",".join(result.trigger_reasons)
        print(
            f"[RTRA-LLM] t={time_value:6.1f}s step={step} "
            f"TS{result.key_vessel_index + 1} q={result.key_risk:.3f} "
            f"{result.encounter_type} -> {result.final_action.value} "
            f"(Kdir={result.kdir}, trigger={reasons}, valid={result.valid})"
        )
        if should_show_debug() and result.response:
            print(result.response)

    return result

def run_simulation(args=None, return_data=False):
    # Load environment variables from .env file if it exists
    load_env_file()

    # Parse command line arguments if not provided
    if args is None:
        args = parse_args()
    else:
        args = SimpleNamespace(**vars(args))
    args = ensure_arg_defaults(args)

    if args.method == 'ppo_lstm':
        from src.core.ppo_lstm_simulation import run_ppo_lstm
        return run_ppo_lstm(args, return_data=return_data)

    if args.seed is not None:
        np.random.seed(int(args.seed))

    if args.all_cases and not return_data:
        from src.core.batch_experiments import run_batch_simulation
        return run_batch_simulation(args)

    # Check if comparison mode is requested (only if not called from comparison)
    if args.compare and not return_data:
        from src.core.comparison_simulation import run_comparison_simulation
        return run_comparison_simulation(args)

    # Initialize parameters
    t = 0.0  # initial time
    dt = args.dt  # time step
    Ts = dt  # sampling time
    if dt <= 0 or args.sim_time <= 0:
        raise ValueError("Simulation time and step must be positive")
    N = round(args.sim_time / dt)
    if not np.isclose(N * dt, args.sim_time):
        raise ValueError("sim_time must be an integer multiple of dt")
    Animation = not args.no_animation and not return_data

    # Initial conditions
    x_v, y_v, psi_v = 0.0, 0.0, np.radians(0)  # initial position and heading
    r_v, b_v, u_v = 0.0, 0.0, 0.0  # initial rates and velocity
    ui_psi1 = 0.0  # initial integral of yaw error

    X_0 = np.array([x_v, y_v, psi_v, r_v, b_v, u_v])
    X = X_0.copy()

    Sat_amp_s = ACTUATOR_SATURATION
    i = 0

    # Waypoints
    Xwpt = [0, nautical_to_meters(40)/1852]
    Ywpt = [0, 0]
    i_wpt = 1

    # Vessel parameters
    LOA_own, BOL_own = OWN_SHIP_LENGTH_M, OWN_SHIP_BEAM_M
    CPA_own = LOA_own * OWN_SHIP_CPA_MULTIPLIER

    # Get obstacle data
    Xob, Yob, Vob, psiob = get_obstacle_data(args.case_number)
    LOA_ob = [TARGET_SHIP_LENGTH_M] * len(Xob)
    BOL_ob = [TARGET_SHIP_BEAM_M] * len(Xob)
    CPA_ob = [TARGET_SHIP_LENGTH_M * TARGET_SHIP_CPA_MULTIPLIER] * len(Xob)

    # Initialize arrays
    time = []
    Kdir = np.ones(N)  # Initialize Kdir array
    x, y, psi = np.zeros(N), np.zeros(N), np.zeros(N)
    r, b, u = np.zeros(N), np.zeros(N), np.zeros(N)
    v_c = np.zeros(N)
    u_p, tau_c, tau_ac = np.zeros(N), np.zeros(N), np.zeros(N)
    psi_p, psi_wp, psi_oa = np.zeros(N), np.zeros(N), np.zeros(N)
    V_x, V_y = np.zeros(N), np.zeros(N)
    x_nmi, y_nmi = np.zeros(N), np.zeros(N)  # Add nautical mile arrays
    # i_wpt is already initialized above as 1

    Xobs, Yobs, Vxobs, Vyobs = (np.zeros((N, len(Xob))) for _ in range(4))
    DCPA, TCPA, Vrel = (np.zeros((N, len(Xob))) for _ in range(3))
    Distance_ob, Bearing_ob, Risk = (np.zeros((N, len(Xob))) for _ in range(3))

    llm_supervisor = initialize_llm_supervisor(args)
    llm_current_kdir = 0
    if llm_supervisor is not None and args.low_level_planner != "vo":
        raise ValueError("Supervised decisions require the constrained VO planner")

    # Observe the same simplified instruction rules even for the unfiltered VO
    # baseline. This is NOT an independent legal assessment of full COLREGs.
    rule_observer = llm_supervisor or RiskTriggeredLLMSupervisor(provider='offline-rules', rule_only=True)
    delivery = DecisionDelivery(args, dt) if llm_supervisor is not None else None

    # One numerical loop for animated and headless runs. All measurements and
    # decisions refer to state t_i; both own ship and targets then advance to t_i+1.
    plan_safe = np.ones(N, dtype=bool)
    direction_override = np.zeros(N, dtype=bool)
    rule_conflicts = np.zeros(N, dtype=bool)
    rule_evaluable = np.zeros(N, dtype=bool)
    rule_violations = np.zeros(N, dtype=bool)
    plan_clearance = np.full(N, np.inf)
    if Animation:
        fig, ax = plt.subplots()
        plt.plot(Xwpt, Ywpt, 'ob', Xwpt, Ywpt, ':b', linewidth=1.0)
        plt.grid(True)
        writer = animation.PillowWriter(fps=5)
        gif_dir = Path(args.output_dir) / 'gif'
        gif_dir.mkdir(parents=True, exist_ok=True)
        animation_context = writer.saving(fig, str(gif_dir / f'scenario_animation{args.case_number}.gif'), dpi=200)
    else:
        animation_context = nullcontext()
    with animation_context:
        for i in range(N):
            t = i * dt
            time.append(t)
            x[i], y[i], psi[i], r[i], b[i], u[i] = X
            x_nmi[i], y_nmi[i] = X[:2] / 1852
            Xob = np.asarray(Xob, dtype=float)
            Yob = np.asarray(Yob, dtype=float)
            Xobs[i], Yobs[i] = Xob, Yob
            Vxobs[i] = np.asarray(Vob) * np.cos(psiob)
            Vyobs[i] = np.asarray(Vob) * np.sin(psiob)
            DCPA[i], TCPA[i], Distance_ob[i], Bearing_ob[i], Vrel[i] = relative_metrics(
                X[:2], psi[i], OWN_SHIP_SPEED_MPS, np.column_stack((Xob, Yob)), Vob, psiob)
            Risk[i] = risk_calculations(DCPA[i], TCPA[i], Distance_ob[i], Vrel[i])
            observation = dict(risk=Risk[i], distance=Distance_ob[i] / 1852,
                bearing=Bearing_ob[i], dcpa=DCPA[i] / 1852, tcpa=TCPA[i],
                target_heading=psiob, own_heading=psi[i], target_speed=Vob)
            llm_result = delivery.advance(i, llm_supervisor, observation) if delivery else None
            if llm_supervisor is None:
                rule_observer.observe(step=i, **observation)
            rule_conflicts[i] = rule_observer.rule_conflict
            requested_direction = None
            if llm_result is not None:
                llm_current_kdir = llm_result.kdir
                Kdir[i] = llm_current_kdir
                rule_conflicts[i] = llm_result.rule_conflict
                if llm_result.high_risk_vessel_indices and not llm_result.rule_conflict:
                    requested_direction = llm_current_kdir
            else:
                Kdir[i] = 0
            i_wpt = waypoint_selection(Xwpt, Ywpt, x_nmi[i], y_nmi[i], i_wpt)
            psi_wp[i] = planning(Xwpt, Ywpt, x_nmi[i], y_nmi[i], i_wpt)
            if (llm_result and llm_result.valid and llm_result.candidate_action == Action.AC
                    and llm_result.high_risk_vessel_indices
                    and llm_supervisor.memory.active_maneuver_action is not None):
                previous_heading = llm_supervisor.memory.last_applied_heading
                if previous_heading is not None:
                    psi_wp[i] = previous_heading
            if args.low_level_planner == 'vo':
                plan = plan_velocity_obstacle(
                    Xob / 1852, Yob / 1852, x_nmi[i], y_nmi[i], psi[i], Vob, psiob,
                    OWN_SHIP_SPEED_MPS, desired_heading=psi_wp[i], kdir=requested_direction,
                    dynamics_state=X if args.vo_prediction_model == 'nomoto' else None,
                    controller_integral=ui_psi1, response_dt_s=args.vo_response_dt_s)
                psi_oa[i] = plan.correction
                psi_p[i] = plan.heading  # absolute evaluated heading; do not add guidance again
                plan_safe[i] = plan.prediction_safe
                direction_override[i] = plan.direction_overridden
                plan_clearance[i] = plan.predicted_clearance_m
            else:
                correction, *_ = reactive_avoidance(
                    Xob / 1852, Yob / 1852, x_nmi[i], y_nmi[i], psi[i], t)
                psi_p[i] = float(wrap_angle(psi_wp[i] + correction))
                psi_oa[i] = float(wrap_angle(psi_p[i] - psi[i]))
                plan_safe[i] = False  # legacy reactive mode has no VO certificate
            executed = (Action.AR if psi_oa[i] < -1e-10 else
                        Action.AL if psi_oa[i] > 1e-10 else Action.A0)
            targets = rule_observer._last_validation_targets
            rule_observer.feedback_execution(psi_p[i], psi[i], i, safe_prediction=bool(plan_safe[i]))
            maneuver = rule_observer.memory.active_maneuver_action or executed
            rule_evaluable[i] = bool(targets)
            rule_violations[i] = any(not rule_observer._is_action_legal_for_target(maneuver, target)
                                     for target in targets)
            if delivery and delivery.delivered_event is not None:
                delivery.delivered_event.update(executed_heading_rad=float(psi_p[i]),
                    executed_instruction=executed.value, planner_status=plan.status,
                    planner_direction_overridden=bool(direction_override[i]))
            u_p[i] = OWN_SHIP_SPEED_MPS
            tau_c[i], v_c[i], ui_psi1 = controller(
                psi_p[i], psi[i], r[i], u_p[i], b[i], ui_psi1, Ts)
            tau_ac[i] = actuator_modeling(tau_c[i], Sat_amp_s)
            X_dot = vessel_dynamics(X.copy(), [tau_ac[i], v_c[i]])
            V_x[i], V_y[i] = X_dot[:2]
            if Animation:
                animate_step(x_nmi[i], y_nmi[i], psi[i], LOA_own / 1852, BOL_own / 1852, CPA_own,
                             Xob / 1852, Yob / 1852, psiob, np.asarray(LOA_ob) / 1852,
                             np.asarray(BOL_ob) / 1852, CPA_ob, Risk[i], Vob, i, len(Vob))
                if i % 101 == 0:
                    writer.grab_frame()
            X = integration(X, X_dot, dt)
            X[2] = float(wrap_angle(X[2]))
            turn = float(wrap_angle(X[2] - psi[i]))
            rule_observer.memory.observed_turn_action = (Action.AR if turn < -1e-10 else
                Action.AL if turn > 1e-10 else Action.A0)
            Xob, Yob, _, _ = obstacle_sim(Xob, Yob, Vob, psiob, dt)
    if Animation:
        plt.close(fig)

    if not return_data:
        fig, axs = plt.subplots(2, 2)
        for i in range(len(Xob)):
            axs[0, 0].plot(time, DCPA[:, i] / 1852, linewidth=1.0)
            axs[0, 1].plot(time, Distance_ob[:, i]/1852, linewidth=1.0)
            axs[1, 0].plot(time, TCPA[:, i], linewidth=1.0, label=f'TS{i+1}')
            axs[1, 1].plot(time, Risk[:, i], linewidth=1.0)

        axs[0, 0].set_xlim([0, args.sim_time])
        axs[0, 0].set_ylabel(r'$d_{\mathrm{CPA}}$ (nmi)', fontsize=20)
        axs[0, 0].tick_params(axis='both', labelsize=15)

        axs[0, 1].set_xlim([0, args.sim_time])
        axs[0, 1].set_ylim([0, 2000/1852])
        axs[0, 1].set_ylabel(r'$R$ (nmi)', fontsize=20)
        axs[0, 1].tick_params(axis='both', labelsize=15)

        axs[1, 0].set_xlim([0, args.sim_time])
        axs[1, 0].set_xlabel('Time (s)', fontsize=20)
        axs[1, 0].set_ylabel(r'$t_{\mathrm{CPA}}$ (s)', fontsize=20)
        axs[1, 0].tick_params(axis='both', labelsize=15)
        axs[1, 0].legend()

        axs[1, 1].set_xlim([0, args.sim_time])
        axs[1, 1].set_ylim([0, 1])
        axs[1, 1].set_xlabel('Time (s)', fontsize=20)
        axs[1, 1].set_ylabel(r'$q$', fontsize=20)
        axs[1, 1].tick_params(axis='both', labelsize=15)

        fig.suptitle(f'Case {args.case_number}', fontsize=20)
        plt.tight_layout()
        save_figure_with_type(fig, 'plot_dcpa_tcpa_risk', 'eps', args.case_number)
        save_figure_with_type(fig, 'plot_dcpa_tcpa_risk', 'png', args.case_number, dpi=300)
        if args.no_animation:
            plt.close(fig)
        else:
            plt.show()

    if not return_data:
        print("\n=== Simulation Summary ===")
        print(f"Case: {args.case_number}")
        print(f"Duration: {args.sim_time} seconds")
        print(f"Total simulation steps: {len(time)}")

        if len(Xob) > 0:
            min_dcpa = np.min(np.abs(DCPA)) / 1852
            max_risk = np.max(Risk)
            print(f"Minimum DCPA: {min_dcpa:.2f} nautical miles")
            print(f"Maximum Risk: {max_risk:.3f}")

            starboard_turns = np.sum(Kdir == 1)
            port_turns = np.sum(Kdir == -1)
            stand_on = np.sum(Kdir == 0)
            print(f"\nManeuver Statistics:")
            print(f"  - Starboard turns: {starboard_turns}")
            print(f"  - Port turns: {port_turns}")
            print(f"  - Stand on: {stand_on}")

        if llm_supervisor is not None:
            print(f"\nLLM Supervisor Statistics:")
            print(f"  - LLM calls: {llm_supervisor.call_count}")
            print(f"  - Trigger events: {len(llm_supervisor.trigger_history)}")
            print(f"  - Trigger mode: {args.llm_trigger_mode}")

        print(f"\nOutput files generated:")
        if Animation:
            print(f"  - Animation: img/gif/scenario_animation{args.case_number}.gif")
        print(f"  - Plots: img/png/plot_dcpa_tcpa_risk_{args.case_number}.png")
        print(f"  - Plots: img/eps/plot_dcpa_tcpa_risk_{args.case_number}.eps")
        print("=" * 50)

    # Build a reusable synchronous trace for every run.
    comparison_kdir = np.zeros(len(Kdir))
    for i in range(len(Kdir)):
        if psi_oa[i] < -1e-10:
            comparison_kdir[i] = 1
        elif psi_oa[i] > 1e-10:
            comparison_kdir[i] = -1
        else:
            comparison_kdir[i] = 0

    result = {
        'time': np.asarray(time),
        'x': x,
        'y': y,
        'psi': psi,
        'kdir': comparison_kdir,
        'control_kdir': Kdir.copy(),
        'risk': Risk,
        'dcpa': DCPA,
        'tcpa': TCPA,
        'distance_ob': Distance_ob,
        'obstacles_x': np.asarray(Xob),
        'obstacles_y': np.asarray(Yob),
        'obstacles_x_history': Xobs,
        'obstacles_y_history': Yobs,
        'obstacles_heading_rad': np.asarray(psiob),
        'target_speed_mps': np.asarray(Vob),
        'waypoints_x': np.asarray(Xwpt),
        'waypoints_y': np.asarray(Ywpt),
        'case_number': args.case_number,
        'dt': dt,
        'llm_call_count': llm_supervisor.call_count if llm_supervisor is not None else 0,
        'llm_trigger_history': llm_supervisor.trigger_history if llm_supervisor is not None else [],
        'llm_trigger_mode': args.llm_trigger_mode,
        'llm_fixed_interval': args.llm_fixed_interval,
        'llm_high_level_update_steps': args.llm_high_level_update_steps,
        'llm_risk_threshold': args.llm_risk_threshold,
        'llm_model': llm_supervisor.model_name if llm_supervisor is not None else '',
        'llm_memory_enabled': not args.disable_memory,
        'llm_validator_enabled': not args.disable_rule_validator,
        'rule_baseline': args.rule_baseline,
        'low_level_planner': args.low_level_planner,
        'vo_prediction_model': args.vo_prediction_model,
        'random_seed': args.seed,
        'simulation_type': 'main_simulation',
        'final_position_m': X[:2].copy(),
        'final_heading_rad': float(X[2]),
        'final_target_positions_m': np.column_stack((Xob, Yob)),
        'desired_heading_rad': psi_p,
        'planner_prediction_safe': plan_safe,
        'planner_direction_override': direction_override,
        'planner_predicted_clearance_m': plan_clearance,
        'rule_conflict': rule_conflicts,
        'rule_evaluable': rule_evaluable,
        'rule_violation': rule_violations,
        'output_parse_failure_pct': (100 * llm_supervisor.parse_failure_count / llm_supervisor.received_response_count
                                     if llm_supervisor and llm_supervisor.received_response_count else float('nan')),
        'execution_rule_violation_pct': (100 * rule_violations.sum() / rule_evaluable.sum()
                                         if rule_evaluable.any() else float('nan')),
        'inference_time_model': args.llm_latency_mode if delivery else 'no_llm',
        'llm_call_failure_count': llm_supervisor.call_failure_count if llm_supervisor else 0,
        'llm_received_response_count': llm_supervisor.received_response_count if llm_supervisor else 0,
        'llm_discarded_decisions': delivery.discarded_count if delivery else 0,
        'llm_pending_at_end': bool(delivery and delivery.pending is not None),
    }
    result['metrics'] = trajectory_metrics(result)
    if args.save_runs:
        result['archive_path'] = save_run(result, args)
    if return_data:
        return result

if __name__ == "__main__":
    run_simulation()
