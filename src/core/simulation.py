import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from src.navigation.planning import waypoint_selection, planning
from src.dynamics.vessel_dynamics import vessel_dynamics
from src.core.integration import integration
from src.navigation.obstacle_sim import obstacle_sim
from src.risk_assessment.cpa_calculations import cpa_calculations
from src.risk_assessment.cpa_calculations_0speed import cpa_calculations_0speed
from src.dynamics.controller import controller
from src.dynamics.actuator_modeling import actuator_modeling
from src.risk_assessment.risk_calculations import risk_calculations
from src.navigation.reactive_avoidance import reactive_avoidance
from src.visualization.animate import animate_step
from src.utils.imazu_cases import nautical_to_meters, get_obstacle_data
# Optional LLM imports
try:
    from src.decision_making.rtra_llm_supervisor import RiskTriggeredLLMSupervisor
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

def parse_args():
    parser = argparse.ArgumentParser(description='Marine Vehicle Simulation')
    parser.add_argument('--case_number', type=int, default=1, help='Simulation case number')
    parser.add_argument('--all_cases', action='store_true', help='Run all supported Imazu cases')
    parser.add_argument('--sim_time', type=float, default=450.0, help='Simulation time in seconds')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step size')
    parser.add_argument('--no_animation', action='store_true', help='Disable animation')
    parser.add_argument('--output_dir', type=str, default='img/', help='Output directory for results')
    parser.add_argument('--llm', type=int, default=0, help='Use LLM for decision making (0=off, 1=on)')
    parser.add_argument('--llm_provider', type=str, default=None, 
                       help='OpenAI-compatible LLM provider to use (for example: openai, zhipu, qwen, deepseek, kimi). If not specified, uses LLM_PROVIDER from .env')
    parser.add_argument('--compare', action='store_true', 
                       help='Run comparison between LLM and baseline simulation')
    parser.add_argument('--llm_trigger_mode', choices=['risk', 'fixed', 'always'], default='risk',
                       help='LLM trigger strategy: risk=paper method, fixed=fixed-interval ablation, always=call every step')
    parser.add_argument('--llm_risk_threshold', type=float, default=0.30,
                       help='Risk threshold q for triggering LLM supervision')
    parser.add_argument('--llm_fixed_interval', type=int, default=200,
                       help='Fixed trigger interval in simulation steps when --llm_trigger_mode fixed is used')
    parser.add_argument('--disable_memory', action='store_true',
                       help='Disable maneuver memory for ablation experiments')
    parser.add_argument('--disable_rule_validator', action='store_true',
                       help='Disable COLREGs-oriented rule validation for ablation experiments')
    return parser.parse_args()

def ensure_arg_defaults(args):
    """Fill new optional arguments when run_simulation is called programmatically."""
    defaults = {
        'compare': False,
        'all_cases': False,
        'llm_trigger_mode': 'risk',
        'llm_risk_threshold': 0.30,
        'llm_fixed_interval': 200,
        'disable_memory': False,
        'disable_rule_validator': False,
        'llm': 0,
        'llm_provider': None,
        'output_dir': 'img/',
        'no_animation': False,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args

def initialize_llm_supervisor(args):
    """Create the risk-triggered LLM supervisor if LLM mode is enabled."""
    if args.llm != 1:
        return None
    if not LLM_AVAILABLE:
        print("Warning: LLM requested but langchain_openai or RTRA supervisor is not available. Running without LLM.")
        return None

    supervisor = RiskTriggeredLLMSupervisor(
        provider=args.llm_provider,
        risk_threshold=args.llm_risk_threshold,
        trigger_mode=args.llm_trigger_mode,
        fixed_interval=args.llm_fixed_interval,
        enable_memory=not args.disable_memory,
        enable_validator=not args.disable_rule_validator,
    )
    if not supervisor.available:
        print("Warning: LLM requested but provider is not configured. Running without LLM.")
        return None
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
    N = round(args.sim_time / dt)
    progress_interval = max(1, int(5 / dt))
    Animation = not args.no_animation and not return_data

    # Initial conditions
    x_v, y_v, psi_v = 0.0, 0.0, np.radians(0)  # initial position and heading
    r_v, b_v, u_v = 0.0, 0.0, 0.0  # initial rates and velocity
    ui_psi1 = 0.0  # initial integral of yaw error

    X_0 = np.array([x_v, y_v, psi_v, r_v, b_v, u_v])
    X = X_0.copy()

    Sat_amp_s = 20
    i = 0

    # Waypoints
    Xwpt = [0, nautical_to_meters(40)/1852]
    Ywpt = [0, 0]
    i_wpt = 1

    # Vessel parameters
    LOA_own, BOL_own = 30, 16
    CPA_own = LOA_own * 2
    
    # Get obstacle data
    Xob, Yob, Vob, psiob = get_obstacle_data(args.case_number)
    LOA_ob = [80] * len(Xob)
    BOL_ob = [30] * len(Xob)
    CPA_ob = [LOA_ob[0] * 1] * len(Xob)

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
    DCPA, TCPA, Vrel, alpha, psi_Vrel = (np.zeros((N, len(Xob))) for i in range(5))
    DCPA[:1], TCPA[:1] = 1000, 1000
    DCPA2, TCPA2, Vrel2, alpha2, psi_Vrel2 = (np.zeros((N, len(Xob))) for _ in range(5))
    Distance_ob, Bearing_ob, Risk = (np.zeros((N, len(Xob))) for _ in range(3))

    llm_supervisor = initialize_llm_supervisor(args)
    llm_current_kdir = 1

    # Prepare for animation if enabled
    if Animation:
        fig, ax = plt.subplots()
        plt.plot(Xwpt, Ywpt, 'ob', Xwpt, Ywpt, ':b', linewidth=1.0)
        plt.grid(True)
        writer = animation.PillowWriter(fps=5)

        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)

    # Main simulation loop
    print(f"\n=== Starting Simulation ===")
    print(f"Case: {args.case_number}")
    print(f"Duration: {args.sim_time} seconds")
    print(f"LLM enabled: {'Yes' if args.llm == 1 else 'No'}")
    print(f"LLM provider: {args.llm_provider}")
    if args.llm == 1:
        print(f"LLM trigger mode: {args.llm_trigger_mode}")
        print(f"LLM risk threshold: {args.llm_risk_threshold}")
        print(f"Maneuver memory: {'Off' if args.disable_memory else 'On'}")
        print(f"Rule validator: {'Off' if args.disable_rule_validator else 'On'}")
    print(f"Animation: {'Yes' if Animation else 'No'}")
    print("-" * 50)

    if Animation:
        # Save in gif directory
        output_path = f"scenario_animation{args.case_number}.gif"
        gif_dir = Path("./img/gif")
        gif_dir.mkdir(parents=True, exist_ok=True)
        full_output_path = gif_dir / output_path

        with writer.saving(fig, str(full_output_path), dpi=200):
            for i in range(len(x)):
                # Record current state
                time.append(t)
                x[i] = X[0]
                y[i] = X[1]
                psi[i] = X[2]
                r[i] = X[3]
                b[i] = X[4]
                u[i] = X[5]
                X_0 = X.copy()

                # Speed command
                u_p[i] = 43.3

                # Convert to nautical miles
                METERS_TO_NMI = (1 / 1852)
                x_nmi = [xi * METERS_TO_NMI for xi in x]
                y_nmi = [yi * METERS_TO_NMI for yi in y]
                Xob_nmi = [xo * METERS_TO_NMI for xo in Xob]
                Yob_nmi = [yo * METERS_TO_NMI for yo in Yob]
                LOA_own_nmi = LOA_own * METERS_TO_NMI
                BOL_own_nmi = BOL_own * METERS_TO_NMI
                LOA_ob_nmi = [loa * METERS_TO_NMI for loa in LOA_ob]
                BOL_ob_nmi = [bol * METERS_TO_NMI for bol in BOL_ob]

                # Path planning and collision avoidance
                i_wpt = waypoint_selection(Xwpt, Ywpt, x_nmi[i], y_nmi[i], i_wpt)
                psi_wp[i] = planning(Xwpt, Ywpt, x_nmi[i], y_nmi[i], i_wpt)
                psi_oa[i], w_B, w_R, Distance_ob[i, :], Bearing_ob[i, :] = reactive_avoidance(
                    Xob_nmi, Yob_nmi, x_nmi[i], y_nmi[i], psi[i], t)
                if llm_supervisor is not None:
                    Kdir[i] = llm_current_kdir

                # Overall yaw command with Kdir
                psi_p[i] = psi_wp[i] + Kdir[i] * psi_oa[i]

                # Controller and actuator
                tau_c[i], v_c[i], ui_psi1 = controller(
                    psi_p[i], psi[i], r[i], u_p[i], b, ui_psi1, Ts)
                tau_ac[i] = actuator_modeling(tau_c[i], Sat_amp_s)
                
                # System dynamics
                inputs = [tau_ac[i], v_c[i]]
                X_dot = vessel_dynamics(X_0, inputs)
                X = integration(X_0, X_dot, dt)
                V_x[i] = X_dot[0]
                V_y[i] = X_dot[1]

                # Obstacles simulation
                Xob, Yob, Vxob, Vyob = obstacle_sim(Xob, Yob, Vob, psiob, dt)
                Xobs[i, :] = Xob
                Yobs[i, :] = Yob
                Vxobs[i, :] = Vxob
                Vyobs[i, :] = Vyob

                # Risk analysis
                for j in range(len(Xob)):
                    if i >= 1:
                        Distance_ob[i, j] = np.sqrt(
                            (np.array(Xobs[i, j]) - x[i])**2 + 
                            (np.array(Yobs[i, j]) - y[i])**2)
                        
                        DCPA[i, j], TCPA[i, j], Vrel[i, j], alpha[i, j], psi_Vrel[i, j] = cpa_calculations(
                            x[i], y[i], x[i-1], y[i-1], Xobs[i, j], Yobs[i, j], 
                            Xobs[i-1, j], Yobs[i-1, j], Ts
                        )

                        DCPA2[i, j], TCPA2[i, j], Vrel2[i, j], alpha2[i, j], psi_Vrel2[i, j] = cpa_calculations_0speed(
                            x[i], y[i], Xobs[i, j], Yobs[i, j], V_x[i], V_y[i], 
                            Vxobs[i, j], Vyobs[i, j], Distance_ob[i, j]
                        )

                    Risk[i, j] = risk_calculations(
                        DCPA[i, j], TCPA[i, j], Distance_ob[i, j], Vrel[i, j])

                llm_result = update_llm_supervisor(
                    llm_supervisor,
                    Risk[i, :],
                    Distance_ob[i, :],
                    Bearing_ob[i, :],
                    DCPA[i, :],
                    TCPA[i, :],
                    i,
                    t,
                    distance_in_meters=i >= 1,
                )
                if llm_result is not None:
                    llm_current_kdir = llm_result.kdir

                # Animation
                l = len(Risk[i, :])
                animate_step(
                    x_nmi[i], y_nmi[i], psi[i],
                    LOA_own_nmi, BOL_own_nmi, CPA_own,
                    Xob_nmi, Yob_nmi, psiob,
                    LOA_ob_nmi, BOL_ob_nmi, CPA_ob,
                    Risk[i, :], Vob, i, l
                )
                
                if i % 101 == 0 and i != 0:
                    writer.grab_frame()

                t += dt

            # Save animation plots
            plt.title(f'Case {args.case_number}', fontsize=25)
            save_figure_with_type(plt.gcf(), 'simulation_result', 'eps', args.case_number)
            save_figure_with_type(plt.gcf(), 'simulation_result', 'png', args.case_number, dpi=300)
            plt.show(block=True)
    else:
        # Run simulation without animation
        for i in range(len(x)):
            # Record current state
            time.append(t)
            x[i] = X[0]
            y[i] = X[1]
            psi[i] = X[2]
            r[i] = X[3]
            b[i] = X[4]
            u[i] = X[5]

            # Convert to nautical miles
            x_nmi[i] = x[i] / 1852
            y_nmi[i] = y[i] / 1852

            # Speed command
            u_p[i] = 43.3

            # Convert to nautical miles
            METERS_TO_NMI = (1 / 1852)
            x_nmi = [xi * METERS_TO_NMI for xi in x]
            y_nmi = [yi * METERS_TO_NMI for yi in y]
            Xob_nmi = [xo * METERS_TO_NMI for xo in Xob]
            Yob_nmi = [yo * METERS_TO_NMI for yo in Yob]
            LOA_own_nmi = LOA_own * METERS_TO_NMI
            BOL_own_nmi = BOL_own * METERS_TO_NMI
            LOA_ob_nmi = [loa * METERS_TO_NMI for loa in LOA_ob]
            BOL_ob_nmi = [bol * METERS_TO_NMI for bol in BOL_ob]

            # Path planning and collision avoidance
            i_wpt = waypoint_selection(Xwpt, Ywpt, x_nmi[i], y_nmi[i], i_wpt)
            psi_wp[i] = planning(Xwpt, Ywpt, x_nmi[i], y_nmi[i], i_wpt)
            psi_oa[i], w_B, w_R, Distance_ob[i, :], Bearing_ob[i, :] = reactive_avoidance(
                Xob_nmi, Yob_nmi, x_nmi[i], y_nmi[i], psi[i], t)
            if llm_supervisor is not None:
                Kdir[i] = llm_current_kdir

            # Overall yaw command with Kdir
            psi_p[i] = psi_wp[i] + Kdir[i] * psi_oa[i]

            # Controller and actuator
            tau_c[i], v_c[i], ui_psi1 = controller(
                psi_p[i], psi[i], r[i], u_p[i], b, ui_psi1, Ts)
            tau_ac[i] = actuator_modeling(tau_c[i], Sat_amp_s)

            # Store current state
            X_0 = X.copy()
            
            # System dynamics
            inputs = [tau_ac[i], v_c[i]]
            X_dot = vessel_dynamics(X_0, inputs)
            X = integration(X_0, X_dot, dt)
            V_x[i] = X_dot[0]
            V_y[i] = X_dot[1]

            # Obstacles simulation
            Xob, Yob, Vxob, Vyob = obstacle_sim(Xob, Yob, Vob, psiob, dt)
            Xobs[i, :] = Xob
            Yobs[i, :] = Yob
            Vxobs[i, :] = Vxob
            Vyobs[i, :] = Vyob

            # Risk analysis
            for j in range(len(Xob)):
                if i >= 1:
                    Distance_ob[i, j] = np.sqrt(
                        (np.array(Xobs[i, j]) - x[i])**2 + 
                        (np.array(Yobs[i, j]) - y[i])**2)
                    
                    DCPA[i, j], TCPA[i, j], Vrel[i, j], alpha[i, j], psi_Vrel[i, j] = cpa_calculations(
                        x[i], y[i], x[i-1], y[i-1], Xobs[i, j], Yobs[i, j], 
                        Xobs[i-1, j], Yobs[i-1, j], Ts
                    )

                    DCPA2[i, j], TCPA2[i, j], Vrel2[i, j], alpha2[i, j], psi_Vrel2[i, j] = cpa_calculations_0speed(
                        x[i], y[i], Xobs[i, j], Yobs[i, j], V_x[i], V_y[i], 
                        Vxobs[i, j], Vyobs[i, j], Distance_ob[i, j]
                    )

                    Risk[i, j] = risk_calculations(
                        DCPA[i, j], TCPA[i, j], Distance_ob[i, j], Vrel[i, j])

            # Print simulation progress every 5 seconds
            if i % progress_interval == 0:
                avg_risk = np.mean(Risk[i, :])
                print(f"Time: {t:6.1f}s | Avg Risk: {avg_risk:.3f} | Kdir: {Kdir[i]:.1f}")

            llm_result = update_llm_supervisor(
                llm_supervisor,
                Risk[i, :],
                Distance_ob[i, :],
                Bearing_ob[i, :],
                DCPA[i, :],
                TCPA[i, :],
                i,
                t,
                distance_in_meters=i >= 1,
            )
            if llm_result is not None:
                llm_current_kdir = llm_result.kdir

            t += dt

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
            min_dcpa = np.min(DCPA) / 1852
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

    # Return data if requested (for comparison mode)
    if return_data:
        # Calculate Kdir for comparison based on actual control values
        # Kdir = 0 when Kdir[i] * psi_oa[i] == 0 (no turn)
        # Kdir = +1 when Kdir[i] * psi_oa[i] > 0 (starboard)
        # Kdir = -1 when Kdir[i] * psi_oa[i] < 0 (port)
        comparison_kdir = np.zeros(len(Kdir))
        for i in range(len(Kdir)):
            control_value = Kdir[i] * psi_oa[i]
            if control_value > 0:
                comparison_kdir[i] = 1  # Starboard
            elif control_value < 0:
                comparison_kdir[i] = -1  # Port
            else:
                comparison_kdir[i] = 0  # No turn
        
        return {
            'time': time,
            'x': x,
            'y': y,
            'psi': psi,
            'kdir': comparison_kdir,  # Use calculated comparison values
            'risk': Risk,
            'dcpa': DCPA,
            'tcpa': TCPA,
            'distance_ob': Distance_ob,
            'obstacles_x': Xobs[-1, :] if len(Xobs) > 0 else [],
            'obstacles_y': Yobs[-1, :] if len(Yobs) > 0 else [],
            'llm_call_count': llm_supervisor.call_count if llm_supervisor is not None else 0,
            'llm_trigger_history': llm_supervisor.trigger_history if llm_supervisor is not None else [],
            'llm_trigger_mode': args.llm_trigger_mode,
            'llm_risk_threshold': args.llm_risk_threshold,
            'llm_memory_enabled': not args.disable_memory,
            'llm_validator_enabled': not args.disable_rule_validator,
            'simulation_type': 'main_simulation'
        }

""""
    # Plot Kdir values
    plt.figure()
    plt.plot(time, Kdir, 'b', linewidth=1.5)
    plt.title(f'Case {args.case_number}')
    plt.xlabel('Time (s)')
    plt.ylabel(r'$K_{dir}$')
    plt.grid(True)
    save_figure_with_type(plt.gcf(), 'plot_kdir', 'png', args.case_number, dpi=300)
    plt.show()"""

if __name__ == "__main__":
    run_simulation()
