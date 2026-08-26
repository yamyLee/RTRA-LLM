import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import os
from pathlib import Path

def save_figure_with_type(fig, base_path, file_type, case_number, provider=None, dpi=None, output_dir='img'):
    """Save figure with proper directory structure based on file type."""
    # Create directory structure
    img_dir = Path(output_dir)
    file_type_dir = img_dir / file_type
    file_type_dir.mkdir(parents=True, exist_ok=True)

    # Save figure
    if file_type == 'eps':
        if provider:
            filepath = file_type_dir / f"{base_path}_{provider.lower()}_case_{case_number}.eps"
        else:
            filepath = file_type_dir / f"{base_path}_{case_number}.eps"
        fig.savefig(filepath, format='eps', bbox_inches='tight', facecolor='white')
    elif file_type == 'png':
        if provider:
            filepath = file_type_dir / f"{base_path}_{provider.lower()}_case_{case_number}.png"
        else:
            filepath = file_type_dir / f"{base_path}_{case_number}.png"
        fig.savefig(filepath, bbox_inches='tight', facecolor='white', dpi=dpi)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    return filepath

def plot_kdir_comparison(time_baseline: List[float], 
                        kdir_baseline: np.ndarray,
                        time_llm: List[float], 
                        kdir_llm: np.ndarray,
                        case_number: int,
                        output_dir: str = 'img/',
                        llm_provider: str = 'LLM') -> str:
    """
    Create a clean comparison plot of Kdir values between baseline and LLM simulations.
    
    Args:
        time_baseline: Time array for baseline simulation
        kdir_baseline: Kdir values for baseline simulation
        time_llm: Time array for LLM simulation
        kdir_llm: Kdir values for LLM simulation
        case_number: Simulation case number
        output_dir: Output directory for plots
        llm_provider: Name of LLM provider used
    
    Returns:
        str: Path to saved plot file
    """
    
    # Create single clean figure
    plt.figure(figsize=(14, 8))
    
    # Plot Kdir comparison with beautiful styling
    plt.plot(time_baseline, kdir_baseline, 'b-', linewidth=3, label='Simulation (Baseline)', alpha=0.8)
    plt.plot(time_llm, kdir_llm, 'r-', linewidth=3, label=f'LLM ({llm_provider})', alpha=0.8)
    
    # Add horizontal reference lines
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.6, linewidth=2)
    plt.axhline(y=1, color='green', linestyle=':', alpha=0.6, linewidth=1)
    plt.axhline(y=-1, color='orange', linestyle=':', alpha=0.6, linewidth=1)
    
    # Styling
    plt.xlabel('Time (seconds)', fontsize=16, fontweight='bold')
    plt.ylabel('Kdir (Turn Direction)', fontsize=16, fontweight='bold')
    plt.title(f'Navigation Decisions Comparison - Case {case_number}', fontsize=18, fontweight='bold', pad=20)
    
    # Beautiful legend
    plt.legend(loc='upper right', fontsize=14, frameon=True, fancybox=True, shadow=True)
    
    # Grid and styling
    plt.grid(True, alpha=0.3, linewidth=0.5)
    plt.ylim(-1.2, 1.2)
    
    # Add direction labels on the right side
    plt.text(1.02, 1, 'Starboard\n(Right Turn)', transform=plt.gca().transAxes, 
            verticalalignment='center', fontsize=12, fontweight='bold', color='green')
    plt.text(1.02, 0.5, 'No Turn\n(Straight)', transform=plt.gca().transAxes, 
            verticalalignment='center', fontsize=12, fontweight='bold', color='gray')
    plt.text(1.02, 0, 'Port\n(Left Turn)', transform=plt.gca().transAxes, 
            verticalalignment='center', fontsize=12, fontweight='bold', color='orange')
    
    # Set y-axis ticks
    plt.yticks([-1, 0, 1], ['−1 (Port)', '0 (Straight)', '+1 (Starboard)'], fontsize=12)
    plt.xticks(fontsize=12)
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot
    filepath = save_figure_with_type(
        plt.gcf(), 'kdir_comparison', 'png', case_number, llm_provider, dpi=300, output_dir=output_dir
    )

    # Also save as EPS
    save_figure_with_type(
        plt.gcf(), 'kdir_comparison', 'eps', case_number, llm_provider, output_dir=output_dir
    )
    plt.close()
    
    return filepath


def create_comparison_summary(baseline_stats: Dict[str, Any], 
                            llm_stats: Dict[str, Any],
                            case_number: int,
                            output_dir: str = 'img/',
                            llm_provider: str = 'LLM') -> str:
    """
    Create a summary comparison report.
    """
    
    summary_text = f"""
# Simulation Comparison Report - Case {case_number}

## Configuration
- **LLM Provider**: {llm_provider}
- **Case Number**: {case_number}
- **Simulation Time**: {baseline_stats.get('sim_time', 'N/A')} seconds
- **LLM Trigger Mode**: {llm_stats.get('trigger_mode', 'N/A')}
- **Risk Threshold**: {llm_stats.get('risk_threshold', 'N/A')}

## Navigation Behavior Comparison

### Turn Commands (Kdir)
- **Baseline Total Turns**: {baseline_stats.get('total_turns', 'N/A')}
- **{llm_provider} Total Turns**: {llm_stats.get('total_turns', 'N/A')}
- **Turn Agreement Rate**: {baseline_stats.get('turn_agreement', 'N/A')}

### LLM Invocation
- **LLM Calls**: {llm_stats.get('llm_calls', 'N/A')}
- **Trigger Events**: {llm_stats.get('trigger_events', 'N/A')}

### Risk Management
- **Baseline Max Risk**: {baseline_stats.get('max_risk', 'N/A'):.3f}
- **{llm_provider} Max Risk**: {llm_stats.get('max_risk', 'N/A'):.3f}
- **Baseline Avg Risk**: {baseline_stats.get('avg_risk', 'N/A'):.3f}
- **{llm_provider} Avg Risk**: {llm_stats.get('avg_risk', 'N/A'):.3f}

### Performance Metrics
- **Baseline Final Distance**: {baseline_stats.get('final_distance', 'N/A'):.2f} nm
- **{llm_provider} Final Distance**: {llm_stats.get('final_distance', 'N/A'):.2f} nm
- **Path Efficiency Difference**: {baseline_stats.get('path_efficiency_diff', 'N/A'):.1f}%

## Analysis
{baseline_stats.get('analysis', 'Analysis not available')}

---
Generated by Marine Vessel Simulation System
"""
    
    # Save summary
    os.makedirs(output_dir, exist_ok=True)
    filename = f'comparison_summary_{llm_provider.lower()}_case_{case_number}.txt'
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write(summary_text)
    
    return filepath
