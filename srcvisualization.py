"""
Visualization module.

This module provides functions for creating publication-quality figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_survival_curves(
    std_times: np.ndarray,
    opt_times: np.ndarray,
    max_time: int = 120,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot survival curves comparing standard and optimal policies.
    
    Parameters:
    -----------
    std_times : np.ndarray
        Absorption times under standard policy
    opt_times : np.ndarray
        Absorption times under optimal policy
    max_time : int
        Maximum time to display
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    fig : plt.Figure
        Figure object
    """
    from .simulation import compute_survival_curve
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute survival curves
    t_std, surv_std = compute_survival_curve(std_times, max_time)
    t_opt, surv_opt = compute_survival_curve(opt_times, max_time)
    
    # Plot
    ax.step(t_std, surv_std, label='Standard Policy', color='red', alpha=0.7, linewidth=1.5)
    ax.step(t_opt, surv_opt, label='Optimal Policy $\\pi^*$', color='green', linewidth=2.5)
    ax.fill_between(t_opt, surv_opt, surv_std, color='gray', alpha=0.15)
    
    # Labels and formatting
    ax.set_xlabel('Instructional Hours', fontsize=12)
    ax.set_ylabel('P(T > t)  Probability of Non-Mastery', fontsize=12)
    ax.set_title('Survival Analysis: Probability of Non-Mastery over Time', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(0, max_time)
    ax.set_ylim(0, 1.05)
    
    # Add horizontal line at 0.5 for median interpretation
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.3, linewidth=0.8)
    
    # Annotate the 49% reduction
    ax.annotate(f'49% reduction at beginner level',
                xy=(44.5, 0.5),
                xytext=(60, 0.6),
                arrowprops=dict(arrowstyle='->', color='darkblue'),
                fontsize=10,
                color='darkblue')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_spectral_radii(
    spectral_radii: Dict[str, float],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot spectral radii for different actions.
    
    Parameters:
    -----------
    spectral_radii : Dict[str, float]
        Action -> spectral radius
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    fig : plt.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by value
    sorted_items = sorted(spectral_radii.items(), key=lambda x: x[1])
    actions = [item[0].split('::')[1].replace('_', ' ') for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    # Colors: green for low, red for high
    colors = plt.cm.RdYlGn_r(np.array(values) / max(values))
    
    bars = ax.barh(actions, values, color=colors)
    ax.set_xlabel('Spectral Radius $\\rho(Q)$', fontsize=12)
    ax.set_ylabel('Pedagogical Action', fontsize=12)
    ax.set_title('Spectral Radius by Pedagogical Action', fontsize=14)
    
    # Add vertical line at mean
    mean_rho = np.mean(values)
    ax.axvline(x=mean_rho, color='black', linestyle='--', alpha=0.5, 
               label=f'Mean: {mean_rho:.3f}')
    ax.legend()
    
    # Highlight the best and worst actions
    best_action = sorted_items[0][0].split('::')[1].replace('_', ' ')
    worst_action = sorted_items[-1][0].split('::')[1].replace('_', ' ')
    ax.annotate(f'Best: {best_action}',
                xy=(sorted_items[0][1], len(actions)-1),
                xytext=(sorted_items[0][1] + 0.05, len(actions)-1),
                fontsize=9, color='green')
    ax.annotate(f'Worst: {worst_action}',
                xy=(sorted_items[-1][1], 0),
                xytext=(sorted_items[-1][1] - 0.25, 0),
                fontsize=9, color='red')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_expected_times(
    expected_times: Dict[str, np.ndarray],
    state_names: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot expected times to mastery for different policies.
    
    Parameters:
    -----------
    expected_times : Dict[str, np.ndarray]
        Policy name -> expected times from each state
    state_names : List[str]
        Names of transient states
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    fig : plt.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(state_names))
    width = 0.35
    
    policy_names = list(expected_times.keys())
    colors = ['#2ecc71', '#e74c3c']
    
    for i, (name, times) in enumerate(expected_times.items()):
        offset = (i - (len(policy_names) - 1) / 2) * width
        bars = ax.bar(x + offset, times, width, label=name, color=colors[i])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}h',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Initial Cognitive State', fontsize=12)
    ax.set_ylabel('Expected Time to Mastery (hours)', fontsize=12)
    ax.set_title('Expected Time to Mastery by Policy', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(state_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_state_distribution(
    df: pd.DataFrame,
    state_names: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distribution of students across states.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with 'Pretest' and 'Final' columns
    state_names : List[str]
        Names of states
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    fig : plt.Figure
        Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pretest distribution
    pretest_counts = df['Pretest'].value_counts().sort_index()
    axes[0].bar(state_names, pretest_counts.values, color='#3498db', alpha=0.7)
    axes[0].set_title('Initial State Distribution (Pretest)', fontsize=12)
    axes[0].set_xlabel('State')
    axes[0].set_ylabel('Number of Students')
    
    # Final distribution
    final_counts = df['Final'].value_counts().sort_index()
    axes[1].bar(state_names, final_counts.values, color='#2ecc71', alpha=0.7)
    axes[1].set_title('Final State Distribution (Post-test)', fontsize=12)
    axes[1].set_xlabel('State')
    axes[1].set_ylabel('Number of Students')
    
    # Add total students
    axes[0].text(0.95, 0.95, f'n = {len(df)}', 
                 transform=axes[0].transAxes, ha='right', va='top', fontsize=10)
    axes[1].text(0.95, 0.95, f'n = {len(df)}', 
                 transform=axes[1].transAxes, ha='right', va='top', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig