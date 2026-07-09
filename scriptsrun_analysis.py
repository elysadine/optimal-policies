#!/usr/bin/env python
"""
Main analysis script for the optimal teaching policies paper.

This script runs the complete analysis pipeline and generates all figures
and tables reported in the paper.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_and_preprocess_data, load_reading_data, get_state_mapping
from src.transition_matrix import estimate_transition_matrices, aggregate_transition_matrix, print_transition_matrix
from src.spectral_analysis import (
    compute_spectral_radius, 
    compute_fundamental_matrix, 
    expected_time_to_mastery,
    estimate_holding_times,
    bootstrap_spectral_radius
)
from src.mdp_solver import solve_value_iteration, find_spectral_optimal_policy, evaluate_policy
from src.simulation import simulate_cohort
from src.visualization import (
    plot_survival_curves,
    plot_spectral_radii,
    plot_expected_times,
    plot_state_distribution
)


def run_analysis(data_file_math: str, data_file_reading: str, output_dir: str = 'results'):
    """
    Run complete analysis pipeline.
    """
    print("=" * 70)
    print("SPECTRAL OPTIMIZATION FOR TARL LEARNING TRAJECTORIES")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    # 1. Load data
    print("\n[1] Loading data...")
    df_math, action_cols, states = load_and_preprocess_data(data_file_math)
    df_reading = load_reading_data(data_file_reading)
    state_names = get_state_mapping()
    state_names_list = [state_names[i] for i in states[:-1]]
    
    # 2. Estimate transition matrices
    print("\n[2] Estimating transition matrices...")
    matrices, sample_sizes = estimate_transition_matrices(df_math, action_cols, states)
    
    # Aggregate reading matrix
    P_read = aggregate_transition_matrix(df_reading, states)
    
    # 3. Compute spectral radii
    print("\n[3] Computing spectral radii...")
    spectral_radii = {}
    for a, P in matrices.items():
        spectral_radii[a] = compute_spectral_radius(P)
    
    rho_math = np.mean(list(spectral_radii.values()))
    rho_read = compute_spectral_radius(P_read)
    
    print(f"  Reading spectral radius: {rho_read:.3f}")
    print(f"  Mathematics spectral radius (avg): {rho_math:.3f}")
    print(f"  Mathematics spectral radius range: [{min(spectral_radii.values()):.3f}, {max(spectral_radii.values()):.3f}]")
    
    # 4. Bootstrap uncertainty
    print("\n[4] Computing bootstrap uncertainty...")
    ci_lower, ci_upper, rhos_boot = bootstrap_spectral_radius(
        df_math, action_cols, states, n_resamples=1000
    )
    print(f"  95% CI for mathematics spectral radius: [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    # 5. Solve Bellman equation
    print("\n[5] Solving Bellman equation...")
    opt_policy, V = solve_value_iteration(matrices, gamma=0.95, tol=1e-6)
    print(f"  Optimal policy: {opt_policy}")
    
    # 6. Compute holding times
    print("\n[6] Estimating holding times...")
    mu = estimate_holding_times(df_math, states)
    print(f"  Holding times: {mu}")
    
    # 7. Evaluate policies
    print("\n[7] Evaluating policies...")
    # Standard policy: random selection (for comparison)
    random_policy = {s: np.random.choice(list(matrices.keys())) for s in range(len(states) - 1)}
    random_policy[states[-1]] = list(matrices.keys())[0]
    
    # Expected times
    expected_times = {}
    for name, policy in [('Standard', random_policy), ('Optimal', opt_policy)]:
        Q = np.zeros((len(states) - 1, len(states) - 1))
        for s in range(len(states) - 1):
            P = matrices[policy[s]]
            Q[s, :] = P[s, :-1]
        t = expected_time_to_mastery(Q, mu[:-1])
        expected_times[name] = t
        print(f"  {name}: {t}")
    
    # 8. Simulation
    print("\n[8] Running Monte Carlo simulations...")
    std_times = simulate_cohort('standard', matrices, n_students=2000)
    opt_times = simulate_cohort('optimal', matrices, opt_policy, n_students=2000)
    
    print(f"  Standard policy mean time: {np.mean(std_times):.1f}h")
    print(f"  Optimal policy mean time: {np.mean(opt_times):.1f}h")
    print(f"  Reduction: {(1 - np.mean(opt_times)/np.mean(std_times)) * 100:.1f}%")
    
    # 9. Generate figures
    print("\n[9] Generating figures...")
    
    # Survival curves
    fig1 = plot_survival_curves(
        std_times, opt_times,
        save_path=os.path.join(output_dir, 'figures', 'Optimal1.png')
    )
    
    # Spectral radii
    fig2 = plot_spectral_radii(
        spectral_radii,
        save_path=os.path.join(output_dir, 'figures', 'spectral_radii.png')
    )
    
    # Expected times
    fig3 = plot_expected_times(
        expected_times,
        state_names_list,
        save_path=os.path.join(output_dir, 'figures', 'expected_times.png')
    )
    
    # State distribution
    fig4 = plot_state_distribution(
        df_math,
        [state_names[i] for i in states],
        save_path=os.path.join(output_dir, 'figures', 'state_distribution.png')
    )
    
    # 10. Print summary tables
    print("\n[10] Summary tables...")
    print("\n" + "=" * 70)
    print("SUMMARY OF RESULTS")
    print("=" * 70)
    print(f"\nReading: ρ = {rho_read:.3f} (nilpotent)")
    print(f"Mathematics: ρ = {rho_math:.3f} [95% CI: {ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"\nExpected time to mastery (beginner):")
    print(f"  Standard policy: {expected_times['Standard'][0]:.1f}h")
    print(f"  Optimal policy: {expected_times['Optimal'][0]:.1f}h")
    print(f"  Reduction: {(1 - expected_times['Optimal'][0]/expected_times['Standard'][0]) * 100:.1f}%")
    
    print("\n" + "-" * 70)
    print("Optimal policy:")
    print("-" * 70)
    for s, a in opt_policy.items():
        if s < len(states) - 1:
            clean_name = a.split('::')[1].replace('_', ' ')
            print(f"  {state_names[s]}: {clean_name}")
    
    print("\n" + "-" * 70)
    print("Transition matrices:")
    print("-" * 70)
    print("\nReading:")
    print_transition_matrix(P_read, states, state_names)
    
    # Average math matrix
    avg_P = np.mean([matrices[a] for a in matrices.keys()], axis=0)
    print("\nMathematics (average):")
    print_transition_matrix(avg_P, states, state_names)
    
    # 11. Save results
    print("\n[11] Saving results...")
    results = {
        'rho_reading': rho_read,
        'rho_math': rho_math,
        'rho_math_ci': (ci_lower, ci_upper),
        'spectral_radii': spectral_radii,
        'opt_policy': opt_policy,
        'expected_times': expected_times,
        'reduction': (1 - expected_times['Optimal'][0] / expected_times['Standard'][0]) * 100
    }
    
    # Save as numpy file
    np.save(os.path.join(output_dir, 'results.npy'), results)
    
    print("\n" + "=" * 70)
    print("Analysis complete! Results saved to:", output_dir)
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    # Define file paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    
    MATH_FILE = os.path.join(DATA_DIR, 'data_maths_elysa.csv')
    READING_FILE = os.path.join(DATA_DIR, 'data_lecture_elysa.xlsx')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'results')
    
    # Run analysis
    results = run_analysis(MATH_FILE, READING_FILE, OUTPUT_DIR)