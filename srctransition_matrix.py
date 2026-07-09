"""
Transition matrix estimation module.

This module provides functions for estimating transition matrices
from empirical data with Laplace smoothing.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List


def estimate_transition_matrices(
    df: pd.DataFrame,
    action_cols: List[str],
    states: List[int],
    alpha: float = 0.1
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[int, int]]]:
    """
    Estimate transition matrices with Laplace smoothing.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Contains 'Pretest', 'Final', and action frequency columns
    action_cols : List[str]
        Column names for TaRL activities
    states : List[int]
        State space [0, 1, 2, 3, 4]
    alpha : float
        Laplace smoothing parameter
        
    Returns:
    --------
    matrices : Dict[str, np.ndarray]
        Action -> transition matrix (5x5)
    sample_sizes : Dict[str, Dict[int, int]]
        Action -> (state -> number of observations)
    """
    n_states = len(states)
    matrices = {}
    sample_sizes = {}
    
    for action in action_cols:
        P = np.zeros((n_states, n_states))
        n_samples = {}
        
        for s in states[:-1]:  # transient states only
            # Students at state s who used this action
            subset = df[(df['Pretest'] == s) & (df[action] > 0)]
            n = len(subset)
            n_samples[s] = n
            
            if n > 0:
                counts = subset['Final'].value_counts()
                for next_s in range(n_states):
                    # Laplace smoothing: (count + alpha) / (n + alpha * n_states)
                    P[s, next_s] = (counts.get(next_s, 0) + alpha) / (n + alpha * n_states)
            else:
                # No data: use uniform distribution with smoothing
                # This is conservative and avoids overconfident estimates
                P[s, :] = 1.0 / n_states
                n_samples[s] = 0
        
        # Absorbing state: mastery
        P[n_states - 1, n_states - 1] = 1.0
        
        matrices[action] = P
        sample_sizes[action] = n_samples
    
    return matrices, sample_sizes


def aggregate_transition_matrix(
    df: pd.DataFrame,
    states: List[int],
    alpha: float = 0.1
) -> np.ndarray:
    """
    Compute aggregate transition matrix (averaged across all actions).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Contains 'Pretest', 'Final' columns
    states : List[int]
        State space
    alpha : float
        Laplace smoothing parameter
        
    Returns:
    --------
    P : np.ndarray
        Aggregate transition matrix (5x5)
    """
    n_states = len(states)
    P = np.zeros((n_states, n_states))
    
    for s in states[:-1]:
        subset = df[df['Pretest'] == s]
        n = len(subset)
        
        if n > 0:
            counts = subset['Final'].value_counts()
            for next_s in range(n_states):
                P[s, next_s] = (counts.get(next_s, 0) + alpha) / (n + alpha * n_states)
        else:
            P[s, :] = 1.0 / n_states
    
    P[n_states - 1, n_states - 1] = 1.0
    
    return P


def get_transient_matrix(P: np.ndarray) -> np.ndarray:
    """
    Extract transient submatrix from full transition matrix.
    
    Parameters:
    -----------
    P : np.ndarray
        Full transition matrix (n_states x n_states)
        
    Returns:
    --------
    Q : np.ndarray
        Transient submatrix ((n_states-1) x (n_states-1))
    """
    return P[:-1, :-1]


def print_transition_matrix(P: np.ndarray, states: List[int], state_names: dict = None):
    """
    Pretty print transition matrix.
    
    Parameters:
    -----------
    P : np.ndarray
        Transition matrix
    states : List[int]
        State indices
    state_names : dict, optional
        Mapping from state index to name
    """
    if state_names is None:
        state_names = {i: f"s{i}" for i in states}
    
    print("Transition Matrix:")
    print("-" * 60)
    print("From\\To", end="")
    for s in states:
        print(f"  {state_names[s]:>10}", end="")
    print()
    
    for i, s in enumerate(states):
        print(f"{state_names[s]:>10}", end="")
        for j in range(len(states)):
            print(f"  {P[i, j]:>10.3f}", end="")
        print()