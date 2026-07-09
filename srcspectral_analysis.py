"""
Spectral analysis module.

This module provides functions for computing spectral radii,
fundamental matrices, and expected hitting times.
"""

import numpy as np
from scipy.linalg import eigvals, inv, norm
from typing import Optional, Tuple


def compute_spectral_radius(P: np.ndarray) -> float:
    """
    Compute the spectral radius of the transient submatrix.
    
    Parameters:
    -----------
    P : np.ndarray
        Full transition matrix (n_states x n_states)
        
    Returns:
    --------
    float: Spectral radius of transient submatrix
    """
    Q = P[:-1, :-1]  # Transient submatrix
    if Q.shape[0] == 0:
        return 0.0
    evals = eigvals(Q)
    return float(np.max(np.abs(evals)))


def compute_fundamental_matrix(Q: np.ndarray) -> np.ndarray:
    """
    Compute the fundamental matrix N = (I - Q)^(-1).
    
    Parameters:
    -----------
    Q : np.ndarray
        Transient submatrix
        
    Returns:
    --------
    N : np.ndarray
        Fundamental matrix
    """
    n = Q.shape[0]
    if n == 0:
        return np.array([[0.0]])
    
    # Check if (I - Q) is invertible
    I = np.eye(n)
    try:
        N = inv(I - Q)
    except np.linalg.LinAlgError:
        # If singular, use pseudo-inverse
        N = np.linalg.pinv(I - Q)
    
    return N


def expected_time_to_mastery(Q: np.ndarray, mu: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute expected time to mastery from each transient state.
    
    Parameters:
    -----------
    Q : np.ndarray
        Transient submatrix
    mu : np.ndarray, optional
        Mean holding times for each state. If None, assumes unit holding time.
        
    Returns:
    --------
    np.ndarray: Expected time to mastery from each state
    """
    N = compute_fundamental_matrix(Q)
    if mu is None:
        return N @ np.ones(N.shape[0])
    else:
        return N @ mu


def estimate_holding_times(df: pd.DataFrame, states: List[int]) -> np.ndarray:
    """
    Estimate mean holding times mu_j from data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Contains 'Pretest' and 'HoursTotal' columns
    states : List[int]
        State space
        
    Returns:
    --------
    mu : np.ndarray
        Mean holding times for each state
    """
    n_states = len(states)
    mu = np.zeros(n_states)
    
    for s in states[:-1]:
        subset = df[df['Pretest'] == s]
        if len(subset) > 0:
            # Estimate time spent in state
            # Using HoursTotal as total time; divide by estimated number of transitions
            # This is a simplified approximation
            mu[s] = np.mean(subset['HoursTotal']) / 4.0  # Rough estimate
        else:
            mu[s] = 10.0  # Default holding time
    
    mu[n_states - 1] = 0.0  # Absorbing state
    
    return mu


def bootstrap_spectral_radius(
    df: pd.DataFrame,
    action_cols: List[str],
    states: List[int],
    n_resamples: int = 1000,
    alpha: float = 0.1
) -> Tuple[float, float, np.ndarray]:
    """
    Bootstrap spectral radius estimates.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with 'Pretest', 'Final', and action columns
    action_cols : List[str]
        List of action column names
    states : List[int]
        State space
    n_resamples : int
        Number of bootstrap resamples
    alpha : float
        Laplace smoothing parameter
        
    Returns:
    --------
    ci_lower : float
        2.5% percentile
    ci_upper : float
        97.5% percentile
    rhos : np.ndarray
        All bootstrap spectral radius estimates
    """
    from .transition_matrix import estimate_transition_matrices
    
    rhos = []
    n_obs = len(df)
    
    for i in range(n_resamples):
        # Resample with replacement
        df_boot = df.sample(n=n_obs, replace=True)
        matrices, _ = estimate_transition_matrices(df_boot, action_cols, states, alpha)
        
        # Compute average spectral radius across actions
        rho_sample = []
        for a, P in matrices.items():
            rho_sample.append(compute_spectral_radius(P))
        rhos.append(np.mean(rho_sample))
        
        if (i + 1) % 100 == 0:
            print(f"Bootstrap iteration {i+1}/{n_resamples}")
    
    rhos = np.array(rhos)
    ci_lower = np.percentile(rhos, 2.5)
    ci_upper = np.percentile(rhos, 97.5)
    
    return ci_lower, ci_upper, rhos