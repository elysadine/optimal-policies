"""
MDP solver module.

This module provides functions for solving Markov Decision Processes
via value iteration and policy iteration.
"""

import numpy as np
from typing import Dict, Tuple, List


def solve_value_iteration(
    matrices: Dict[str, np.ndarray],
    gamma: float = 0.95,
    tol: float = 1e-6,
    max_iter: int = 10000
) -> Tuple[Dict[int, str], np.ndarray]:
    """
    Solve Bellman optimality equation using value iteration.
    
    Parameters:
    -----------
    matrices : Dict[str, np.ndarray]
        Action -> transition matrix (n_states x n_states)
    gamma : float
        Discount factor
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum number of iterations
        
    Returns:
    --------
    policy : Dict[int, str]
        Optimal action for each state
    V : np.ndarray
        Optimal value function
    """
    # Get first matrix to determine dimensions
    first_matrix = next(iter(matrices.values()))
    n_states = first_matrix.shape[0]
    n_transient = n_states - 1
    
    V = np.zeros(n_states)
    policy = {}
    iteration = 0
    
    for iteration in range(max_iter):
        V_new = np.zeros(n_states)
        delta = 0
        
        for s in range(n_transient):  # transient states only
            q_values = {}
            for a, P in matrices.items():
                q_values[a] = 1 + gamma * np.dot(P[s, :], V)
            
            best_a = min(q_values, key=q_values.get)
            V_new[s] = q_values[best_a]
            delta = max(delta, abs(V_new[s] - V[s]))
        
        V = V_new
        
        if delta < tol:
            print(f"Value iteration converged in {iteration+1} iterations")
            break
    
    # Extract optimal policy
    for s in range(n_transient):
        q_values = {}
        for a, P in matrices.items():
            q_values[a] = 1 + gamma * np.dot(P[s, :], V)
        policy[s] = min(q_values, key=q_values.get)
    
    # Absorbing state policy (no action needed)
    policy[n_states - 1] = list(matrices.keys())[0]
    
    return policy, V


def solve_bellman(
    matrices: Dict[str, np.ndarray],
    gamma: float = 0.95,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Tuple[Dict[int, str], np.ndarray]:
    """
    Alias for solve_value_iteration.
    """
    return solve_value_iteration(matrices, gamma, tol, max_iter)


def find_spectral_optimal_policy(
    matrices: Dict[str, np.ndarray],
    states: List[int]
) -> Tuple[Dict[int, str], Dict[str, float]]:
    """
    Find policy that minimizes spectral radius (greedy approach).
    
    Parameters:
    -----------
    matrices : Dict[str, np.ndarray]
        Action -> transition matrix
    states : List[int]
        State indices
        
    Returns:
    --------
    policy : Dict[int, str]
        Greedy spectral policy for each state
    spectral_radii : Dict[str, float]
        Spectral radius for each action
    """
    from .spectral_analysis import compute_spectral_radius
    
    policy = {}
    spectral_radii = {}
    
    # Compute spectral radius for each action
    for a, P in matrices.items():
        spectral_radii[a] = compute_spectral_radius(P)
    
    # For each state, choose action that minimizes local friction (diagonal entry)
    n_transient = len(states) - 1
    for s in range(n_transient):
        best_a = min(matrices.keys(), key=lambda a: matrices[a][s, s])
        policy[s] = best_a
    
    # Absorbing state
    policy[states[-1]] = list(matrices.keys())[0]
    
    return policy, spectral_radii


def evaluate_policy(
    policy: Dict[int, str],
    matrices: Dict[str, np.ndarray],
    states: List[int]
) -> np.ndarray:
    """
    Evaluate a policy to compute expected times to mastery.
    
    Parameters:
    -----------
    policy : Dict[int, str]
        Policy mapping state -> action
    matrices : Dict[str, np.ndarray]
        Action -> transition matrix
    states : List[int]
        State indices
        
    Returns:
    --------
    expected_times : np.ndarray
        Expected time to mastery from each state
    """
    from .spectral_analysis import expected_time_to_mastery
    
    n_states = len(states)
    Q = np.zeros((n_states - 1, n_states - 1))
    
    for s in range(n_states - 1):
        P = matrices[policy[s]]
        Q[s, :] = P[s, :-1]  # Transient part
    
    return expected_time_to_mastery(Q)