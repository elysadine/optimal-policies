"""
Data loading and preprocessing module.

This module handles loading and preprocessing of TaRL data from CSV and Excel files.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


def load_and_preprocess_data(file_path: str) -> Tuple[pd.DataFrame, List[str], List[int]]:
    """
    Load and preprocess mathematics data.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing mathematics data
        
    Returns:
    --------
    df : pd.DataFrame
        Cleaned dataframe with relevant columns
    action_cols : List[str]
        List of action column names
    states : List[int]
        State space [0, 1, 2, 3, 4]
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Identify action columns (TaRL activities)
    action_cols = [c for c in df.columns if 'freq_pos1::' in c]
    
    # State space: 0:Beginner, 1:1-digit, 2:2-digit, 3:Operations, 4:Mastery
    states = [0, 1, 2, 3, 4]
    
    # Extract relevant columns
    required_cols = ['Pretest', 'Final', 'HoursTotal', 'Delta'] + action_cols
    available_cols = [c for c in required_cols if c in df.columns]
    df_clean = df[available_cols].copy()
    
    # Drop rows with missing pretest or final
    df_clean = df_clean.dropna(subset=['Pretest', 'Final'])
    
    # Ensure pretest and final are integers in valid range
    df_clean['Pretest'] = df_clean['Pretest'].astype(int)
    df_clean['Final'] = df_clean['Final'].astype(int)
    df_clean = df_clean[df_clean['Pretest'].isin(states)]
    df_clean = df_clean[df_clean['Final'].isin(states)]
    
    print(f"Loaded {len(df_clean)} student records with {len(action_cols)} actions")
    print(f"State distribution (pretest):\n{df_clean['Pretest'].value_counts().sort_index()}")
    
    return df_clean, action_cols, states


def load_reading_data(file_path: str) -> pd.DataFrame:
    """
    Load reading data from Excel file.
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file containing reading data
        
    Returns:
    --------
    df : pd.DataFrame
        Cleaned dataframe with reading data
    """
    df = pd.read_excel(file_path)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Rename columns for consistency
    # Map Test1, Test2, Test3 to pretest and final
    df['Pretest'] = df['Test1']
    df['Final'] = df['Test3']
    
    # Drop rows with missing data
    df_clean = df.dropna(subset=['Pretest', 'Final'])
    
    # Ensure integer types
    df_clean['Pretest'] = df_clean['Pretest'].astype(int)
    df_clean['Final'] = df_clean['Final'].astype(int)
    
    print(f"Loaded {len(df_clean)} reading student records")
    print(f"State distribution (pretest):\n{df_clean['Pretest'].value_counts().sort_index()}")
    
    return df_clean


def get_state_mapping() -> dict:
    """
    Return mapping between state indices and state names.
    
    Returns:
    --------
    dict: State index -> State name
    """
    return {
        0: 'Beginner',
        1: '1-digit',
        2: '2-digit',
        3: 'Operations',
        4: 'Mastery'
    }