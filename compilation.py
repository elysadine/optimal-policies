"""generate_figure_data.py
Génère les données pour les figures LaTeX à partir des résultats de l'analyse.
"""

import numpy as np
import pandas as pd
import os

def generate_survival_data(std_times, opt_times, max_time=120):
    """Génère les données de survie pour les figures."""
    time_points = np.arange(0, max_time + 1, 5)
    
    surv_std = [np.mean(std_times > t) for t in time_points]
    surv_opt = [np.mean(opt_times > t) for t in time_points]
    
    # Format pour LaTeX (tableau)
    latex_data = []
    for t, s_std, s_opt in zip(time_points, surv_std, surv_opt):
        latex_data.append(f"{t:.0f} & {s_std:.3f} & {s_opt:.3f} \\\\")
    
    return "\n".join(latex_data)

def generate_latex_table(data, headers, caption, label):
    """Génère un tableau LaTeX."""
    table = f"""
\\begin{{table}}[ht]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{{'c' * len(headers)}}}
\\toprule
{headers}
\\midrule
{data}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    return table

def main():
    # Simulation des données (à remplacer par vos données réelles)
    np.random.seed(42)
    std_times = np.random.gamma(2, 20, 2000)
    opt_times = np.random.gamma(1.5, 15, 2000)
    
    # Génération des données
    survival_data = generate_survival_data(std_times, opt_times)
    
    # Sauvegarde
    with open('survival_data.tex', 'w') as f:
        f.write(survival_data)
    
    print("Données générées avec succès!")

if __name__ == "__main__":
    main()