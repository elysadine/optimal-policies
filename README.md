# Minimizing the Spectral Radius: A New Principle for Optimal Teaching

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
<!-- Add the arXiv / DOI badge once the identifier exists:
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX) -->

Code and data for the paper:

> **Minimizing the Spectral Radius: A New Principle for Optimal Teaching. Evidence from Madagascar**
> Elysa Marie Alfredine Razafindrafara, Mahefa Abel Razafinirina, Jean Berky Nguala, Solym Mawaki Manou-Abi, Angelo Raherinirina.
> *Journal of Applied Statistics* (under revision, 2026).

## Overview

We model learning as an absorbing controlled Markov process and ask which pedagogical policy brings learners to mastery fastest. The framework has three parts:

- **Spectral radius as a control objective.** The spectral radius ρ(Q) of the transient transition matrix sets the asymptotic rate at which the probability of not having reached mastery decays.
- **Universal optimality under spectral monotonicity.** When the family of policy matrices satisfies spectral monotonicity, a policy that minimizes ρ(Q) also minimizes the expected number of steps to mastery from every starting state (Theorem 1). This is a sufficient condition, not a general property.
- **Semi-Markov extension.** Expected time in hours is obtained from the fundamental matrix and the mean holding times: E_i[T] = Σ_j N_ij μ_j.

## Empirical results

The repository covers two empirical applications.

**1. Linked pupil cohorts (Sections 5.1–5.6).** Pre/post assessments of 1,056 pupils in a TaRL programme (243 in reading, Analamanga; 813 in mathematics, Haute Matsiatra).

| | Spectral radius | Expected time to mastery from Beginner |
|---|---|---|
| Reading | ρ = 0 (nilpotent) | 61.7 h |
| Mathematics | ρ ≈ 0.68 | 87.3 h (baseline) → 44.5 h under the spectral policy π\*, a 49% model-based reduction |

**2. CISCO-level administrative data (Section 5.7).** Level distributions by school district (CISCO) and grade (T2–T5), with pupils not linked across tests:

- **Mathematics, 2025–2026:** 59 CISCOs in 12 regions, about 1.1 million pupils.
- **Reading, 2021–2022:** 45 CISCOs in 9 regions, about 0.85 million pupils.

Transition matrices are estimated per region by constrained least squares for aggregate data, with shrinkage toward the national matrix tuned by leave-one-CISCO-out cross-validation and a CISCO-level cluster bootstrap.

| Ladder | National ρ̂ [95% CI] | Spearman(ρ̂, expected time) across regions | Pairs satisfying strict spectral monotonicity |
|---|---|---|---|
| Reading | 0.33 [0.24, 0.42] | 0.90 | 72% |
| Maths: operations | 0.45 [0.41, 0.68] | 0.49 | 24% |
| Maths: number recognition | 0.33 [0.28, 0.41] | 0.90 | 58% |

At this scale reading is **not** nilpotent: ρ̂ lies between 0.22 and 0.53 in every region. The ρ = 0 result of the first application is therefore specific to that cohort.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── code/
│   ├── <original cohort scripts>      # Sections 5.1–5.6 (see below)
│   └── cisco/                         # Section 5.7
│       ├── prep.py                    # Excel files → tidy CSV (one row per CISCO × grade × test)
│       ├── estimate.py                # estimation, cross-validation, bootstrap, monotonicity check
│       └── figures.py                 # Figures 3 and 4
├── data/
│   ├── <original cohort data>
│   └── cisco/
│       ├── datasets2maths.xlsx        # sheets PRETEST / POSTTEST, 2025–2026
│       └── datasets2reading.xlsx      # pre-test and final-test sheets, 2021–2022
└── results/
    └── cisco/                         # created by the scripts
        ├── maths_cells.csv, reading_cells.csv
        ├── results.json               # all estimates (matrices, ρ̂, CIs, monotonicity counts)
        ├── estimate_log.txt
        └── fig_rho_regions.pdf/png, fig_rho_vs_time.pdf/png
```

> **To complete:** replace the two `<original cohort ...>` lines with the actual file names of your scripts and data for Sections 5.1–5.6, and add a run command for them under "Reproducing the results".

## Installation

```bash
git clone https://github.com/elysadine/optimal-policies.git
cd optimal-policies
pip install -r requirements.txt
```

Tested with Python 3.13, NumPy 2.5, pandas 3.0, SciPy 1.18 and Matplotlib 3.11. Older versions listed in `requirements.txt` should also work.

## Reproducing the results

### Section 5.7 (CISCO-level data)

Run the three scripts in order. They can be run from any directory, and all outputs go to `results/cisco/`.

```bash
python code/cisco/prep.py        # ~5 s
python code/cisco/estimate.py    # ~3 min (B = 500 bootstrap replications)
python code/cisco/figures.py     # Figures 3 and 4
```

For a quick check, pass a smaller number of bootstrap replications. Point estimates are unchanged; only the confidence intervals differ.

```bash
python code/cisco/estimate.py 20
```

The random seed is fixed (`20261014`), so a full run reproduces Table 4 of the paper exactly.

### Sections 5.1–5.6 (linked cohorts)

<!-- Add the commands for the original analysis here. -->

## Method notes for the CISCO data

**Learning ladders.** States are ordered from lowest to highest, and the last one is absorbing (mastery).

| Ladder | Transient states | Mastery |
|---|---|---|
| Reading | Beginner, Letter, Word, Paragraph | Story |
| Maths: operations | None, Addition, +Subtraction, +Multiplication | All four operations |
| Maths: number recognition | Beginner, 1-digit, 2-digit | 3-digit |

The operations ladder assumes a cumulative ordering of skills (addition ⊇ subtraction ⊇ multiplication ⊇ division). The ordering is violated in 25 of 472 records (CISCO × grade × test, 5.3%), where counts are truncated to preserve nesting.

**Estimator.** Let p_c and q_c be the pre- and post-test level distributions in cell c (a CISCO × grade), and n_c its baseline count. For each region, the transition matrix P solves

```
min_P  Σ_c (n_c / Σ n) · ‖q_c − Pᵀ p_c‖²  +  λ ‖P − P_national‖²_F
s.t.   P ≥ 0,  rows sum to 1,  mastery row absorbing
```

The penalty λ is selected by leave-one-CISCO-out cross-validation. It is 0.1 for reading, 0.3 for operations and 0.03 for number recognition. References: Lee, Judge & Zellner (1970); Kalbfleisch & Lawless (1984).

**Time unit.** One step equals one intervention interval: about 8 months in mathematics and 5–6 months in reading. Expected times are therefore given in intervention periods, not hours.

**Limitations.**
- The analysis is ecological: it assumes a common matrix within each region and stable cohort composition between the two tests.
- Test coverage changed markedly in Amoron'i Mania, Betsiboka and Bongolava, where post-test counts are 1.26–1.56 times the pre-test counts.
- The files record no pedagogical actions, so the optimal policy π\* is not estimated from these data.

## Data

The CISCO-level summaries are aggregate counts by school district and grade from the national TaRL programme. They were provided by the Ministry of National Education of Madagascar and contain no individual information. <!-- Confirm the reuse terms with the Ministry; if they differ from MIT, state them here and add a separate data licence. -->

Known issues in the raw files, which the scripts handle:
- Some counts are stored as text with thousands separators (e.g. `"2 686"`).
- In 46 of 472 mathematics records (CISCO × grade × test), the level counts do not sum exactly to the number of pupils tested. Proportions are computed from the level counts.
- In the mathematics file, the CISCOs of Diana reuse row numbers 34–38, which are also used in Analamanga. Rows are identified by region and CISCO name, not by number.

## Citation

```bibtex
@article{razafindrafara2026spectral,
  title   = {Minimizing the Spectral Radius: A New Principle for Optimal Teaching. Evidence from Madagascar},
  author  = {Razafindrafara, Elysa Marie Alfredine and Razafinirina, Mahefa Abel and Nguala, Jean Berky and Manou-Abi, Solym Mawaki and Raherinirina, Angelo},
  journal = {Journal of Applied Statistics},
  year    = {2026},
  note    = {Under revision}
}
```

## License

The code is released under the MIT License. The terms for the data are given in the Data section above.

## Contact

Elysa M. A. Razafindrafara, elysadine@gmail.com
