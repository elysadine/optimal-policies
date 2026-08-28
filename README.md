# Minimizing the Spectral Radius: A New Principle for Optimal Teaching

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

## 📋 Overview

This repository contains the code, data, and supplementary materials for the paper:

**"Minimizing the Spectral Radius: A New Principle for Optimal Teaching. Evidence from Madagascar"**

Elysa Marie Alfredine RAZAFINDRAFARA, Mahefa Abel RAZAFINIRINA, Jean Berky NGUALA, Solym Mawaki MANOU-ABI, Angelo RAHERINIRINA

### Abstract

We introduce a statistical framework for modeling and optimizing student learning trajectories using absorbing controlled Markov processes. The central goal is to minimize the expected time required for a learner to reach mastery under a sequence of pedagogical interventions.

**Key innovations:**
- **Spectral radius as control objective**: We demonstrate that minimizing $\rho(Q)$ is structurally related to maximizing the exponential rate of convergence to mastery
- **Universal optimality**: Under spectral monotonicity, the policy that minimizes $\rho(Q)$ is optimal for every possible initial student distribution
- **Semi-Markov extension**: We bridge cognitive steps and real instructional hours via $\mathbb{E}_i[T] = \sum_j N_{ij}\mu_j$

**Empirical validation**: Using longitudinal data from a TaRL program in Madagascar (1,056 students), we show:
- Reading: $\rho = 0$ (nilpotent structure)
- Mathematics: $\rho \approx 0.68$ (stochastic friction)
- Estimated gain: 49% reduction in expected mastery time (87.3h → 44.5h)

## 📁 Repository Structure
