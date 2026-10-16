# Numerical Reproducibility: Minimax Generalization Lower Bounds in QML

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.30+-orange.svg)](https://pennylane.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides the self-contained numerical simulation scripts and benchmark datasets supporting the paper:

> **A Tight Lower Bound on Generalization in Quantum Machine Learning via Dynamical Lie Algebra Dimension**  
> **Author:** Ayad Alhusseiny (*College of Science, University of Kerbala*)  
> **Correspondence:** `ayad.a@uokerbala.edu.iq`

---

## Overview

Orbit dimension, not parameter count, bounds quantum generalization. For continuous Parameterized Quantum Circuits (PQCs) on $n$ qubits, the minimax estimation risk obeys the lower bound
$$\mathcal{R}(d_{\mathrm{eff}}, M) \;\geq\; \Omega\!\left(\sqrt{\frac{d_{\mathrm{eff}}}{M}}\right),$$
where $d_{\mathrm{eff}} = \dim(\mathcal{O}_O)$ denotes the real dimension of the observable's adjoint orbit under the dynamical Lie group. This fundamental scaling reveals that statistical capacity is dictated by Riemannian orbit volume rather than Euclidean parameter counts or abstract Hilbert space dimensions.

This repository provides reproducible benchmarks across $n \in \{1, 2, 3, 4\}$ qubits for hardware-efficient ansätze under rank-1 projective observables $O = |\mathbf{0}\rangle\langle\mathbf{0}|$ ($d_{\mathrm{eff}}^{\mathrm{proj}} = 2(2^n-1) \in \{2, 6, 14, 30\}$):
1. **Exponential Threshold Growth:** Verifies that the sample complexity threshold $M^* = \sigma^2 d_{\mathrm{eff}} / (4\varepsilon^2)$ scales exponentially with qubit count $n$.
2. **Estimation Risk Phase Transition:** Demonstrates that training at $M_{\mathrm{high}} \approx 6M^*$ reduces Root Mean Squared Error (RMSE) by $\geq 2\times$ relative to $M_{\mathrm{low}} \approx M^*/3$. This empirical reduction establishes the $\Omega(\sqrt{d_{\mathrm{eff}}/M})$ lower bound.
3. **Underparameterized Rate Transition:** Confirms that in shallow circuits ($p < d_{\mathrm{eff}}$), estimation capacity collapses to the parameter rank rate $\Omega(\sqrt{p/M})$, validating Corollary 3.

---

## Repository Contents

* [`experiments_and_figures.py`](experiments_and_figures.py): Standalone, fully vectorized PennyLane script that trains the hardware-efficient circuits, evaluates test RMSE across multiple seeds, prints Table 2, and plots Figure 1 (`fig_experiments.pdf` and `fig_experiments.png`).
* [`experiments_results.json`](experiments_results.json): Precomputed raw benchmark data containing evaluation runs across all independent random seeds with exact means and standard deviations.
* [`requirements.txt`](requirements.txt): Minimal Python package dependencies.

---

## Experimental Setup and Parameters

| $n$ (Qubits) | $d_{\mathrm{eff}}^{\mathrm{proj}}$ | Depth $L$ | Parameters $p = Ln$ | Noise $\sigma$ | Target $\varepsilon$ | Threshold $M^*$ | $M_{\mathrm{low}}$ | $M_{\mathrm{high}}$ | Independent Seeds |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 2 | 2 | 2 | 0.25 | 0.08 | **4.9** | 2 | 30 | 5 |
| **2** | 6 | 3 | 6 | 0.25 | 0.08 | **14.6** | 3 | 146 | 5 |
| **3** | 14 | 5 | 15 | 0.25 | 0.08 | **34.2** | 14 | 341 | 4 |
| **4** | 30 | 8 | 32 | 0.03 | 0.01 | **67.5** | 27 | 675 | 4 |

> **Note on $n=4$ Noise Calibration:** For $n=4$, the rank-1 projector expectation value scales as $\mathcal{O}(2^{-n}) \approx 0.0625$. The noise level is calibrated to $\sigma = 0.03, \varepsilon = 0.01$ to preserve the signal-to-noise ratio $\mathrm{SNR} \approx 2.1$ and the ratio $\varepsilon/\sigma \approx 1/3$, identical to the $n=1,2,3$ regimes.

---

## Installation & Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Ayad-Alhusseiny/qml-minimax-lower-bounds.git
cd qml-minimax-lower-bounds
```

### 2. Set up a Python environment
Python 3.9 or higher is recommended:
```bash
python -m venv venv

# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install requirements
```bash
pip install -r requirements.txt
```

---

## Reproducing Table 2 and Figure 1

Run the standalone reproduction script:
```bash
python experiments_and_figures.py
```

### Expected Output
1. **Console Output (Table 2):**
   ```text
   ========================================================================================
   TABLE 2: EMPIRICAL VALIDATION OF MINIMAX LOWER BOUND
   ========================================================================================
   n   d_eff   M*     M_low: RMSE (mean ± std)       M_high: RMSE (mean ± std)      Ratio
   ----------------------------------------------------------------------------------------
   1   2       4.9    M=2   : 0.093 ± 0.060          M=30  : 0.028 ± 0.028          3.28×
   2   6       14.6   M=3   : 0.278 ± 0.021          M=146 : 0.041 ± 0.020          6.69×
   3   14      34.2   M=14  : 0.138 ± 0.018          M=341 : 0.069 ± 0.005          2.01×
   4   30      67.5   M=27  : 0.048 ± 0.011          M=675 : 0.017 ± 0.004          2.86×
   ========================================================================================
   ```
2. **Generated Figures:**
   * `fig_experiments.pdf`: High-resolution vector graphic containing Panels (a) Sample Complexity Scaling, (b) Phase Transition Curves, and (c) Underparameterized Verification.
   * `fig_experiments.png`: Rasterized graphic suitable for web viewing.

**Execution Time:** Approximately 6–8 minutes on a standard modern multicore CPU.

---

## Citation

If you find this work or implementation useful in your research, please cite:

```bibtex
@article{alhusseiny2026tight,
  title   = {A Tight Lower Bound on Generalization in Quantum Machine Learning via Dynamical Lie Algebra Dimension},
  author  = {Alhusseiny, Ayad},
  journal = {Preprint},
  year    = {2026}
}
```

---

## License

This project is licensed under the MIT License; see the [LICENSE](LICENSE) file for details.
