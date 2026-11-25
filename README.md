# fibral-rigidity

**Continuum carbon-fiber truss homogenization, area-grading optimality, and specific-rigidity limits on the bridged cross-polytope (`incentre-lattices`) framework.**

Part of [Foundation-0](https://github.com/Foundation-0), extending the unit-spring rigidity theory of [Foundation-0/incentre-lattices](https://github.com/Foundation-0/incentre-lattices) ([`results/rigidity-nd/PROOF.md`](https://github.com/Foundation-0/incentre-lattices/blob/main/results/rigidity-nd/PROOF.md)) to physical axial carbon-fiber reinforced polymer (CFRP) struts, area-graded tow allocations, $45^\circ$ octagonal-pore tension bracing, and primary-datasheet tensile/modulus benchmarks.

---

## Core Question

When the abstract unit springs (`k = 1`) of the $n$-dimensional bridged cross-polytope framework $F_n(t)$ are replaced by real unidirectional carbon-fiber composite struts ($k_b = E_1 A_b / L_b$), what are the exact homogenized elastic moduli, where does internal joint relaxation waste stiffness, how close can pure or tension-braced carbon-fiber realizations get to fundamental theoretical stiffness bounds, and where do Euler buckling vs. tensile limits govern?

---

## Summary of Checked Results

Every claim is labelled strictly per the [Foundation-0](https://github.com/Foundation-0) repository standard (`PROVEN` = written proof + symbolic check; `VERIFIED` = deterministic numerical check with must-fail controls; `KNOWN` = cited literature/datasheet):

| Result | Key Formula / Value | Label | Check |
| :--- | :--- | :--- | :--- |
| **Continuum-bar moduli $F_n(t,\alpha)$** | $C_{11}/C_{12} = (n-1)(1 + \frac{n-2}{\lambda})$, $\nu_{100} = \frac{\lambda}{(n-1)(n-2)+(2n-3)\lambda} < \frac{1}{2n-3}$, $C_{44}=0$, where $\lambda = \frac{\alpha t}{\sqrt{2}(1-t)}$ | `PROVEN` ($n \ge 2$); `VERIFIED` ($n = 2..7$) | `python3 verify.py` |
| **Equal-area relaxation penalty** | At $n=3, t=2-\sqrt{2}, \alpha=1$: $K / K_{\max} = \frac{2(3+2\sqrt{2})}{15} = 0.777124$ (**22.29% bulk stiffness loss**) | `PROVEN` & `VERIFIED` | `python3 verify.py` |
| **Universal bulk-modulus saturation** | Area ratio $\alpha_K^* = \sqrt{2}(n-1)$ ($2\sqrt{2} \approx 2.8284$ in 3D) gives $K_n = \frac{\bar{\rho} E_1}{n^2}$ (**100% of Cauchy–Maxwell bound**) for **all** $t \in (0,1)$ with 15 bars/cell | `PROVEN` ($n \ge 2$); `VERIFIED` ($n = 2..7$) | `python3 verify.py` |
| **Uniaxial $[100]$ stiffness optimum** | Interior optimum $\alpha_E^* = \frac{2\sqrt{6}}{3} \approx 1.6330$ reaches $\frac{E_{[100]}}{\bar{\rho} E_1} = \frac{3+2\sqrt{2}}{3(7+2\sqrt{6})} = 0.163275$ (**+46.95%** over octet truss $\frac{1}{9}$) | `PROVEN` & `VERIFIED` | `python3 verify.py` |
| **Octagonal-pore $45^\circ$ shear ties ($F_3^+$)** | 12 non-intersecting pore chords/cell (exact length $2$ at $t=2-\sqrt{2}$); isotropic at $\alpha=2\sqrt{2}, \beta_{\text{iso}}=0.362323$ with $E_{\text{iso}} = 0.12805\,\bar{\rho} E_1$, $K_{\text{iso}} = 0.10891\,\bar{\rho} E_1$ | `VERIFIED` | `python3 verify.py` |
| **Same-material bound check (honest limit)** | No pin-jointed truss beats the closed-cell plate Hashin–Shtrikman (HS) bound of the *same* solid ($0.1281 < 0.5115$ at $\bar{\rho}=5\%$, [Berger et al. 2017](https://pubmed.ncbi.nlm.nih.gov/28219078/)) | `PROVEN` / `KNOWN` | Must-fail #3 & #4 |
| **CFRP specific-rigidity vs. metal HS bounds** | At $\bar{\rho}=5\%$ ($V_f=0.60$), isotropic-braced **T1100G** ($16.13$), **M55J** ($25.63$), and **K13D2U** ($40.01\text{ MPa}/(\text{kg/m}^3)$) beat the **closed-cell HS upper bounds** of Al 6061-T6 ($13.08$), Ti-6Al-4V ($13.18$), and Steel 4340 ($13.36$) | `VERIFIED` | `python3 verify.py` |
| **Compressive kink-band & SP-8007 tube limit** | Pin-jointed ($K=1$, $r_g=r/2$) hollow tubes ($\eta=0.80$) give $4.56\times$ Euler gain **only** while $\sigma \le \min(\sigma_{\text{kink}}, \gamma_{\text{SP-8007}}\sigma_{\text{shell}})$; using SHA-256-pinned representative UD compressive engineering assumptions in `data/carbon_fibers.json` (`1200/1470/1500/750/350 MPa`, not primary filament-datasheet lower bounds) and axisymmetric shell estimate ($E_2=8\text{ GPa}, \nu_{12}=0.30$, isotropic SP-8007 $\gamma$, no $G_{LT}$; thin-shell $R/t \ge 10$ holds at $\eta=0.99$ with $R/t=99.5$ and not at $\eta=0.80$ with $R/t=4.5$), at $\bar{\rho}=5\%$ K13D2U is kink-band capped at **$350\text{ MPa}$** ($\text{effective gain } 1.0\times$, vs. $2,225.54\text{ MPa}$ ROM tensile), and ultra-thin $\eta=0.99$ tubes are shell-limited ($226.68\text{ MPa} < \sigma_{\text{kink}}$) | `VERIFIED` *(conditional on hashed UD allowables)* | `python3 verify.py` (Must-fail #8) |

---

## CFRP Benchmark Table ($\bar{\rho} = 5\%$, Unidirectional $V_f = 0.60$ in Epoxy)

Fiber properties are loaded from [`data/carbon_fibers.json`](data/carbon_fibers.json) using official [Torayca Carbon Fiber](https://www.toraycma.com/products/carbon-fiber/) ([T300](https://www.toraycma.com/wp-content/uploads/T300-Data-Sheet.pdf), [Technical Manual](https://www.toraycma.com/wp-content/uploads/Torayca-Technical-Manual_4-28-2020.pdf), [T1100G](https://www.toraycma.com/wp-content/uploads/T1100G-Data-Sheet.pdf)) and [Mitsubishi DIALEAD Pitch Fiber](https://www.m-chemical.co.jp/carbon-fiber/en/product/tow/) ([K13D2U Datasheet](https://www.hajime.com.tw/Upload/product_201805210812374.pdf)) specifications:

| Material / Architecture | $E_{\text{constituent}}$ (GPa) | $\rho_{\text{lattice}}$ ($\text{g/cm}^3$) | Graded $E_{[100]} / \rho$ ($\frac{\text{MPa}}{\text{kg/m}^3}$) | Isotropic $E_{\text{iso}} / \rho$ ($\frac{\text{MPa}}{\text{kg/m}^3}$) | Graded $K^* / \rho$ ($\frac{\text{MPa}}{\text{kg/m}^3}$) | Compressive $\bar{\rho}_{\text{crit}}$ ($\eta=0.8$, $\sigma_{\text{kink}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Al 6061-T6 (HS Plate Upper Bound)** | 68.9 | 0.1350 | — | 13.08 | 8.75 | — |
| **Ti-6Al-4V (HS Plate Upper Bound)** | 113.8 | 0.2215 | — | 13.18 | 8.95 | — |
| **AISI 4340 Steel (HS Plate Upper Bound)** | 205.0 | 0.3925 | — | 13.36 | 8.43 | — |
| **Torayca T300 CFRP** ($V_f=0.60$) | 139.4 | 0.0768 | 13.93 | 11.62 | 10.08 | 0.350% |
| **Torayca T700S CFRP** ($V_f=0.60$) | 139.4 | 0.0780 | 13.72 | 11.44 | 9.93 | 0.429% |
| **Torayca T1100G CFRP** ($V_f=0.60$, $\sigma_f=7.0\text{ GPa}$) | 195.8 | 0.0777 | **19.34** | **16.13** | **14.00** | 0.312% |
| **Torayca M55J CFRP** ($V_f=0.60$) | 325.4 | 0.0813 | **30.72** | **25.63** | **22.24** | 0.094% |
| **Mitsubishi K13D2U CFRP** ($V_f=0.60$) | 562.4 | 0.0900 | **47.96** | **40.01** | **34.72** | **0.025%** |

*(At $\bar{\rho}=5\%$, all five grades sit above their compressive Euler crossover, so hollow-tube compressive gain saturates at $1.0\times$ at the SHA-256-pinned representative UD compressive kink-band allowables in `data/carbon_fibers.json`: `1200 / 1470 / 1500 / 750 / 350 MPa` for `T300 / T700S / T1100G / M55J / K13D2U`, respectively. Tensile-crossover $\bar{\rho}_{\text{crit}}$ values remain recorded alongside compressive values in `results/certificate.json`, and `hollow_tube_eta08_sp8007_thin_shell_valid=false` explicitly flags that $R/t = 4.5 < 10$ at $\eta = 0.80$.)*

---

## Repository Layout & Reproduction

- [`PROOF.md`](PROOF.md) — Full mathematical derivations, scope boundaries, and prior-art citations.
- [`data/carbon_fibers.json`](data/carbon_fibers.json) — Primary-sourced fiber and reference alloy properties with SHA-256 integrity check.
- [`verify.py`](verify.py) — Deterministic checker (`sympy` symbolic proofs for $n=2..7$, `numpy` 60-case coordinate-space Schur-complement truss homogenizer, CFRP/HS benchmarks, and 8 must-fail falsification controls).
- [`results/certificate.json`](results/certificate.json) — Committed machine-verifiable output certificate.

### Run Locally

```bash
python3 -m pip install numpy sympy
python3 verify.py
```

`python3 verify.py` recomputes every symbolic and numerical result from scratch and executes all 8 rejecting controls. A failed check raises even under `python3 -O`. The numerical residual is stored as a pass or fail under 1e-9, not as the raw digits. The run still rejects a discrepancy of 1e-11 or more. The command fails if a check raises or if the certificate differs.

---

## Open Research Tracks & Contribution Rules

1. **Finite-joint overlap & Timoshenko beam bending (`task`):** Add rigid/bonded CFRP node bending stiffness ($EI/L^3$) to quantify $\bar{\rho}^2$ shear scaling without diagonal ties and correct solid volume fraction for joint-collar overlaps above $\bar{\rho} > 0.10$.
2. **Pre-tensioned unilateral cable mechanics (`task`):** Formulate exact pre-strain bounds $\varepsilon_0 \in (0, \varepsilon_{\text{ult}})$ on the 12 octagonal-pore carbon-fiber filaments so compression struts remain below Euler/kink-band limits while tension ties stay taut under arbitrary multiaxial shear.
3. **Folding-to-locked CFRP deployment (`task`):** Couple the continuous `m = 1` folding path of [Foundation-0/incentre-lattices](https://github.com/Foundation-0/incentre-lattices) with snap-taut carbon-fiber tension cables that lock at target deployment angle.

### Contributing & AI Disclosure (Foundation-0 Charter)

Follow [Foundation-0/f0's JOIN.md](https://github.com/Foundation-0/f0/blob/main/JOIN.md) and [CONTRIBUTING.md](https://github.com/Foundation-0/f0/blob/main/CONTRIBUTING.md):
- Open one pull request per result, disclose AI assistance and the model used, and name the person signing for the work.
- Never edit `verify.py` to weaken a check alongside a claimed result; include must-fail controls for new verifiers.
- Label every claim honestly (`PROVEN`, `VERIFIED`, `KNOWN`, or `CONJECTURE`), and never call "not found" "new".

---

## Credits & Disclosure

- **Maintainer & Human Sign-Off:** Shahid Arifuddin (`@abu-aska`)
- **Foundational Framework (`incentre-lattices`):** Ahraz Arifuddin (`@ahrazzle`)
- **AI Assistance:** Prepared by Google Gemini (autonomous research & verification agent) working for Shahid Arifuddin (`@abu-aska`), following [Foundation-0](https://github.com/Foundation-0) principles 0.2–0.6. test for superior rigidity limits.
