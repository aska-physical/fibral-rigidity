# Continuum Bar Homogenization, Area-Grading Optimality, and Carbon-Fiber Rigidity Limits on the Bridged Cross-Polytope Truss

**Evidence labels** (per [Foundation-0/incentre-lattices](https://github.com/Foundation-0/incentre-lattices)):
- `PROVEN`: closed-form mathematical proof below (and symbolically checked in `verify.py` via `sympy`).
- `VERIFIED`: deterministic finite-dimensional numerical check (`verify.py` via `numpy`, tolerance `< 1e-11`).
- `KNOWN`: established mechanics or materials result from cited primary literature/datasheets.

---

## 1. Geometry and Continuum Axial-Bar Stiffness Model

On the periodic cell $Q = [-1, 1]^n$ of volume $V_{\text{cell}} = 2^n$ (matching [incentre-lattices/results/rigidity-nd/PROOF.md](https://github.com/Foundation-0/incentre-lattices/blob/main/results/rigidity-nd/PROOF.md)), place $2n$ joints at $\mathbf{p}(i, s) = s t \mathbf{e}_i$ for $i = 1, \dots, n$, $s \in \{-1, +1\}$, and $0 < t < 1$. At the incentre point, $t = c = 2 - \sqrt{2}$ and $R = 1 - c = \sqrt{2} - 1$.

Instead of assigning equal unit spring constants (`k = 1`) to all bars, model every member as a unidirectional carbon-fiber reinforced polymer (CFRP) axial bar of longitudinal modulus $E_1$:
1. **Intra-cell cross-polytope bars** ($N_{\text{intra}} = 2n(n-1)$ per cell): length $L_1(t) = t\sqrt{2}$, cross-sectional area $A_1$, axial spring constant
   $$k_1 = \frac{E_1 A_1}{t\sqrt{2}}.$$
2. **Inter-cell axial cross bars $X_i$** ($N_{\text{cross}} = n$ per cell): length $L_2(t) = 2(1 - t)$, cross-sectional area $A_2 = \alpha A_1$ ($\alpha > 0$), axial spring constant
   $$k_2 = \frac{E_1 \alpha A_1}{2(1 - t)}.$$

Define the dimensionless spring-stiffness ratio
$$\lambda(t, \alpha) = \frac{k_2}{k_1} = \frac{\alpha t}{\sqrt{2}(1 - t)}.$$
At the incentre realization $t = 2 - \sqrt{2}$, $L_1(c) = L_2(c) = 2(\sqrt{2} - 1)$, so $\lambda(c, \alpha) = \alpha$. Away from $t = c$, equal-area struts ($\alpha = 1$) do **not** have equal spring constants.

In the thin-strut (first-order non-overlapping) regime, the solid volume fraction is
$$\bar{\rho}(n, t, \alpha) = \frac{N_{\text{intra}} A_1 L_1 + N_{\text{cross}} A_2 L_2}{2^n} = \frac{2n A_1 \left[(n - 1)t\sqrt{2} + \alpha(1 - t)\right]}{2^n}.$$

---

## 2. Theorem 1 (`PROVEN`, $n \ge 2$; `VERIFIED`, $n = 2 \dots 7$): Continuum-Bar Elastic Constants

**Statement.** For any $n \ge 2$, $t \in (0, 1)$, and stiffness ratio $\lambda = k_2 / k_1 > 0$, relaxing all joints of $F_n(t, \alpha)$ under a homogeneous strain $\boldsymbol{\varepsilon}$ yields strain energy density
$$W(\boldsymbol{\varepsilon}) = \frac{1}{2} C_{11} \sum_{i=1}^n \varepsilon_{ii}^2 + C_{12} \sum_{1 \le i < j \le n} \varepsilon_{ii}\varepsilon_{jj},$$
with all shear moduli identically zero ($C_{44} = 0$) and
$$C_{11}(n, \lambda) = 2^{2-n} k_1 \frac{\lambda (n - 1)(n - 2 + \lambda)}{(n - 2 + 2\lambda)(n - 1 + \lambda)}, \qquad C_{12}(n, \lambda) = 2^{2-n} k_1 \frac{\lambda^2}{(n - 2 + 2\lambda)(n - 1 + \lambda)}.$$
Consequently, for $n \ge 3$:
$$\frac{C_{11}}{C_{12}} = (n - 1)\left(1 + \frac{n - 2}{\lambda}\right), \qquad \nu_{100}(n, \lambda) = \frac{C_{12}}{C_{11} + (n - 2)C_{12}} = \frac{\lambda}{(n - 1)(n - 2) + (2n - 3)\lambda}.$$
In particular, $\nu_{100}(n, \lambda)$ is strictly increasing in $\lambda$ with supremum $\lim_{\lambda \to \infty} \nu_{100}(n, \lambda) = \frac{1}{2n - 3}$ ($\frac{1}{3}$ in $3\text{D}$), reducing to $\frac{1}{n^2 - n - 1}$ at $\lambda = 1$.

**Proof.** By Lemma 1 of [incentre-lattices](https://github.com/Foundation-0/incentre-lattices/blob/main/results/rigidity-nd/PROOF.md) with $c \mapsto t$ and $R \mapsto 1 - t$, choosing $a_i = 0$ and $\sigma_{ij} = -2t\varepsilon_{ij}$ relaxes all off-diagonal shear strains to zero energy via rigid-body cross-polytope rotations ($C_{44} = 0$). Writing $e_i = \varepsilon_{ii}$ and $y_i = b_i^i + t e_i$, $(1 - t)e_i - b_i^i = e_i - y_i$, so the cell energy is
$$E_{\text{cell}}(e; y) = k_1 \sum_{1 \le i < j \le n} (y_i + y_j)^2 + 2\lambda k_1 \sum_{i=1}^n (e_i - y_i)^2.$$
Its Hessian in $y$ is $2 k_1 \left[(n - 2 + 2\lambda)I + J\right]$, which is symmetric positive definite for all $n \ge 2, \lambda > 0$. Stationarity $\partial E_{\text{cell}}/\partial y_m = 0$ gives
$$(n - 2 + 2\lambda) y_m + \sum_{j=1}^n y_j = 2\lambda e_m.$$
Summing over $m = 1, \dots, n$ yields $(2n - 2 + 2\lambda)\sum_j y_j = 2\lambda \sum_j e_j$, i.e., $\sum_j y_j = \frac{\lambda}{n - 1 + \lambda}\sum_j e_j$. Thus
$$e_m - y_m^* = \frac{n - 2}{n - 2 + 2\lambda} e_m + \frac{\lambda}{(n - 2 + 2\lambda)(n - 1 + \lambda)} \sum_{j=1}^n e_j.$$
By Euler's homogeneous quadratic identity at $\nabla_y E_{\text{cell}} = 0$, $E_{\min}(e) = \frac{1}{2}\sum_{m=1}^n e_m (\partial E_{\text{cell}}/\partial e_m)|_{y^*} = 2\lambda k_1 \sum_{m=1}^n e_m (e_m - y_m^*)$, giving
$$W(e) = 2^{-n} E_{\min}(e) = 2^{1-n} k_1 \left[ \frac{\lambda(n - 2)}{n - 2 + 2\lambda} \sum_{i=1}^n e_i^2 + \frac{\lambda^2}{(n - 2 + 2\lambda)(n - 1 + \lambda)} \left(\sum_{i=1}^n e_i\right)^2 \right].$$
Differentiating with respect to $e_1^2$ and $e_1 e_2$ gives $C_{11}$ and $C_{12}$, since $(n - 2)(n - 1 + \lambda) + \lambda = (n - 1)(n - 2 + \lambda)$. Substituting into $\nu_{100} = C_{12}/(C_{11} + (n - 2)C_{12})$ yields the stated rational formula, whose $\lambda$-derivative $\frac{(n-1)(n-2)}{[(n-1)(n-2)+(2n-3)\lambda]^2} > 0$ for all $n \ge 3$. $\square

---

## 3. Theorem 2 (`PROVEN`, $n \ge 2$): Relaxation Loss vs. Universal Cauchy–Maxwell Saturation

**Statement.** For any pin-jointed periodic bar framework in $\mathbb{R}^n$ of solid volume fraction $\bar{\rho}$ and bar modulus $E_1$, the bulk modulus $K_n = \frac{C_{11} + (n - 1)C_{12}}{n}$ satisfies the Cauchy–Maxwell (Voigt dilatation) bound
$$K_n \le K_n^{\max} = \frac{\bar{\rho} E_1}{n^2}.$$
For $F_n(t, \alpha)$:
1. $K_n(t, \alpha) = K_n^{\max}$ **if and only if** the cross-to-intra strut area ratio is
   $$\alpha_K^* = \sqrt{2}(n - 1) \qquad (\text{independent of } t \in (0, 1)),$$
   at which $b_i^i = 0$ under hydrostatic strain ($\nu_{100} = \frac{6 - \sqrt{2}}{17} \approx 0.2698$ at $n = 3, t = c$).
2. At the equal-area incentre realization ($n = 3, t = 2 - \sqrt{2}, \alpha = 1$), internal joint relaxation reduces the bulk modulus to
   $$\frac{K_3(c, 1)}{K_3^{\max}} = \frac{2(3 + 2\sqrt{2})}{15} \approx 0.777124 \qquad (\text{a } 22.29\% \text{ stiffness penalty}).$$
3. In $3\text{D}$ at $t = 2 - \sqrt{2}$, the specific uniaxial modulus $E_{[100]} / (\bar{\rho} E_1)$ vanishes as $\alpha \to 0$ and $\alpha \to \infty$ (series bottleneck) and attains its unique global maximum at
   $$\alpha_E^* = \frac{2\sqrt{6}}{3} \approx 1.6330, \qquad \max_{\alpha > 0} \frac{E_{[100]}}{\bar{\rho} E_1} = \frac{3 + 2\sqrt{2}}{3(7 + 2\sqrt{6})} \approx 0.163275,$$
   exceeding the Deshpande–Fleck–Ashby octet-truss value $\frac{1}{9} \approx 0.111111$ by $+46.95\%$.

**Proof.** Under affine dilatation $\boldsymbol{\varepsilon} = \varepsilon I$ with $u = 0$, every bar stretches by $\delta_b = \varepsilon L_b$, giving upper-bound energy density $\frac{1}{2}\varepsilon^2 V_{\text{cell}}^{-1} \sum_b (E_1 A_b / L_b) L_b^2 = \frac{1}{2}\bar{\rho} E_1 \varepsilon^2$; comparing with $\frac{1}{2}n^2 K_n \varepsilon^2$ gives $K_n \le \bar{\rho} E_1 / n^2$, with equality iff $u = 0$ ($b_i^i = 0$) is already in joint equilibrium. From Theorem 1,
$$K_n = 2^{2-n} k_1 \frac{(n - 1)\lambda}{n(n - 1 + \lambda)}.$$
At $e_1 = \dots = e_n = \varepsilon$, $y_m^* = \frac{\lambda}{n - 1 + \lambda}\varepsilon$, so $b_m^m = y_m^* - t\varepsilon = 0 \iff \frac{\lambda}{n - 1 + \lambda} = t \iff \lambda = \frac{(n - 1)t}{1 - t}$. Substituting $\lambda = \frac{\alpha t}{\sqrt{2}(1 - t)}$ gives $\alpha = \sqrt{2}(n - 1)$ uniquely. Evaluating $K_3 / (\bar{\rho} E_1 / 9)$ at $n = 3, t = 2 - \sqrt{2}, \alpha = 1$ yields $\frac{4 \cdot 9}{3 \cdot 15 \cdot 4(\sqrt{2}-1)^2} = \frac{2}{15(3 - 2\sqrt{2})} = \frac{2(3 + 2\sqrt{2})}{15}$. Finally, at $n = 3, t = c$ ($\lambda = \alpha$), $C_{11} - \frac{2C_{12}^2}{C_{11}+C_{12}} = \frac{\alpha}{3\alpha + 2} k_1$, while $\bar{\rho} E_1 = \frac{3}{8}(\alpha + 4) k_1 (2R)^2$, giving $\frac{E_{[100]}}{\bar{\rho} E_1} = \frac{2}{3(\sqrt{2}-1)^2 (3\alpha + 14 + 8/\alpha)}$. By AM–GM, $3\alpha + 8/\alpha \ge 4\sqrt{6}$ with equality uniquely at $\alpha_E^* = \sqrt{8/3} = \frac{2\sqrt{6}}{3}$. $\square$

---

## 4. Shear Bracing (`VERIFIED`), Buckling Envelope, and Specific-Rigidity Limits

1. **Octagonal-pore $45^\circ$ carbon-fiber tension/compression ties ($F_3^+$):** Adding 4 planar ties per coordinate plane across the open octagonal pores of the `4.8.8` net (connecting $\mathbf{p}(i,-)+2\mathbf{e}_i \leftrightarrow \mathbf{p}(j,-)+2\mathbf{e}_j$, etc.; exact length $L_{\text{tie}} = \sqrt{2}(2 - t) = 2$ at $t = 2 - \sqrt{2}$) locks the rotational zero modes with full cubic $O_h$ symmetry (`C44 > 0`). At $\alpha = 2\sqrt{2}$ and $\beta_{\text{iso}} = A_{\text{diag}}/A_{\text{intra}} \approx 0.362323$, the truss is elastically isotropic ($2 C_{44} = C_{11} - C_{12}$ to $< 10^{-11}$), with $E_{\text{iso}}/(\bar{\rho} E_1) = 0.128052$ (below the Gurtner–Durand isotropic truss bound $1/6 \approx 0.166667$) and $K_{\text{iso}}/(\bar{\rho} E_1) = 0.108908$.
2. **Comparison against Hashin–Shtrikman bounds (`PROVEN` bound vs. `VERIFIED` specific properties):** No pin-jointed truss exceeds the closed-cell plate Hashin–Shtrikman (HS) bound of the *same* isotropic solid ([Berger, Wadley & McMeeking 2017](https://pubmed.ncbi.nlm.nih.gov/28219078/)). However, unidirectional CFRP struts ($V_f = 0.60$ in epoxy) align covalent carbon stiffness purely along member axes: at $\bar{\rho} = 5\%$, **Torayca T1100G** ($16.13$), **Torayca M55J** ($25.63$), and **Mitsubishi DIALEAD K13D2U** ($40.01\text{ MPa}/(\text{kg/m}^3)$) isotropic-braced trusses exceed the **closed-cell HS upper bounds** of Aluminum 6061-T6 ($13.08$), Titanium Ti-6Al-4V ($13.18$), and AISI 4340 Steel ($13.36\text{ MPa}/(\text{kg/m}^3)$).
3. **Compressive buckling, kink-band, and NASA SP-8007 thin-shell envelope (`VERIFIED`, conditional on hashed UD compressive allowables in `data/carbon_fibers.json`):** For a pin-jointed ($K = 1$) solid circular strut of outer radius $r$, radius of gyration $r_g = r/2$ gives Euler buckling stress $\sigma_{\text{Euler}} = \frac{\pi^2 E_1}{4}(r/L)^2$, reaching a target stress cap $\sigma_{\text{cap}}$ at $(r/L)_{\text{crit}} = \frac{2}{\pi}\sqrt{\sigma_{\text{cap}} / E_1}$. Replacing the solid rod by an equal-area hollow tube of inner/outer radius ratio $\eta \in (0, 1)$ multiplies the second moment of area by $(1+\eta^2)/(1-\eta^2)$ ($4.56\times$ at $\eta = 0.80$), but axial compressive capacity cannot mirror the composite rule-of-mixtures tensile strength $\sigma_T$: it is capped by
   $$\sigma_{\text{allow}} = \min\!\left(\sigma_{\text{Euler}},\; \sigma_{\text{kink}},\; \gamma_{\text{SP-8007}}\,\sigma_{\text{shell}}\right), \qquad \sigma_{\text{shell}} = \frac{\sqrt{E_1 E_2}}{\sqrt{3(1 - \nu_{12}\nu_{21})}}\,\frac{t}{R_{\text{mid}}},$$
   where $R_{\text{mid}}/t = \frac{1+\eta}{2(1-\eta)}$, $\nu_{21} = \nu_{12} E_2 / E_1$, and $\gamma_{\text{SP-8007}} = 1 - 0.901\bigl(1 - \exp(-\sqrt{R_{\text{mid}}/t}/16)\bigr)$ (axisymmetric orthotropic estimate with $E_2 = 8\text{ GPa}, \nu_{12} = 0.30$, isotropic SP-8007 knockdown, no $G_{LT}$ shear correction; classical thin-shell theory assumes $R_{\text{mid}}/t \ge 10$, which holds at $\eta = 0.99$ where $R_{\text{mid}}/t = 99.5$ and does not hold at $\eta = 0.80$ where $R_{\text{mid}}/t = 4.5$). Under the SHA-256-pinned representative UD $V_f = 0.60$ compressive kink-band engineering assumptions in `data/carbon_fibers.json` (`1200 / 1470 / 1500 / 750 / 350 MPa`, all strictly below ROM $\sigma_T$), the hollow-tube compressive crossover drops well below the tensile-mirrored crossover (`4.00%` solid / `0.88%` hollow for T1100G tensile vs. `0.312%` hollow for T1100G compressive and `0.025%` hollow for K13D2U compressive). At $\bar{\rho} = 5\%$, all five grades are kink-band governed (e.g., K13D2U at $350\text{ MPa}$ vs. $2,225.54\text{ MPa}$ ROM tensile, saturating effective tube gain at $1.0\times$), while ultra-thin $\eta = 0.99$ tubes ($R_{\text{mid}}/t = 99.5$) switch to SP-8007 local shell buckling ($226.68\text{ MPa} < 350\text{ MPa}$, effective gain $0.6476 < 1.0$). Pre-tensioning the octagonal-pore ties keeps shear braces in pure tension, avoiding compressive microbuckling.

---

## 5. Scope and Prior Art

- **Scope:** Linear small-strain periodic homogenization (`q = 0`), first-order strut volume fraction (neglecting joint overlap volumes at higher $\bar{\rho}$), and longitudinal rule-of-mixtures CFRP axial stiffness ($V_f = 0.60$). Real composite nodes require finite-element joint-fillet and compressive-kink-band validation before hardware claims.
- **Prior art (`KNOWN`):** Base framework geometry and unit-spring formulas are from [Foundation-0/incentre-lattices](https://github.com/Foundation-0/incentre-lattices) (Ahraz Arifuddin, 2026) and C. S. Borcea & I. Streinu, *Math. Mech. Solids* 26, 199 (2021). Stretching-dominated octet-truss benchmarks follow V. S. Deshpande, N. A. Fleck & M. F. Ashby, *J. Mech. Phys. Solids* 49, 1747 (2001) ([PDF reference](https://people.ub.edu/parkhs/Papers/heJMPS2017.pdf)) and CFRP octet lattices in L. Dong & H. N. G. Wadley, *Compos. Sci. Technol.* (2015) ([Semantic Scholar](https://www.semanticscholar.org/paper/Mechanical-properties-of-carbon-fiber-composite-Dong-Wadley/648bc3f379f186f0f339cbb2df15fe04cbc83cbd)). Isotropic plate HS limits follow J. B. Berger, H. N. G. Wadley & R. M. McMeeking, *Nature* 543, 533 (2017) ([PubMed](https://pubmed.ncbi.nlm.nih.gov/28219078/)).$
