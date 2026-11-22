# Study: Exact-$\mathbb{Q}$ Unilateral Prestress, Octagonal-Pore / Checkerboard Bracing & Bar-Cut Localization

- **Evidence Classification (`aska-physical/dock#8`):** **`MODEL` / `CHECKER`** (pure Python 3 standard library exact rational arithmetic; `0` floats; `0` bare `assert`s; not a physical bench measurement).
- **Pinned Upstream Reference:** `aska-physical/fibral-rigidity @ 06520520dff2e956715f18cfa3ba10e7ee83487b` (`PROOF.md:15-22,73-75`; `verify.py:22-54,367-368`).
- **Scope:** Strictly add-only companion study under `studies/prestress-bracing-audit/` (`0` edits to `verify.py`, `length_box.py`, `PROOF.md`, or `results/certificate.json`).

## Exact-$\mathbb{Q}$ Certified Results (`MODEL` / `CHECKER`)

1. **Part A — Unilateral Tension-Only Prestress Certificate (`PROOF.md:75`):**
   - Replaces prose-only tension stability with an exact rational Connelly–Whiteley stress matrix certificate via symmetric Bunch–Kaufman $P L D L^\top P^\top$ Sylvester inertia (`0` floats).
   - Anchored 6-spoke integer-eigenvector cable unit with strictly positive tensions `(1, 10, 11)` certifies exact zero equilibrium residual, strictly positive-definite interior stress matrix $\Omega_{\mathrm{int}}(\omega)$ inertia **`(7, 0, 0)`**, and PSD full matrix **`(12, 0, 1)`** (`PASS`).
   - Compression negative control `(1, -10, -9)` balances static forces yet flips interior inertia to **`(0, 7, 0)`** (`n_neg = 7`, immediate buckling instability, `FAIL`).
2. **Part B — Periodic `F_3(t = 3/5)` Octagonal-Pore Bracing (`verify.py:22-54,367-368`) & Cylindrical Checkerboard Rigidity:**
   - Exact rational Schur-complement periodic homogenization of the real `build_fn_bars(n=3, t=3/5)` cell (`k_intra = k_cross = k_tie = 1`):
     - `none` (`0` ties, `15` bars): exact `C11 = 2/9`, `C12 = 1/18` (matching `PROOF.md` Theorem 1), `C44 = 0`, macroscopic rank `3/6`.
     - `one_per_plane` (`3` ties): `C44 = 34/253 > 0`, **but introduces parasitic axial-shear coupling `C14 = -26/253 != 0`**.
     - **`half_2_opp` (`6` ties = exact half of `full_4`'s `12` ties, orthogonal $\pm 45^\circ$ pair per plane):** achieves full `6/6` macroscopic rank, strictly PD $C_{\mathrm{cell}}$ `(6, 0, 0)`, exact **`C44 = C55 = C66 = 1/4 > 0`**, **zero axial-shear coupling `C14 = 0`**, and identical shear-per-tie efficiency **`C44 / n_ties = 1/24`** to `full_4` (`12` ties, `C44 = 1/2`)!
     - Cutting cross-bar `idx = 12` (`verify.py:367-368`) breaks exact cubic axial symmetry (`C11 != C22`) in pure $\mathbb{Q}$.
   - Exact $4 \times 4$ cylindrical checkerboard Maxwell–Calladine comparison certifies the Bolker–Crapo parity obstruction (`checker_1` & `checker_x` kill global torsion yet leave `1` zigzag mode, closed to `0` floppy modes by `+1` parity bridge or minimal `7`-diagonal Bolker–Crapo tree).
3. **Part C — Sherman–Morrison Rank-1 Tie-Cut Localization Across Stiffness Grading $\lambda = k_2/k_1 \in \{1, 10, 100\}$ (`PROOF.md:15-22`):**
   - Exact-$\mathbb{Q}$ Sherman–Morrison LODF agrees to zero residual with independent direct post-cut re-solve across all $\lambda$; remote-tie max redistribution drops by **`70835025156751/3147372856747`** ($\approx 22.51\times$) at $\lambda = 100$.

## Reproduce & Verify

```bash
cd studies/prestress-bracing-audit
for flags in "-I -B" "-I -O -B" "-I -B -W error"; do
  python3 $flags prestress_localization_audit.py --check prestress_localization_certificate.json
  python3 $flags test_prestress_localization_r5.py
done
```
