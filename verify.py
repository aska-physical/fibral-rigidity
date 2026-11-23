#!/usr/bin/env python3
"""Deterministic verification suite for Foundation-0/fibral-rigidity (SymPy + NumPy)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "carbon_fibers.json"
CERT_PATH = ROOT / "results" / "certificate.json"

class CheckFailure(RuntimeError):
    """A failed check. Raised as a normal exception so python -O cannot strip it."""


def require(ok: bool, message: str) -> None:
    if not ok:
        raise CheckFailure(message)


def sanitize(value: Any) -> Any:
    """Drop the sign on a zero so two machines write the same certificate."""
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, float) and value == 0.0:
        return 0.0
    return value


NUMERICAL_BOUND = 1e-9




def build_fn_bars(
    n: int, t: float, E1: float, A_intra: float, A_cross: float, A_diag: float = 0.0
) -> tuple[np.ndarray, list[tuple[int, int, np.ndarray, float]]]:
    """Build nodal coordinates and periodic bars on Q = [-1, 1]^n (volume 2^n)."""
    coords = np.zeros((2 * n, n), dtype=float)
    for i in range(n):
        coords[2 * i, i] = -t
        coords[2 * i + 1, i] = +t

    bars: list[tuple[int, int, np.ndarray, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            for si_idx in (0, 1):
                for sj_idx in (0, 1):
                    bars.append((2 * i + si_idx, 2 * j + sj_idx, np.zeros(n), E1 * A_intra))

    for i in range(n):
        shift = np.zeros(n)
        shift[i] = 2.0
        bars.append((2 * i + 1, 2 * i + 0, shift, E1 * A_cross))

    if A_diag > 0.0:
        # Four 45-degree octagonal-pore ties per coordinate plane (i < j); length = 2 at t = 2 - sqrt(2)
        for i in range(n):
            for j in range(i + 1, n):
                s_neg = np.zeros(n); s_neg[i] = -2.0; s_neg[j] = +2.0
                s_pos = np.zeros(n); s_pos[i] = +2.0; s_pos[j] = +2.0
                bars.append((2 * i + 0, 2 * j + 0, s_neg, E1 * A_diag))
                bars.append((2 * j + 1, 2 * i + 1, s_neg.copy(), E1 * A_diag))
                bars.append((2 * i + 1, 2 * j + 0, s_pos, E1 * A_diag))
                bars.append((2 * j + 1, 2 * i + 0, s_pos.copy(), E1 * A_diag))

    return coords, bars


def homogenize_periodic_truss(
    n: int, coords: np.ndarray, bars: list[tuple[int, int, np.ndarray, float]], cell_edge: float = 2.0
) -> tuple[np.ndarray, float]:
    """Coordinate-space Schur-complement homogenization under periodic q = 0 relaxation."""
    ndof = coords.shape[0] * n
    shear_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    n_strain = n + len(shear_pairs)
    cell_vol = cell_edge ** n

    B_u = np.zeros((len(bars), ndof), dtype=float)
    B_e = np.zeros((len(bars), n_strain), dtype=float)
    k_vec = np.zeros(len(bars), dtype=float)
    ea_vol = 0.0

    for b_idx, (u, v, shift, EA) in enumerate(bars):
        vec = (coords[v] + shift) - coords[u]
        length = float(np.linalg.norm(vec))
        n_hat = vec / length
        k_vec[b_idx] = EA / length
        ea_vol += EA * length
        B_u[b_idx, v * n : (v + 1) * n] += n_hat
        B_u[b_idx, u * n : (u + 1) * n] -= n_hat
        for i in range(n):
            B_e[b_idx, i] = length * (n_hat[i] ** 2)
        for s_idx, (i, j) in enumerate(shear_pairs):
            B_e[b_idx, n + s_idx] = length * n_hat[i] * n_hat[j]

    K_uu = (B_u.T * k_vec) @ B_u
    K_ue = (B_u.T * k_vec) @ B_e
    K_ee = (B_e.T * k_vec) @ B_e
    U_relax, _, _, _ = np.linalg.lstsq(K_uu, -K_ue, rcond=1e-13)
    C_cell = K_ee + B_e.T @ ((k_vec[:, None]) * (B_u @ U_relax))
    return 0.5 * (C_cell + C_cell.T) / cell_vol, ea_vol / cell_vol


def hashin_shtrikman_upper_bounds(Es: float, nu_s: float, phi: float) -> tuple[float, float, float]:
    """Exact 3D Hashin-Shtrikman isotropic solid-void upper bounds (K_HS, G_HS, E_HS)."""
    Ks = Es / (3.0 * (1.0 - 2.0 * nu_s))
    Gs = Es / (2.0 * (1.0 + nu_s))
    K_hs = Ks * (4.0 * Gs * phi) / (4.0 * Gs + 3.0 * Ks * (1.0 - phi))
    G_hs = Gs * phi * (9.0 * Ks + 8.0 * Gs) / (9.0 * Ks + 8.0 * Gs + 6.0 * (Ks + 2.0 * Gs) * (1.0 - phi))
    E_hs = 9.0 * K_hs * G_hs / (3.0 * K_hs + G_hs)
    return K_hs, G_hs, E_hs


def run_symbolic_checks() -> dict[str, Any]:
    n_sym = sp.Symbol("n", integer=True, positive=True)
    lam, k1, u_pos, alpha, E1, A1 = sp.symbols("lam k1 u_pos alpha E1 A1", positive=True)
    t = u_pos / (1 + u_pos)

    denom = (n_sym - 2 + 2 * lam) * (n_sym - 1 + lam)
    C11_expr = sp.Integer(2) ** (2 - n_sym) * k1 * lam * (n_sym - 1) * (n_sym - 2 + lam) / denom
    C12_expr = sp.Integer(2) ** (2 - n_sym) * k1 * lam**2 / denom

    require(sp.simplify(C11_expr.subs({lam: 1, k1: 1}) - 4 * (n_sym - 1) ** 2 / (n_sym**2 * sp.Integer(2) ** n_sym)) == 0, 'sp.simplify(C11_expr.subs({lam: 1, k1: 1}) - 4 * (n_sym - 1) ** 2 / (n_sym**2 * sp.Integer(2) ** n_sym)) == 0')
    require(sp.simplify(C12_expr.subs({lam: 1, k1: 1}) - 4 / (n_sym**2 * sp.Integer(2) ** n_sym)) == 0, 'sp.simplify(C12_expr.subs({lam: 1, k1: 1}) - 4 / (n_sym**2 * sp.Integer(2) ** n_sym)) == 0')
    require(sp.simplify(C11_expr / C12_expr - (n_sym - 1) * (n_sym - 2 + lam) / lam) == 0, 'sp.simplify(C11_expr / C12_expr - (n_sym - 1) * (n_sym - 2 + lam) / lam) == 0')

    for n_val in range(2, 8):
        e_s = sp.symbols(f"e0:{n_val}", real=True)
        y_s = sp.symbols(f"y0:{n_val}", real=True)
        E_cell = k1 * sum((y_s[i] + y_s[j]) ** 2 for i in range(n_val) for j in range(i + 1, n_val)) + 2 * lam * k1 * sum((e_s[i] - y_s[i]) ** 2 for i in range(n_val))
        sol_y = sp.solve([sp.diff(E_cell, y_s[i]) for i in range(n_val)], y_s, dict=True)
        W_min = sp.simplify(E_cell.subs(sol_y[0]) / (2**n_val))
        require(sp.simplify(sp.diff(W_min, e_s[0], 2) - C11_expr.subs(n_sym, n_val)) == 0, 'sp.simplify(sp.diff(W_min, e_s[0], 2) - C11_expr.subs(n_sym, n_val)) == 0')
        require(sp.simplify(sp.diff(sp.diff(W_min, e_s[0]), e_s[1]) - C12_expr.subs(n_sym, n_val)) == 0, 'sp.simplify(sp.diff(sp.diff(W_min, e_s[0]), e_s[1]) - C12_expr.subs(n_sym, n_val)) == 0')

    nu_expr = sp.simplify(C12_expr / (C11_expr + (n_sym - 2) * C12_expr))
    nu_expected = lam / ((n_sym - 1) * (n_sym - 2) + (2 * n_sym - 3) * lam)
    require(sp.simplify(nu_expr - nu_expected) == 0, 'sp.simplify(nu_expr - nu_expected) == 0')
    nu_3d = nu_expected.subs(n_sym, 3)
    require(sp.simplify(sp.diff(nu_3d, lam) - 2 / (2 + 3 * lam) ** 2) == 0, 'sp.simplify(sp.diff(nu_3d, lam) - 2 / (2 + 3 * lam) ** 2) == 0')
    require(sp.limit(nu_3d, lam, sp.oo) == sp.Rational(1, 3), 'sp.limit(nu_3d, lam, sp.oo) == sp.Rational(1, 3)')
    nu_3d_optK = sp.simplify(nu_3d.subs(lam, 2 * sp.sqrt(2)))
    require(sp.simplify(nu_3d_optK - (6 - sp.sqrt(2)) / 17) == 0, 'sp.simplify(nu_3d_optK - (6 - sp.sqrt(2)) / 17) == 0')

    K_expr = sp.simplify((C11_expr + (n_sym - 1) * C12_expr) / n_sym)
    k1_bar = E1 * A1 / (t * sp.sqrt(2))
    k2_bar = E1 * alpha * A1 / (2 * (1 - t))
    lam_bar = sp.simplify(k2_bar / k1_bar)
    rho_bar = sp.simplify(2 * n_sym * A1 * ((n_sym - 1) * t * sp.sqrt(2) + alpha * (1 - t)) / (sp.Integer(2) ** n_sym))
    K_cont = sp.simplify(K_expr.subs({k1: k1_bar, lam: lam_bar}))
    K_bound = sp.simplify(rho_bar * E1 / (n_sym**2))
    gap = sp.together(K_bound - K_cont)
    require(sp.simplify(gap.subs(alpha, sp.sqrt(2) * (n_sym - 1))) == 0, 'sp.simplify(gap.subs(alpha, sp.sqrt(2) * (n_sym - 1))) == 0')

    for n_val in range(2, 8):
        quot_n = sp.cancel(gap.subs(n_sym, n_val) / (alpha - sp.sqrt(2) * (n_val - 1)) ** 2, extension=[sp.sqrt(2)])
        num_q, den_q = sp.fraction(quot_n)
        if sp.Poly(sp.expand(den_q), u_pos, alpha, E1, A1).LC() < 0:
            num_q, den_q = -num_q, -den_q
        require(all(c > 0 for c in sp.Poly(sp.expand(num_q), u_pos, alpha, E1, A1).coeffs()), 'all(c > 0 for c in sp.Poly(sp.expand(num_q), u_pos, alpha, E1, A1).coeffs())')
        require(all(c > 0 for c in sp.Poly(sp.expand(den_q), u_pos, alpha, E1, A1).coeffs()), 'all(c > 0 for c in sp.Poly(sp.expand(den_q), u_pos, alpha, E1, A1).coeffs())')

    u_inc = sp.sqrt(2)
    ratio_eq = sp.simplify((K_cont / K_bound).subs({n_sym: 3, u_pos: u_inc, alpha: 1}))
    require(sp.simplify(ratio_eq - sp.Rational(2, 15) * (3 + 2 * sp.sqrt(2))) == 0, 'sp.simplify(ratio_eq - sp.Rational(2, 15) * (3 + 2 * sp.sqrt(2))) == 0')

    C11_3d = C11_expr.subs({n_sym: 3, k1: k1_bar, lam: lam_bar})
    C12_3d = C12_expr.subs({n_sym: 3, k1: k1_bar, lam: lam_bar})
    E100_3d = sp.simplify(C11_3d - 2 * C12_3d**2 / (C11_3d + C12_3d))
    f_E100 = sp.simplify((E100_3d / (rho_bar.subs(n_sym, 3) * E1)).subs(u_pos, u_inc))
    require(sp.limit(f_E100, alpha, 0) == 0 and sp.limit(f_E100, alpha, sp.oo) == 0, 'sp.limit(f_E100, alpha, 0) == 0 and sp.limit(f_E100, alpha, sp.oo) == 0')
    alpha_E_opt = 2 * sp.sqrt(6) / 3
    require(sp.simplify(sp.diff(f_E100, alpha).subs(alpha, alpha_E_opt)) == 0, 'sp.simplify(sp.diff(f_E100, alpha).subs(alpha, alpha_E_opt)) == 0')
    norm_E100_max = sp.simplify(f_E100.subs(alpha, alpha_E_opt))
    require(sp.simplify(norm_E100_max - (3 + 2 * sp.sqrt(2)) / (3 * (7 + 2 * sp.sqrt(6)))) == 0, 'sp.simplify(norm_E100_max - (3 + 2 * sp.sqrt(2)) / (3 * (7 + 2 * sp.sqrt(6)))) == 0')

    return {
        "symbolic_dimensions_checked": list(range(2, 8)),
        "C11_over_C12_formula": "(n - 1) * (1 + (n - 2) / lambda)",
        "nu_100_formula": "lambda / ((n - 1)*(n - 2) + (2*n - 3)*lambda)",
        "nu_100_supremum_3d": "1/3",
        "nu_100_at_bulk_optimum_3d_exact": "(6 - sqrt(2)) / 17",
        "nu_100_at_bulk_optimum_3d_value": round(float(nu_3d_optK.evalf()), 12),
        "optimal_bulk_area_ratio_formula": "sqrt(2) * (n - 1)",
        "optimal_bulk_area_ratio_3d": round(float((2 * sp.sqrt(2)).evalf()), 12),
        "incentre_equal_area_K_over_Kmax_exact": "2*(3 + 2*sqrt(2))/15",
        "incentre_equal_area_K_over_Kmax_value": round(float(ratio_eq.evalf()), 12),
        "incentre_equal_area_relaxation_loss_pct": round(float((1 - ratio_eq).evalf() * 100.0), 8),
        "incentre_equal_area_E100_over_rhoE1": round(float(f_E100.subs(alpha, 1).evalf()), 12),
        "incentre_optK_E100_over_rhoE1": round(float(f_E100.subs(alpha, 2 * sp.sqrt(2)).evalf()), 12),
        "optimal_E100_area_ratio_3d_exact": "2*sqrt(6)/3",
        "optimal_E100_area_ratio_3d_value": round(float(alpha_E_opt.evalf()), 12),
        "incentre_max_E100_over_rhoE1_exact": "(3 + 2*sqrt(2)) / (3*(7 + 2*sqrt(6)))",
        "incentre_max_E100_over_rhoE1_value": round(float(norm_E100_max.evalf()), 12),
    }


def run_numerical_truss_checks() -> dict[str, Any]:
    c_inc = 2.0 - math.sqrt(2.0)
    max_err = 0.0
    cases = 0

    for n in range(2, 7):
        alpha_opt = math.sqrt(2.0) * (n - 1)
        for t in (0.35, c_inc, 0.72):
            for alpha in (0.5, 1.0, alpha_opt, 4.0):
                E1, A1 = 300.0, 1.0e-3
                coords, bars = build_fn_bars(n, t, E1, A1, alpha * A1, A_diag=0.0)
                C_mat, ea_vol = homogenize_periodic_truss(n, coords, bars)
                rho_bar = ea_vol / E1
                k1 = E1 * A1 / (t * math.sqrt(2.0))
                k2 = E1 * (alpha * A1) / (2.0 * (1.0 - t))
                lam = k2 / k1
                denom = (n - 2 + 2.0 * lam) * (n - 1 + lam)
                scale = (2.0 ** (2 - n)) * k1
                c11_th = scale * lam * (n - 1) * (n - 2 + lam) / denom
                c12_th = scale * (lam**2) / denom
                err = max(abs(C_mat[0, 0] - c11_th), abs(C_mat[0, 1] - c12_th), float(np.max(np.abs(C_mat[n:, :]))))
                max_err = max(max_err, err)
                require(err < 1e-11, 'err < 1e-11')
                K_num = (C_mat[0, 0] + (n - 1) * C_mat[0, 1]) / n
                K_max = rho_bar * E1 / (n**2)
                require(K_num <= K_max + 1e-12, 'K_num <= K_max + 1e-12')
                if abs(alpha - alpha_opt) < 1e-14:
                    require(abs(K_num - K_max) < 1e-11, 'abs(K_num - K_max) < 1e-11')
                cases += 1

    alpha_opt_3d = 2.0 * math.sqrt(2.0)
    braced_profiles: list[dict[str, Any]] = []
    for beta in (0.10, 0.25, 0.50):
        coords, bars = build_fn_bars(3, c_inc, 1.0, 1.0, alpha_opt_3d, A_diag=beta)
        C_mat, rho_bar = homogenize_periodic_truss(3, coords, bars)
        C11, C12, C44 = float(C_mat[0, 0]), float(C_mat[0, 1]), float(C_mat[3, 3])
        require(C44 > 0.0 and np.allclose(np.diag(C_mat[:3, :3]), C11, atol=1e-12) and np.allclose(np.diag(C_mat[3:, 3:]), C44, atol=1e-12), 'C44 > 0.0 and np.allclose(np.diag(C_mat[:3, :3]), C11, atol=1e-12) and np.allclose(np.diag(C_mat[3:, 3:]), C44, atol=1e-12)')
        K_eff = (C11 + 2.0 * C12) / 3.0
        E_100 = C11 - 2.0 * (C12**2) / (C11 + C12)
        E_111 = 1.0 / (1.0 / (9.0 * K_eff) + 1.0 / (3.0 * C44))
        braced_profiles.append({
            "beta_diag_over_intra": beta,
            "C11_over_rhoE1": round(C11 / rho_bar, 10),
            "C12_over_rhoE1": round(C12 / rho_bar, 10),
            "C44_over_rhoE1": round(C44 / rho_bar, 10),
            "K_over_rhoE1": round(K_eff / rho_bar, 10),
            "E100_over_rhoE1": round(E_100 / rho_bar, 10),
            "E111_over_rhoE1": round(E_111 / rho_bar, 10),
            "zener_anisotropy": round(2.0 * C44 / (C11 - C12), 10),
        })

    lo, hi = 0.01, 5.0
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        coords, bars = build_fn_bars(3, c_inc, 1.0, 1.0, alpha_opt_3d, A_diag=mid)
        C_m, _ = homogenize_periodic_truss(3, coords, bars)
        if 2.0 * C_m[3, 3] - (C_m[0, 0] - C_m[0, 1]) < 0.0:
            lo = mid
        else:
            hi = mid
    beta_iso = 0.5 * (lo + hi)
    coords, bars = build_fn_bars(3, c_inc, 1.0, 1.0, alpha_opt_3d, A_diag=beta_iso)
    C_iso, rho_iso = homogenize_periodic_truss(3, coords, bars)
    C11_i, C12_i, C44_i = float(C_iso[0, 0]), float(C_iso[0, 1]), float(C_iso[3, 3])
    require(abs(2.0 * C44_i - (C11_i - C12_i)) < 1e-11, 'abs(2.0 * C44_i - (C11_i - C12_i)) < 1e-11')
    E_iso = C11_i - 2.0 * (C12_i**2) / (C11_i + C12_i)
    K_iso = (C11_i + 2.0 * C12_i) / 3.0
    require(E_iso / rho_iso <= (1.0 / 6.0) + 1e-12, 'E_iso / rho_iso <= (1.0 / 6.0) + 1e-12')
    discrepancy_within_bound = bool(max_err < NUMERICAL_BOUND)
    require(discrepancy_within_bound, "numerical discrepancy left the 1e-9 bound")

    return {
        "coordinate_homogenization_cases_checked": cases,
        "max_discrepancy_within_1e-9": discrepancy_within_bound,
        "braced_3d_profiles": braced_profiles,
        "isotropic_braced_point": {
            "alpha_cross_over_intra": round(alpha_opt_3d, 10),
            "beta_diag_over_intra": round(beta_iso, 10),
            "E_iso_over_rhoE1": round(E_iso / rho_iso, 10),
            "G_iso_over_rhoE1": round(C44_i / rho_iso, 10),
            "K_iso_over_rhoE1": round(K_iso / rho_iso, 10),
            "nu_iso": round(C12_i / (C11_i + C12_i), 10),
            "gurtner_durand_isotropic_truss_bound_E_over_rhoE1": round(1.0 / 6.0, 10),
        },
    }


def run_cfrp_benchmark_checks(num_res: dict[str, Any]) -> dict[str, Any]:
    raw = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    Em = float(raw["matrix_reference"]["E_GPa"])
    rho_m = float(raw["matrix_reference"]["rho_g_cm3"])
    Vf = float(raw["matrix_reference"]["fiber_volume_fraction_Vf"])
    rho_bar_ref = 0.05
    c_val = 2.0 - math.sqrt(2.0)
    L_bar = 2.0 * (math.sqrt(2.0) - 1.0)

    coords_opt, bars_opt = build_fn_bars(3, c_val, 1.0, 1.0, 2.0 * math.sqrt(2.0), A_diag=0.0)
    C_opt, rho_opt = homogenize_periodic_truss(3, coords_opt, bars_opt)
    norm_K_opt = float((C_opt[0, 0] + 2.0 * C_opt[0, 1]) / (3.0 * rho_opt))
    norm_E100_opt = float((C_opt[0, 0] - 2.0 * (C_opt[0, 1] ** 2) / (C_opt[0, 0] + C_opt[0, 1])) / rho_opt)
    norm_E_iso = float(num_res["isotropic_braced_point"]["E_iso_over_rhoE1"])
    norm_K_iso = float(num_res["isotropic_braced_point"]["K_iso_over_rhoE1"])

    metal_hs: list[dict[str, Any]] = []
    max_m_E, max_m_K = 0.0, 0.0
    for m in raw["reference_isotropic_solids"]:
        K_hs, _, E_hs = hashin_shtrikman_upper_bounds(float(m["Es_GPa"]), float(m["nu_s"]), rho_bar_ref)
        rho_lat = rho_bar_ref * float(m["rho_s_g_cm3"])
        s_E, s_K = E_hs / rho_lat, K_hs / rho_lat
        max_m_E, max_m_K = max(max_m_E, s_E), max(max_m_K, s_K)
        metal_hs.append({
            "id": m["id"],
            "rho_lattice_g_cm3": round(rho_lat, 5),
            "K_HS_GPa": round(K_hs, 6),
            "E_HS_GPa": round(E_hs, 6),
            "specific_K_HS_MPa_per_kg_m3": round(s_K, 4),
            "specific_E_HS_MPa_per_kg_m3": round(s_E, 4),
        })

    fibers: list[dict[str, Any]] = []
    eta_tube = 0.80
    tube_gain = (1.0 + eta_tube**2) / (1.0 - eta_tube**2)
    for fib in raw["carbon_fibers"]:
        Ef, sig_f, rho_f = float(fib["Ef_GPa"]), float(fib["sigma_ult_MPa"]), float(fib["rho_f_g_cm3"])
        E1_c = Vf * Ef + (1.0 - Vf) * Em
        rho_c = Vf * rho_f + (1.0 - Vf) * rho_m
        sig_c_MPa = E1_c * ((sig_f / 1000.0) / Ef) * 1000.0
        rho_lat = rho_bar_ref * rho_c
        K_g, E100_g = norm_K_opt * rho_bar_ref * E1_c, norm_E100_opt * rho_bar_ref * E1_c
        E_i, K_i = norm_E_iso * rho_bar_ref * E1_c, norm_K_iso * rho_bar_ref * E1_c
        s_K, s_E100, s_Ei = K_g / rho_lat, E100_g / rho_lat, E_i / rho_lat
        r_over_L = math.sqrt(4.0 * (sig_c_MPa / 1000.0) / (math.pi**2 * E1_c))
        rho_crit_solid = (12.0 + 6.0 * math.sqrt(2.0)) * (math.pi * (L_bar**2) * (r_over_L**2)) * L_bar / 8.0
        fibers.append({
            "fiber_id": fib["id"],
            "Ef_GPa": Ef,
            "E1_CFRP_Vf60_GPa": round(E1_c, 4),
            "rho_CFRP_Vf60_g_cm3": round(rho_c, 4),
            "sigma_tensile_CFRP_Vf60_MPa": round(sig_c_MPa, 2),
            "at_5pct_relative_density": {
                "rho_eff_g_cm3": round(rho_lat, 5),
                "K_graded_incentre_GPa": round(K_g, 6),
                "E100_graded_incentre_GPa": round(E100_g, 6),
                "E_isotropic_braced_GPa": round(E_i, 6),
                "K_isotropic_braced_GPa": round(K_i, 6),
                "specific_K_graded_MPa_per_kg_m3": round(s_K, 4),
                "specific_E100_graded_MPa_per_kg_m3": round(s_E100, 4),
                "specific_E_isotropic_braced_MPa_per_kg_m3": round(s_Ei, 4),
                "exceeds_all_metal_HS_specific_E_bounds": bool(s_Ei > max_m_E),
                "exceeds_all_metal_HS_specific_K_bounds": bool(s_K > max_m_K),
            },
            "buckling_vs_tensile_envelope": {
                "solid_rod_critical_r_over_L": round(r_over_L, 6),
                "solid_rod_critical_rho_bar": round(rho_crit_solid, 6),
                "hollow_tube_eta08_buckling_gain": round(tube_gain, 6),
                "hollow_tube_eta08_critical_rho_bar": round(rho_crit_solid / tube_gain, 6),
            },
        })

    return {
        "benchmark_relative_density": rho_bar_ref,
        "reference_metal_HS_upper_bounds": metal_hs,
        "cfrp_truss_benchmarks": fibers,
    }


def shear_within_bound(stiffness: np.ndarray, bound: float = NUMERICAL_BOUND) -> bool:
    return abs(float(stiffness[3, 3])) <= bound


def run_must_fail_controls(num_res: dict[str, Any]) -> list[dict[str, Any]]:
    c_val = 2.0 - math.sqrt(2.0)
    coords, bars = build_fn_bars(3, c_val, 1.0, 1.0, 1.0)
    C_eq, rho_eq = homogenize_periodic_truss(3, coords, bars)
    K_eq, K_max_eq = (C_eq[0, 0] + 2.0 * C_eq[0, 1]) / 3.0, rho_eq / 9.0
    equal_area_misses_bound = bool(abs(K_eq - K_max_eq) > 0.01 * K_max_eq)
    require(equal_area_misses_bound, "equal-area cell met the bulk bound")
    unbraced_has_no_shear = shear_within_bound(C_eq)
    require(unbraced_has_no_shear, "unbraced cell has shear stiffness")
    planted = np.array(C_eq, copy=True)
    planted[3, 3] = 1.0
    planted_shear_rejected = not shear_within_bound(planted)
    require(planted_shear_rejected, "planted shear stiffness was accepted")

    bound_holds = True
    for a_t in (0.2, 1.0, 2.0 * math.sqrt(2.0), 10.0):
        c_t, b_t = build_fn_bars(3, 0.45, 1.0, 1.0, a_t)
        C_t, r_t = homogenize_periodic_truss(3, c_t, b_t)
        within = bool((C_t[0, 0] + 2.0 * C_t[0, 1]) / 3.0 <= r_t / 9.0 + 1e-12)
        bound_holds = bound_holds and within
        require(within, "pin-jointed truss exceeded the bulk bound")

    e_iso = float(num_res["isotropic_braced_point"]["E_iso_over_rhoE1"])
    _, _, E_hs = hashin_shtrikman_upper_bounds(1.0, 0.25, 0.05)
    below_same_material_plate = e_iso < (E_hs / 0.05) - 0.1
    require(below_same_material_plate, "isotropic truss met the same-material plate bound")

    C_cut, _ = homogenize_periodic_truss(3, coords, [b for idx, b in enumerate(bars) if idx != 12])
    cut_breaks_axial = bool(abs(C_cut[0, 0] - C_cut[1, 1]) > 1e-2)
    require(cut_breaks_axial, "cutting a cross bar preserved axial stiffness")

    c_off, b_off = build_fn_bars(3, 0.35, 1.0, 1.0, 1.0)
    C_off, _ = homogenize_periodic_truss(3, c_off, b_off)
    nu_off = float(C_off[0, 1] / (C_off[0, 0] + C_off[0, 1]))
    poisson_moves = abs(nu_off - 0.20) > 0.02
    require(poisson_moves, "equal-area Poisson ratio stayed at the unit-spring value")

    return [
        {"control": "equal_area_saturates_bulk_bound", "rejected": equal_area_misses_bound, "observed_K_over_Kmax": round(float(K_eq / K_max_eq), 8)},
        {"control": "unbraced_incentre_has_shear_rigidity", "rejected": unbraced_has_no_shear},
        {"control": "planted_shear_stiffness_is_accepted", "rejected": planted_shear_rejected},
        {"control": "pin_jointed_truss_exceeds_cauchy_maxwell_bulk_bound", "rejected": bound_holds},
        {"control": "isotropic_truss_exceeds_same_material_HS_plate_bound", "rejected": below_same_material_plate, "truss_E_over_rhoE1": round(e_iso, 8), "hs_plate_E_over_rhoEs_at_5pct": round(E_hs / 0.05, 8)},
        {"control": "cut_cross_bar_preserves_axial_stiffness", "rejected": cut_breaks_axial, "C11_after_cut": round(float(C_cut[0, 0]), 10), "C22_after_cut": round(float(C_cut[1, 1]), 10)},
        {"control": "unit_spring_poisson_holds_for_equal_area_bars_at_general_t", "rejected": poisson_moves, "nu_100_at_t035_equal_area": round(nu_off, 8), "unit_spring_nu_100": 0.2},
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify fibral-rigidity results.")
    parser.add_argument("--write", action="store_true", help="Write results/certificate.json")
    args = parser.parse_args()

    sym = run_symbolic_checks()
    num = run_numerical_truss_checks()
    cfrp = run_cfrp_benchmark_checks(num)
    controls = run_must_fail_controls(num)
    cert = {
        "repository": "Foundation-0/fibral-rigidity",
        "certificate_version": "1.0.0",
        "inputs": {"data/carbon_fibers.json_sha256": hashlib.sha256(json.dumps(json.loads(DATA_PATH.read_text(encoding="utf-8")), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()},
        "symbolic_verification": sym,
        "numerical_homogenization": num,
        "cfrp_material_benchmarks": cfrp,
        "must_fail_controls": controls,
    }
    rendered = json.dumps(sanitize(cert), indent=2, sort_keys=True) + "\n"

    if args.write:
        CERT_PATH.parent.mkdir(parents=True, exist_ok=True)
        CERT_PATH.write_text(rendered, encoding="utf-8")

    if not CERT_PATH.exists() or json.loads(CERT_PATH.read_text(encoding="utf-8")) != cert:
        print("[FAIL] Certificate mismatch or missing.", file=sys.stderr)
        return 1

    print(f"[PASS] Symbolic (n=2..7), Numerical ({num['coordinate_homogenization_cases_checked']} cases, discrepancy within 1e-9), CFRP (5 grades), Controls ({len(controls)}/{len(controls)}): OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
