#!/usr/bin/env python3
"""
Receiving-Side Exact-Q Unilateral Prestress, Octagonal-Pore / Checkerboard Bracing,
and Sherman-Morrison Bar-Cut Localization Checker for `aska-physical/fibral-rigidity`
(`origin/main @ 06520520dff2e956`, `PROOF.md:15-22,73-75`, `verify.py:22-54,367-368`).

Evidence classification (`aska-physical/dock#8`): MODEL / CHECKER (exact pure-Q algebra,
not bench measurement). Zero floats on trust paths. Zero bare `assert`s (`-O` safe).
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import hmac
import json
from pathlib import Path
import sys
from typing import Any

CERT_SCHEMA = "aska-physical.fibral-rigidity.prestress-bracing-audit.v1"
UPSTREAM_REPO = "aska-physical/fibral-rigidity"
UPSTREAM_MAIN_SHA = "06520520dff2e956715f18cfa3ba10e7ee83487b"
HERE = Path(__file__).resolve().parent
DEFAULT_CERT_PATH = HERE / "prestress_localization_certificate.json"

MAX_JSON_DEPTH = 32
MAX_JSON_BYTES = 2_000_000
MAX_INT_DIGITS = 400
MAX_JSON_INT = 10**MAX_INT_DIGITS
MAX_SEQ_LEN = 256
MAX_MATRIX_DIM = 48
_HEX_LOWER = frozenset("0123456789abcdef")
_CERT_KEYS = frozenset({"schema", "upstream_repo", "upstream_main_sha", "payload_sha256", "certificate_sha256", "payload"})


class PrestressAuditError(ValueError):
    """Fail-closed typed error (never stripped under python3 -O)."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


# ---------------------------------------------------------------------------
# Strict Input Hardening & Iterative Zero-Float Canonical JSON
# ---------------------------------------------------------------------------

def require_int(val: Any, label: str = "int", *, lo: int | None = None, hi: int | None = None) -> int:
    if type(val) is not int:
        raise PrestressAuditError("INVALID_INT_TYPE", f"{label} must be strict int")
    if lo is not None and val < lo:
        raise PrestressAuditError("INT_BELOW_MIN", f"{label} below min {lo}")
    if hi is not None and val > hi:
        raise PrestressAuditError("INT_ABOVE_MAX", f"{label} above max {hi}")
    return val


def require_rat(val: Any, label: str = "rational") -> Fraction:
    if type(val) is Fraction:
        return val
    if type(val) is int:
        return Fraction(val, 1)
    raise PrestressAuditError("INVALID_RATIONAL_TYPE", f"{label} must be strict int or Fraction")


def require_pos_rat(val: Any, label: str = "pos_rational") -> Fraction:
    q = require_rat(val, label)
    if q <= 0:
        raise PrestressAuditError("NON_POSITIVE_RATIONAL", f"{label} must be > 0")
    return q


def require_seq(val: Any, label: str = "seq", *, min_len: int = 1, max_len: int = MAX_SEQ_LEN) -> tuple[Any, ...]:
    if type(val) not in (list, tuple):
        raise PrestressAuditError("INVALID_SEQ_TYPE", f"{label} must be strict list or tuple")
    n = len(val)
    if n < min_len:
        raise PrestressAuditError("SEQ_TOO_SHORT", f"{label} length {n} < {min_len}")
    if n > max_len:
        raise PrestressAuditError("SEQ_TOO_LONG", f"{label} length {n} > {max_len}")
    return tuple(val)


def fmt_q(q: Fraction) -> str:
    r = require_rat(q, "q")
    if r.denominator == 1:
        return str(r.numerator)
    return f"{r.numerator}/{r.denominator}"


def validate_json_value(obj: Any, *, max_depth: int = MAX_JSON_DEPTH) -> None:
    """Iterative zero-float, depth-capped, strict-key JSON tree validator (zero recursion)."""
    limit = require_int(max_depth, "max_depth", lo=1, hi=128)
    stack: list[tuple[Any, int]] = [(obj, 0)]
    while stack:
        cur, depth = stack.pop()
        if depth > limit:
            raise PrestressAuditError("MAX_DEPTH_EXCEEDED", f"JSON nesting exceeds {limit}")
        t = type(cur)
        if t is dict:
            for k, v in cur.items():
                if type(k) is not str:
                    raise PrestressAuditError("NON_STR_KEY", "dict keys must be strict str")
                stack.append((v, depth + 1))
        elif t is list:
            for item in cur:
                stack.append((item, depth + 1))
        elif t is int:
            if abs(cur) > MAX_JSON_INT:
                raise PrestressAuditError("INT_TOO_LARGE", "integer exceeds digit cap")
        elif t in (str, bool) or cur is None:
            continue
        elif t is float:
            raise PrestressAuditError("FLOAT_FORBIDDEN", "float values are forbidden")
        else:
            raise PrestressAuditError("UNSUPPORTED_JSON_TYPE", f"unsupported type {t.__name__}")


def _reject_float_token(tok: str) -> Any:
    raise PrestressAuditError("JSON_FLOAT_FORBIDDEN", f"float literal forbidden: {tok!r}")


def _reject_constant_token(tok: str) -> Any:
    raise PrestressAuditError("JSON_CONSTANT_FORBIDDEN", f"non-RFC8259 constant forbidden: {tok!r}")


def _parse_capped_int_token(tok: str) -> int:
    digits = tok[1:] if tok.startswith("-") else tok
    if len(digits) > MAX_INT_DIGITS:
        raise PrestressAuditError("INT_DIGIT_LIMIT", f"integer token exceeds {MAX_INT_DIGITS} digits")
    return int(tok)


def _pairs_no_dup(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise PrestressAuditError("JSON_DUPLICATE_KEY", f"duplicate JSON key: {k!r}")
        out[k] = v
    return out


def parse_strict_json(text: str) -> Any:
    if type(text) is not str:
        raise PrestressAuditError("INVALID_JSON_INPUT", "JSON input must be strict str")
    if len(text.encode("utf-8")) > MAX_JSON_BYTES:
        raise PrestressAuditError("JSON_TOO_LARGE", "JSON input exceeds byte cap")
    if not text.strip():
        raise PrestressAuditError("EMPTY_JSON", "JSON input must be non-empty")
    try:
        parsed = json.loads(
            text,
            parse_int=_parse_capped_int_token,
            parse_float=_reject_float_token,
            parse_constant=_reject_constant_token,
            object_pairs_hook=_pairs_no_dup,
        )
    except PrestressAuditError:
        raise
    except (ValueError, RecursionError) as exc:
        raise PrestressAuditError("JSON_PARSE_ERROR", str(exc)) from exc
    validate_json_value(parsed)
    return parsed


def canonical_json_bytes(obj: Any) -> bytes:
    validate_json_value(obj)
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# Exact-Q Linear Algebra: Multi-RHS Solver, RREF Rank, Bunch-Kaufman Inertia
# ---------------------------------------------------------------------------

def solve_linear_multi_q(matrix: Any, rhs_rows: Any) -> list[list[Fraction]]:
    rows_raw = require_seq(matrix, "matrix", min_len=1, max_len=MAX_MATRIX_DIM)
    rhs_raw = require_seq(rhs_rows, "rhs_rows", min_len=1, max_len=MAX_MATRIX_DIM)
    n = len(rows_raw)
    if len(rhs_raw) != n:
        raise PrestressAuditError("DIM_MISMATCH", "rhs row count must equal matrix dimension")
    first_rhs = require_seq(rhs_raw[0], "rhs_rows[0]", min_len=1, max_len=MAX_MATRIX_DIM)
    m_rhs = len(first_rhs)
    aug: list[list[Fraction]] = []
    for i in range(n):
        row = require_seq(rows_raw[i], f"row[{i}]", min_len=n, max_len=n)
        b_row = require_seq(rhs_raw[i], f"rhs_rows[{i}]", min_len=m_rhs, max_len=m_rhs)
        aug.append(
            [require_rat(x, f"A[{i}][{j}]") for j, x in enumerate(row)]
            + [require_rat(y, f"B[{i}][{k}]") for k, y in enumerate(b_row)]
        )
    tot_cols = n + m_rhs
    for col in range(n):
        piv = -1
        for r in range(col, n):
            if aug[r][col] != 0:
                piv = r
                break
        if piv < 0:
            raise PrestressAuditError("SINGULAR_MATRIX", f"singular matrix at column {col}")
        if piv != col:
            aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        piv_row = [x / pv for x in aug[col]]
        aug[col] = piv_row
        nz = [(j, v) for j, v in enumerate(piv_row) if v != 0]
        for r in range(n):
            if r != col and aug[r][col] != 0:
                factor = aug[r][col]
                row_r = aug[r]
                for j, v in nz:
                    row_r[j] -= factor * v
    return [aug[i][n:tot_cols] for i in range(n)]


def matrix_rank_q(rows: Any, ncols: int) -> tuple[int, list[int]]:
    """Exact Gaussian elimination rank and pivot column indices over Q."""
    nc = require_int(ncols, "ncols", lo=1, hi=MAX_SEQ_LEN)
    rows_raw = require_seq(rows, "rows", min_len=1, max_len=MAX_SEQ_LEN)
    mat: list[list[Fraction]] = []
    for i, raw_r in enumerate(rows_raw):
        rseq = require_seq(raw_r, f"row[{i}]", min_len=nc, max_len=nc)
        mat.append([require_rat(x, f"M[{i}][{j}]") for j, x in enumerate(rseq)])
    piv_cols: list[int] = []
    r = 0
    nrows = len(mat)
    for c in range(nc):
        piv = -1
        for i in range(r, nrows):
            if mat[i][c] != 0:
                piv = i
                break
        if piv < 0:
            continue
        if piv != r:
            mat[r], mat[piv] = mat[piv], mat[r]
        pv = mat[r][c]
        nz = [(j, mat[r][j] / pv) for j in range(c, nc) if mat[r][j] != 0]
        for j, v in nz:
            mat[r][j] = v
        for i in range(r + 1, nrows):
            if mat[i][c] != 0:
                f = mat[i][c]
                row_i = mat[i]
                for j, v in nz:
                    row_i[j] -= f * v
        piv_cols.append(c)
        r += 1
        if r == nrows:
            break
    return r, piv_cols


def sylvester_inertia_q(matrix: Any) -> tuple[int, int, int]:
    """Exact symmetric Bunch-Kaufman 1x1/2x2 congruence Sylvester inertia (n_pos, n_neg, n_zero)."""
    rows_raw = require_seq(matrix, "matrix", min_len=1, max_len=MAX_MATRIX_DIM)
    n = len(rows_raw)
    A: list[list[Fraction]] = []
    for i in range(n):
        rseq = require_seq(rows_raw[i], f"row[{i}]", min_len=n, max_len=n)
        A.append([require_rat(x, f"S[{i}][{j}]") for j, x in enumerate(rseq)])
    for i in range(n):
        for j in range(i + 1, n):
            if A[i][j] != A[j][i]:
                raise PrestressAuditError("NON_SYMMETRIC_MATRIX", f"asymmetry at ({i},{j})")

    n_pos = 0
    n_neg = 0
    n_zero = 0
    k = 0
    while k < n:
        diag_piv = -1
        for i in range(k, n):
            if A[i][i] != 0:
                diag_piv = i
                break
        if diag_piv >= 0:
            if diag_piv != k:
                A[k], A[diag_piv] = A[diag_piv], A[k]
                for row in A:
                    row[k], row[diag_piv] = row[diag_piv], row[k]
            d = A[k][k]
            if d > 0:
                n_pos += 1
            else:
                n_neg += 1
            for i in range(k + 1, n):
                f = A[i][k] / d
                if f != 0:
                    for j in range(k + 1, n):
                        A[i][j] -= f * A[k][j]
            k += 1
            continue

        off_i = -1
        off_j = -1
        for i in range(k, n):
            for j in range(i + 1, n):
                if A[i][j] != 0:
                    off_i = i
                    off_j = j
                    break
            if off_i >= 0:
                break
        if off_i < 0:
            n_zero += n - k
            break

        if off_i != k:
            A[k], A[off_i] = A[off_i], A[k]
            for row in A:
                row[k], row[off_i] = row[off_i], row[k]
        if off_j != k + 1:
            A[k + 1], A[off_j] = A[off_j], A[k + 1]
            for row in A:
                row[k + 1], row[off_j] = row[off_j], row[k + 1]

        b = A[k][k + 1]
        n_pos += 1
        n_neg += 1
        for i in range(k + 2, n):
            c0 = A[i][k + 1] / b
            c1 = A[i][k] / b
            for j in range(k + 2, n):
                A[i][j] -= c0 * A[k][j] + c1 * A[k + 1][j]
        k += 2

    return (n_pos, n_neg, n_zero)


# ---------------------------------------------------------------------------
# Part A: Unilateral Tension-Only Prestress Certificate (PROOF.md:75)
# ---------------------------------------------------------------------------

def verify_anchored_prestress_q(omega_tie: Any, omega_inner: Any, omega_anchor: Any) -> dict[str, Any]:
    """
    Exact-Q Connelly-Whiteley unilateral prestress certifier on an anchored 6-spoke
    integer-coordinate cable unit (13 nodes: 1 hub + 6 inner ring + 6 boundary anchors; 18 cables).
    Verifies equilibrium residuals and Sylvester inertia of Omega_int and Omega_full.
    """
    w_t = require_rat(omega_tie, "omega_tie")
    w_i = require_rat(omega_inner, "omega_inner")
    w_a = require_rat(omega_anchor, "omega_anchor")
    ring: tuple[tuple[Fraction, Fraction], ...] = (
        (Fraction(2), Fraction(0)),
        (Fraction(1), Fraction(2)),
        (Fraction(-1), Fraction(2)),
        (Fraction(-2), Fraction(0)),
        (Fraction(-1), Fraction(-2)),
        (Fraction(1), Fraction(-2)),
    )
    coords: list[tuple[Fraction, Fraction]] = [(Fraction(0), Fraction(0))]
    coords.extend(ring)
    coords.extend([(2 * x, 2 * y) for x, y in ring])

    edges_w: list[tuple[int, int, Fraction]] = []
    for k in range(6):
        edges_w.append((0, 1 + k, w_i))
        edges_w.append((1 + k, 1 + (k + 1) % 6, w_t))
        edges_w.append((1 + k, 7 + k, w_a))

    eq_ok = True
    for v in range(7):
        fx = Fraction(0)
        fy = Fraction(0)
        px, py = coords[v]
        for a, b, w in edges_w:
            if a == v:
                qx, qy = coords[b]
                fx += w * (px - qx)
                fy += w * (py - qy)
            elif b == v:
                qx, qy = coords[a]
                fx += w * (px - qx)
                fy += w * (py - qy)
        if (fx, fy) != (0, 0):
            eq_ok = False

    n_full = 13
    omega_full = [[Fraction(0)] * n_full for _ in range(n_full)]
    for a, b, w in edges_w:
        omega_full[a][a] += w
        omega_full[b][b] += w
        omega_full[a][b] -= w
        omega_full[b][a] -= w
    omega_int = [row[:7] for row in omega_full[:7]]
    in_int = sylvester_inertia_q(omega_int)
    in_full = sylvester_inertia_q(omega_full)
    strictly_tension = w_t > 0 and w_i > 0 and w_a > 0
    return {
        "strictly_tension_only": strictly_tension,
        "interior_equilibrium_zero_residual": eq_ok,
        "omega_int_inertia": list(in_int),
        "omega_full_inertia": list(in_full),
        "omega_int_strictly_pd": in_int == (7, 0, 0),
        "omega_full_psd": in_full[1] == 0,
        "receiving_prestress_gate_pass": all((strictly_tension, eq_ok, in_int == (7, 0, 0), in_full[1] == 0)),
    }


# ---------------------------------------------------------------------------
# Part B: Exact-Q Periodic F_3 Octagonal-Pore Bracing (`verify.py:22-54,367-368`)
#         & Euplectella Checkerboard Cylindrical Shell Maxwell-Calladine
# ---------------------------------------------------------------------------

def homogenize_f3_exact_q(
    t_param: Any,
    k_cross: Any,
    k_tie: Any,
    tie_mode: str,
    *,
    drop_bar_index: int = -1,
) -> dict[str, Any]:
    """
    Exact-Q periodic Schur-complement homogenization of the REAL `fibral-rigidity` F_3 cell
    (`verify.py:22-54`) on Q = [-1, 1]^3 with spring constants (k_intra=1, k_cross, k_tie)
    and exact rational coordinates at any rational t in (0, 1).
    """
    t = require_pos_rat(t_param, "t_param")
    if t >= 1:
        raise PrestressAuditError("INVALID_T_PARAM", "t_param must lie strictly in (0, 1)")
    kc = require_pos_rat(k_cross, "k_cross")
    kt = require_pos_rat(k_tie, "k_tie")
    if tie_mode not in ("none", "one_per_plane", "half_2_opp", "full_4"):
        raise PrestressAuditError("UNKNOWN_TIE_MODE", f"unknown tie_mode: {tie_mode!r}")

    coords: list[list[Fraction]] = []
    for i in range(3):
        p_neg = [Fraction(0)] * 3
        p_neg[i] = -t
        p_pos = [Fraction(0)] * 3
        p_pos[i] = t
        coords.extend([p_neg, p_pos])

    bars: list[tuple[int, int, list[Fraction], Fraction]] = []
    for i in range(3):
        for j in range(i + 1, 3):
            for si in (0, 1):
                for sj in (0, 1):
                    bars.append((2 * i + si, 2 * j + sj, [Fraction(0)] * 3, Fraction(1)))
    for i in range(3):
        sh = [Fraction(0)] * 3
        sh[i] = Fraction(2)
        bars.append((2 * i + 1, 2 * i + 0, sh, kc))
    for i in range(3):
        for j in range(i + 1, 3):
            s_neg = [Fraction(0)] * 3
            s_neg[i] = Fraction(-2)
            s_neg[j] = Fraction(2)
            s_pos = [Fraction(0)] * 3
            s_pos[i] = Fraction(2)
            s_pos[j] = Fraction(2)
            plane_ties = (
                (2 * i + 0, 2 * j + 0, s_neg, kt),
                (2 * j + 1, 2 * i + 1, s_neg, kt),
                (2 * i + 1, 2 * j + 0, s_pos, kt),
                (2 * j + 1, 2 * i + 0, s_pos, kt),
            )
            if tie_mode == "full_4":
                bars.extend(plane_ties)
            elif tie_mode == "half_2_opp":
                bars.extend((plane_ties[0], plane_ties[2]))
            elif tie_mode == "one_per_plane":
                bars.append(plane_ties[0])

    drop_idx = require_int(drop_bar_index, "drop_bar_index", lo=-1, hi=len(bars) - 1)
    if drop_idx >= 0:
        bars = [b for idx, b in enumerate(bars) if idx != drop_idx]

    K_uu = [[Fraction(0)] * 18 for _ in range(18)]
    K_ue = [[Fraction(0)] * 6 for _ in range(18)]
    K_ee = [[Fraction(0)] * 6 for _ in range(6)]
    compat_rows: list[list[Fraction]] = []
    for u, v, sh, kb in bars:
        vec = [coords[v][d] + sh[d] - coords[u][d] for d in range(3)]
        L2 = sum((x * x for x in vec), Fraction(0))
        wb = kb / L2
        bu = [Fraction(0)] * 18
        for d in range(3):
            bu[3 * v + d] += vec[d]
            bu[3 * u + d] -= vec[d]
        be = [vec[0] ** 2, vec[1] ** 2, vec[2] ** 2, vec[0] * vec[1], vec[0] * vec[2], vec[1] * vec[2]]
        compat_rows.append(bu + be)
        nz_u = [(a, val) for a, val in enumerate(bu) if val != 0]
        for a, va in nz_u:
            for b, vb in nz_u:
                K_uu[a][b] += wb * va * vb
            for e in range(6):
                if be[e] != 0:
                    K_ue[a][e] += wb * va * be[e]
        for e1 in range(6):
            for e2 in range(6):
                K_ee[e1][e2] += wb * be[e1] * be[e2]

    rk_tot, _ = matrix_rank_q(compat_rows, 24)
    rk_u, pivs = matrix_rank_q(K_uu, 18)
    K_red = [[K_uu[i][j] for j in pivs] for i in pivs]
    B_red = [[-K_ue[i][e] for e in range(6)] for i in pivs]
    U_red = solve_linear_multi_q(K_red, B_red)
    U_full = [[Fraction(0)] * 6 for _ in range(18)]
    for idx, pcol in enumerate(pivs):
        U_full[pcol] = U_red[idx]

    eq_zero = all(
        sum((K_uu[i][j] * U_full[j][e] for j in range(18)), Fraction(0)) + K_ue[i][e] == 0
        for i in range(18)
        for e in range(6)
    )
    C = [
        [(K_ee[a][b] + sum((K_ue[i][a] * U_full[i][b] for i in range(18)), Fraction(0))) / Fraction(8) for b in range(6)]
        for a in range(6)
    ]
    c_inertia = sylvester_inertia_q(C)
    n_ties = len(bars) - 15 + (1 if drop_idx >= 0 else 0)
    return {
        "tie_mode": tie_mode,
        "drop_bar_index": drop_idx,
        "n_bars": len(bars),
        "n_ties": n_ties,
        "total_periodic_rank": rk_tot,
        "internal_rank": rk_u,
        "macro_elastic_rank": rk_tot - rk_u,
        "relaxation_equilibrium_zero_residual": eq_zero,
        "C11": C[0][0],
        "C22": C[1][1],
        "C12": C[0][1],
        "C44": C[3][3],
        "C55": C[4][4],
        "C66": C[5][5],
        "C14_shear_coupling": C[0][3],
        "C_cell_inertia": list(c_inertia),
    }


def analyze_checkerboard_shell_q(n_rows: int, n_cols: int, pattern: str) -> dict[str, Any]:
    """Exact-Q Maxwell-Calladine rank-nullity on a cylindrical square shell."""
    nr = require_int(n_rows, "n_rows", lo=2, hi=6)
    nc = require_int(n_cols, "n_cols", lo=3, hi=6)
    if pattern not in ("none", "checker_1", "checker_1_bridge", "checker_x", "bolker_crapo_tree", "full_1"):
        raise PrestressAuditError("UNKNOWN_BRACING_PATTERN", f"unknown pattern: {pattern!r}")
    ny = nr + 1
    n_verts = ny * nc
    ncols = 2 * n_verts

    def vid(r: int, c: int) -> int:
        return r * nc + (c % nc)

    rows: list[list[Fraction]] = []
    for r in range(ny):
        for c in range(nc):
            u, v = vid(r, c), vid(r, c + 1)
            row = [Fraction(0)] * ncols
            row[2 * u] = Fraction(-1)
            row[2 * v] = Fraction(1)
            rows.append(row)
    for r in range(nr):
        for c in range(nc):
            u, v = vid(r, c), vid(r + 1, c)
            row = [Fraction(0)] * ncols
            row[2 * u + 1] = Fraction(-1)
            row[2 * v + 1] = Fraction(1)
            rows.append(row)
    n_diags = 0
    for r in range(nr):
        for c in range(nc):
            u00, u10, u01, u11 = vid(r, c), vid(r, c + 1), vid(r + 1, c), vid(r + 1, c + 1)
            add_slash = False
            add_back = False
            if pattern == "checker_1" and (r + c) % 2 == 0:
                add_slash = True
            elif pattern == "checker_1_bridge" and ((r + c) % 2 == 0 or (r == 0 and c == 1)):
                add_slash = True
            elif pattern == "checker_x" and (r + c) % 2 == 0:
                add_slash = True
                add_back = True
            elif pattern == "bolker_crapo_tree" and (r == 0 or c == 0):
                add_slash = True
            elif pattern == "full_1":
                add_slash = True
            if add_slash:
                row = [Fraction(0)] * ncols
                row[2 * u00] = Fraction(-1)
                row[2 * u00 + 1] = Fraction(-1)
                row[2 * u11] = Fraction(1)
                row[2 * u11 + 1] = Fraction(1)
                rows.append(row)
                n_diags += 1
            if add_back:
                row = [Fraction(0)] * ncols
                row[2 * u10] = Fraction(1)
                row[2 * u10 + 1] = Fraction(-1)
                row[2 * u01] = Fraction(-1)
                row[2 * u01 + 1] = Fraction(1)
                rows.append(row)
                n_diags += 1

    rk, _ = matrix_rank_q(rows, ncols)
    internal_dofs = ncols - 2
    torsion_vec: list[Fraction] = []
    for r in range(ny):
        for _ in range(nc):
            torsion_vec.extend([Fraction(r), Fraction(0)])
    torsion_energy = sum(
        (sum((v * torsion_vec[j] for j, v in enumerate(row) if v != 0), Fraction(0)) ** 2 for row in rows),
        Fraction(0),
    )
    return {
        "pattern": pattern,
        "n_diagonals": n_diags,
        "rank": rk,
        "floppy_modes": internal_dofs - rk,
        "self_stresses": len(rows) - rk,
        "global_torsion_energy": fmt_q(torsion_energy),
    }


# ---------------------------------------------------------------------------
# Part C: Exact-Q Sherman-Morrison Rank-1 Bar-Cut Localization Across lambda
# ---------------------------------------------------------------------------

def _is_connected(n: int, edges: tuple[tuple[int, int], ...]) -> bool:
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for w in adj[u]:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return len(seen) == n


def sherman_morrison_cut_q(
    n_nodes: int,
    edges: Any,
    stiffnesses: Any,
    cut_index: int,
) -> dict[str, Any]:
    """Exact-Q Sherman-Morrison rank-1 edge-cut LODF cross-checked by direct post-cut re-solve."""
    n = require_int(n_nodes, "n_nodes", lo=2, hi=MAX_MATRIX_DIM - 1)
    eseq = require_seq(edges, "edges", min_len=1, max_len=MAX_SEQ_LEN)
    kseq = require_seq(stiffnesses, "stiffnesses", min_len=len(eseq), max_len=len(eseq))
    es: list[tuple[int, int]] = []
    ks: list[Fraction] = []
    for idx, (e, k_val) in enumerate(zip(eseq, kseq)):
        pair = require_seq(e, f"edge[{idx}]", min_len=2, max_len=2)
        u = require_int(pair[0], f"edge[{idx}].u", lo=0, hi=n - 1)
        v = require_int(pair[1], f"edge[{idx}].v", lo=0, hi=n - 1)
        if u == v:
            raise PrestressAuditError("SELF_LOOP_FORBIDDEN", f"self-loop at vertex {u}")
        es.append((u, v))
        ks.append(require_pos_rat(k_val, f"stiffnesses[{idx}]"))
    if not _is_connected(n, tuple(es)):
        raise PrestressAuditError("DISCONNECTED_GRAPH", "graph must be connected")
    cidx = require_int(cut_index, "cut_index", lo=0, hi=len(es) - 1)

    L = [[Fraction(0)] * n for _ in range(n)]
    for (u, v), k in zip(es, ks):
        L[u][u] += k
        L[v][v] += k
        L[u][v] -= k
        L[v][u] -= k
    bordered = [row + [Fraction(1)] for row in L] + [[Fraction(1)] * n + [Fraction(0)]]
    inv_n = Fraction(1, n)
    rhs_rows = [[(Fraction(1) if i == j else Fraction(0)) - inv_n for j in range(n)] for i in range(n)] + [[Fraction(0)] * n]
    Lp = solve_linear_multi_q(bordered, rhs_rows)[:n]

    def quad(u: int, v: int, x: int, y: int) -> Fraction:
        return Lp[u][x] - Lp[u][y] - Lp[v][x] + Lp[v][y]

    eu, ev = es[cidx]
    k_cut = ks[cidx]
    h_cut = k_cut * quad(eu, ev, eu, ev)
    if h_cut == 1:
        raise PrestressAuditError("BRIDGE_EDGE_CUT", f"edge {cidx} is a bridge (h_e = 1)")

    denom = Fraction(1) - h_cut
    lodf = [Fraction(0) if idx == cidx else (k * quad(u, v, eu, ev) / denom) for idx, ((u, v), k) in enumerate(zip(es, ks))]

    # Independent direct post-cut re-solve under unit dipole p = b_e
    L_rem = [row[:] for row in L]
    L_rem[eu][eu] -= k_cut
    L_rem[ev][ev] -= k_cut
    L_rem[eu][ev] += k_cut
    L_rem[ev][eu] += k_cut
    bordered_rem = [row + [Fraction(1)] for row in L_rem] + [[Fraction(1)] * n + [Fraction(0)]]
    dipole_rhs = [[Fraction(1 if i == eu else (-1 if i == ev else 0))] for i in range(n + 1)]
    u_after = [row[0] for row in solve_linear_multi_q(bordered_rem, dipole_rhs)[:n]]
    u_before = [Lp[i][eu] - Lp[i][ev] for i in range(n)]
    f_before = k_cut * (u_before[eu] - u_before[ev])
    direct_match = all(
        (k * ((u_after[u] - u_after[v]) - (u_before[u] - u_before[v])) / f_before) == lodf[idx]
        for idx, ((u, v), k) in enumerate(zip(es, ks))
        if idx != cidx
    )
    foster_sum = sum((k * quad(u, v, u, v) for (u, v), k in zip(es, ks)), Fraction(0))
    return {
        "foster_sum": foster_sum,
        "h_cut": h_cut,
        "lodf": lodf,
        "direct_resolve_exact_match": direct_match,
    }


def _build_graded_web(lam: int) -> tuple[int, list[tuple[int, int]], list[Fraction], list[tuple[str, int, int]]]:
    n_spokes, n_rings = 6, 2
    n = 1 + n_spokes * n_rings
    edges: list[tuple[int, int]] = []
    ks: list[Fraction] = []
    kinds: list[tuple[str, int, int]] = []
    for j in range(n_spokes):
        edges.append((0, 1 + j))
        ks.append(Fraction(lam, 1))
        kinds.append(("primary", 0, j))
    for j in range(n_spokes):
        edges.append((1 + j, 1 + n_spokes + j))
        ks.append(Fraction(lam, 1))
        kinds.append(("primary", 1, j))
    for r in range(n_rings):
        for j in range(n_spokes):
            edges.append((1 + r * n_spokes + j, 1 + r * n_spokes + (j + 1) % n_spokes))
            ks.append(Fraction(1, 1))
            kinds.append(("tie", r, j))
    return n, edges, ks, kinds


def build_study_payload() -> dict[str, Any]:
    """Build deterministic exact-Q Rank #2 receiving-side payload for aska-physical/fibral-rigidity."""
    # Part A: Unilateral tension-only prestress PASS vs compression FAIL (PROOF.md:75)
    prestress_pass = verify_anchored_prestress_q(Fraction(1), Fraction(10), Fraction(11))
    prestress_comp_fail = verify_anchored_prestress_q(Fraction(1), Fraction(-10), Fraction(-9))

    # Part B1: Real F_3(t = 3/5) periodic homogenization (`verify.py:22-54,367-368`)
    t_rat = Fraction(3, 5)
    f3_none = homogenize_f3_exact_q(t_rat, Fraction(1), Fraction(1), "none")
    f3_one = homogenize_f3_exact_q(t_rat, Fraction(1), Fraction(1), "one_per_plane")
    f3_half = homogenize_f3_exact_q(t_rat, Fraction(1), Fraction(1), "half_2_opp")
    f3_full = homogenize_f3_exact_q(t_rat, Fraction(1), Fraction(1), "full_4")
    f3_cut12 = homogenize_f3_exact_q(t_rat, Fraction(1), Fraction(1), "none", drop_bar_index=12)

    def _ser_f3(d: dict[str, Any]) -> dict[str, Any]:
        return {
            "tie_mode": d["tie_mode"],
            "drop_bar_index": d["drop_bar_index"],
            "n_bars": d["n_bars"],
            "n_ties": d["n_ties"],
            "total_periodic_rank": d["total_periodic_rank"],
            "internal_rank": d["internal_rank"],
            "macro_elastic_rank": d["macro_elastic_rank"],
            "relaxation_equilibrium_zero_residual": d["relaxation_equilibrium_zero_residual"],
            "C11": fmt_q(d["C11"]),
            "C22": fmt_q(d["C22"]),
            "C12": fmt_q(d["C12"]),
            "C44": fmt_q(d["C44"]),
            "C14_shear_coupling": fmt_q(d["C14_shear_coupling"]),
            "C_cell_inertia": d["C_cell_inertia"],
        }

    # Part B2: 4x4 cylindrical checkerboard Maxwell-Calladine comparison
    cyl_rows = [
        analyze_checkerboard_shell_q(4, 4, pat)
        for pat in ("none", "checker_1", "checker_1_bridge", "checker_x", "bolker_crapo_tree", "full_1")
    ]
    cyl_map = {r["pattern"]: r for r in cyl_rows}

    # Part C: Sherman-Morrison tie-cut localization across stiffness grading lambda in {1, 10, 100}
    lodf_rows: list[dict[str, Any]] = []
    remote_maxes: list[Fraction] = []
    direct_checks: list[bool] = []
    for lam in (1, 10, 100):
        n, edges, ks, kinds = _build_graded_web(lam)
        cut_idx = kinds.index(("tie", 1, 0))
        res = sherman_morrison_cut_q(n, edges, ks, cut_idx)
        direct_checks.append(all((res["direct_resolve_exact_match"], res["foster_sum"] == Fraction(n - 1))))
        rem_ties = [abs(res["lodf"][i]) for i, kd in enumerate(kinds) if kd[0] == "tie" and kd[2] != 0]
        rmax = max(rem_ties)
        remote_maxes.append(rmax)
        lodf_rows.append({
            "lambda_k2_over_k1": lam,
            "foster_sum": fmt_q(res["foster_sum"]),
            "tie_cut_leverage_h": fmt_q(res["h_cut"]),
            "remote_tie_max_lodf": fmt_q(rmax),
            "direct_resolve_exact_match": res["direct_resolve_exact_match"],
        })

    all_passed = all((
        prestress_pass["receiving_prestress_gate_pass"],
        not prestress_comp_fail["receiving_prestress_gate_pass"],
        prestress_comp_fail["omega_int_inertia"][1] >= 1,
        f3_none["relaxation_equilibrium_zero_residual"],
        f3_none["C11"] == Fraction(2, 9),
        f3_none["C12"] == Fraction(1, 18),
        f3_none["C44"] == 0,
        f3_none["macro_elastic_rank"] == 3,
        f3_one["C44"] > 0,
        f3_one["C14_shear_coupling"] != 0,
        f3_half["relaxation_equilibrium_zero_residual"],
        f3_half["macro_elastic_rank"] == 6,
        f3_half["C44"] == Fraction(1, 4),
        f3_half["C14_shear_coupling"] == 0,
        f3_half["C_cell_inertia"] == [6, 0, 0],
        f3_full["C44"] == Fraction(1, 2),
        f3_full["C14_shear_coupling"] == 0,
        f3_half["C44"] / Fraction(f3_half["n_ties"]) == f3_full["C44"] / Fraction(f3_full["n_ties"]),
        f3_cut12["C11"] != f3_cut12["C22"],
        cyl_map["none"]["floppy_modes"] == 7,
        cyl_map["checker_1"]["floppy_modes"] == 1,
        cyl_map["checker_1_bridge"]["floppy_modes"] == 0,
        cyl_map["bolker_crapo_tree"]["floppy_modes"] == 0,
        all(direct_checks),
        remote_maxes[0] > remote_maxes[1] > remote_maxes[2],
        remote_maxes[0] / remote_maxes[2] > Fraction(20),
    ))

    return {
        "evidence_label": "model_checker_exact_Q_not_bench_measurement",
        "upstream_anchors": {
            "repo": UPSTREAM_REPO,
            "main_sha": UPSTREAM_MAIN_SHA,
            "target_lines": "PROOF.md:15-22,73-75; verify.py:22-54,367-368",
            "rational_t_param": fmt_q(t_rat),
        },
        "part_a_unilateral_prestress": {
            "tension_only_pass_control": prestress_pass,
            "compression_must_fail_control": prestress_comp_fail,
        },
        "part_b1_f3_periodic_octagonal_pore_bracing": {
            "unbraced_none_fails_shear": _ser_f3(f3_none),
            "one_per_plane_3ties_breaks_cubic_coupling": _ser_f3(f3_one),
            "half_density_6ties_passes_cubic_shear": _ser_f3(f3_half),
            "full_density_12ties_reference": _ser_f3(f3_full),
            "cut_cross_bar_idx12_breaks_axial_symmetry": _ser_f3(f3_cut12),
            "shear_per_tie_half_equals_full": fmt_q(f3_half["C44"] / Fraction(f3_half["n_ties"])),
        },
        "part_b2_cylindrical_checkerboard_maxwell_calladine": cyl_rows,
        "part_c_sherman_morrison_cut_localization": {
            "grading_sweep": lodf_rows,
            "remote_tie_max_reduction_lam1_to_lam100": fmt_q(remote_maxes[0] / remote_maxes[2]),
        },
        "all_passed": all_passed,
    }


# ---------------------------------------------------------------------------
# Fail-Closed Certificate Builder, Checker & CLI
# ---------------------------------------------------------------------------

def _require_sha256_hex(val: Any, label: str) -> str:
    if type(val) is not str:
        raise PrestressAuditError("INVALID_DIGEST_TYPE", f"{label} must be strict str")
    if len(val) != 64:
        raise PrestressAuditError("INVALID_DIGEST_LEN", f"{label} must have length 64")
    if any(ch not in _HEX_LOWER for ch in val):
        raise PrestressAuditError("INVALID_DIGEST_HEX", f"{label} must be lowercase hex")
    return val


def _require_no_symlink_path(path: Path) -> None:
    if path.is_symlink():
        raise PrestressAuditError("SYMLINK_FORBIDDEN", f"symlink path forbidden: {path}")
    for parent in path.parents:
        if parent.is_symlink():
            raise PrestressAuditError("SYMLINK_ANCESTOR_FORBIDDEN", f"symlink ancestor forbidden: {parent}")


def build_certificate() -> dict[str, Any]:
    payload = build_study_payload()
    payload_sha256 = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    envelope = {
        "schema": CERT_SCHEMA,
        "upstream_repo": UPSTREAM_REPO,
        "upstream_main_sha": UPSTREAM_MAIN_SHA,
        "payload_sha256": payload_sha256,
        "payload": payload,
    }
    certificate_sha256 = hashlib.sha256(canonical_json_bytes(envelope)).hexdigest()
    return {
        "schema": CERT_SCHEMA,
        "upstream_repo": UPSTREAM_REPO,
        "upstream_main_sha": UPSTREAM_MAIN_SHA,
        "payload_sha256": payload_sha256,
        "certificate_sha256": certificate_sha256,
        "payload": payload,
    }


def verify_certificate_dict(cert: Any) -> dict[str, Any]:
    if type(cert) is not dict:
        raise PrestressAuditError("CERT_NOT_DICT", "certificate root must be a JSON object")
    if set(cert.keys()) != _CERT_KEYS:
        raise PrestressAuditError("CERT_SCHEMA_KEYS", f"unexpected keys: {sorted(cert.keys())}")
    if cert["schema"] != CERT_SCHEMA:
        raise PrestressAuditError("CERT_SCHEMA_MISMATCH", "schema mismatch")
    if cert["upstream_repo"] != UPSTREAM_REPO:
        raise PrestressAuditError("CERT_REPO_MISMATCH", "upstream_repo mismatch")
    if cert["upstream_main_sha"] != UPSTREAM_MAIN_SHA:
        raise PrestressAuditError("CERT_SHA_MISMATCH", "upstream_main_sha mismatch")

    stored_payload_sha = _require_sha256_hex(cert["payload_sha256"], "payload_sha256")
    stored_cert_sha = _require_sha256_hex(cert["certificate_sha256"], "certificate_sha256")
    computed_payload_sha = hashlib.sha256(canonical_json_bytes(cert["payload"])).hexdigest()
    if not hmac.compare_digest(stored_payload_sha, computed_payload_sha):
        raise PrestressAuditError("PAYLOAD_SHA_MISMATCH", "payload_sha256 mismatch")

    envelope = {
        "schema": cert["schema"],
        "upstream_repo": cert["upstream_repo"],
        "upstream_main_sha": cert["upstream_main_sha"],
        "payload_sha256": stored_payload_sha,
        "payload": cert["payload"],
    }
    computed_cert_sha = hashlib.sha256(canonical_json_bytes(envelope)).hexdigest()
    if not hmac.compare_digest(stored_cert_sha, computed_cert_sha):
        raise PrestressAuditError("CERTIFICATE_SHA_MISMATCH", "certificate_sha256 mismatch")

    expected = build_certificate()
    if not hmac.compare_digest(computed_payload_sha, expected["payload_sha256"]):
        raise PrestressAuditError("RECOMPUTED_PAYLOAD_DRIFT", "payload digest differs from exact recomputation")

    return {
        "valid": True,
        "schema": CERT_SCHEMA,
        "payload_sha256": computed_payload_sha,
        "certificate_sha256": computed_cert_sha,
    }


def check_certificate_file(path: Path) -> dict[str, Any]:
    if not isinstance(path, Path):
        raise PrestressAuditError("INVALID_PATH_TYPE", "path must be a pathlib.Path")
    _require_no_symlink_path(path)
    if not path.is_file():
        raise PrestressAuditError("CERT_FILE_NOT_FOUND", f"certificate file not found: {path}")
    if path.stat().st_size > MAX_JSON_BYTES:
        raise PrestressAuditError("CERT_FILE_TOO_LARGE", "certificate file exceeds byte cap")
    raw = path.read_bytes()
    if b"\r\n" in raw:
        raise PrestressAuditError("CRLF_FORBIDDEN", "CRLF line endings forbidden")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PrestressAuditError("INVALID_UTF8", f"invalid UTF-8: {exc}") from exc
    parsed = parse_strict_json(text)
    canonical_raw = canonical_json_bytes(parsed)
    if raw not in (canonical_raw, canonical_raw + b"\n"):
        raise PrestressAuditError("NON_CANONICAL_CERT_BYTES", "certificate file is not canonical JSON")
    return verify_certificate_dict(parsed)


def write_certificate_file(path: Path) -> dict[str, Any]:
    _require_no_symlink_path(path)
    cert = build_certificate()
    raw = canonical_json_bytes(cert) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return check_certificate_file(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exact-Q prestress, bracing & bar-cut checker for fibral-rigidity")
    parser.add_argument("--standalone", action="store_true", help="Run full standalone exact verification")
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON certificate to stdout")
    parser.add_argument("--write", nargs="?", const=str(DEFAULT_CERT_PATH), help="Write canonical certificate")
    parser.add_argument("--check", type=str, help="Fail-closed verification of a certificate JSON file")
    args = parser.parse_args(argv)

    try:
        if args.check is not None:
            res = check_certificate_file(Path(args.check))
            print(
                f"OK {CERT_SCHEMA} payload_sha256={res['payload_sha256']} "
                f"certificate_sha256={res['certificate_sha256']}"
            )
            return 0
        if args.write is not None:
            res = write_certificate_file(Path(args.write))
            print(
                f"WROTE {CERT_SCHEMA} payload_sha256={res['payload_sha256']} "
                f"certificate_sha256={res['certificate_sha256']}"
            )
            return 0
        cert = build_certificate()
        if args.json:
            sys.stdout.buffer.write(canonical_json_bytes(cert) + b"\n")
            return 0
        print(
            f"OK {CERT_SCHEMA} payload_sha256={cert['payload_sha256']} "
            f"certificate_sha256={cert['certificate_sha256']} "
            f"all_passed={cert['payload']['all_passed']}"
        )
        return 0
    except (PrestressAuditError, ValueError, RecursionError, OSError) as exc:
        print(f"REJECT {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
