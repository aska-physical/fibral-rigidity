"""R5 regression suite for aska-physical/fibral-rigidity PR #3 companion hardening.

Covers:
  - Zero bare asserts and zero unguarded max/min/cert-dict/json exposures `(0, 0, 0, 0, 0)` in verify.py
  - Persistent -O-safe planted shear stiffness (C_eq[3, 3] = 1.0) rejection under -I -O -B
  - NaN-swallowing closure in strict_finite_max, homogenization, HS bounds, CFRP benchmarks, and sanitize()
  - Full Zone-B certificate & dataset hardening (deep symlink ancestors,
    symlinked PWD, FIFO non-blocking rejection, lone UTF-16 surrogates, depth cap, size cap, concrete Path)
  - Symmetric sanitize() write/verify canonical byte match against committed results/certificate.json
  - Uses real sympy whenever installed; on zero-sympy hosts, injects a test-only stub strictly inside
    the test harness while running all NumPy homogenization, CFRP, controls, and Zone-B checks live.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock
from contextlib import redirect_stderr, redirect_stdout

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERIFY_PATH = ROOT / "verify.py"
CERT_PATH = ROOT / "results" / "certificate.json"
DATA_PATH = ROOT / "data" / "carbon_fibers.json"
HAS_SYMPY = importlib.util.find_spec("sympy") is not None


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(f"FIBRAL_R5_FAILED: {msg}")


def _load_verify_module():
    if not HAS_SYMPY and "sympy" not in sys.modules:
        sys.modules["sympy"] = types.ModuleType("sympy")
    root_s = str(ROOT)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    import verify
    return verify


class _EvilPath(type(Path())):
    pass


class _StrSub(str):
    pass


class _SubList(list):
    pass


class _SubTuple(tuple):
    pass


class TestFibralRigidityNanZoneB(unittest.TestCase):
    def test_01_ast_zero_s21c_exposure_and_zero_bare_asserts(self) -> None:
        for path in (VERIFY_PATH, Path(__file__).resolve()):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            asserts = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assert)]
            _check(len(asserts) == 0, f"bare asserts in {path.name}: {asserts}")

        src = VERIFY_PATH.read_text(encoding="utf-8")
        tree = ast.parse(src)
        stack: list[tuple[ast.AST, bool]] = [(tree, False)]
        maxmin = 0
        unguarded_json = 0
        cert_dict_eq = 0
        while stack:
            node, in_try = stack.pop()
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name) and fn.id in ("max", "min"):
                    maxmin += 1
                elif (
                    isinstance(fn, ast.Attribute)
                    and isinstance(fn.value, ast.Name)
                    and fn.value.id == "json"
                    and fn.attr in ("load", "loads")
                    and not in_try
                ):
                    unguarded_json += 1
            elif isinstance(node, ast.Compare):
                if any(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops):
                    sides = [node.left, *node.comparators]
                    has_cert = any(isinstance(s, ast.Name) and s.id in ("cert", "saved", "fresh") for s in sides)
                    has_json = any(
                        isinstance(s, ast.Call)
                        and isinstance(s.func, ast.Attribute)
                        and isinstance(s.func.value, ast.Name)
                        and s.func.value.id == "json"
                        and s.func.attr in ("load", "loads")
                        for s in sides
                    )
                    if has_cert and has_json:
                        cert_dict_eq += 1
            if isinstance(node, ast.Try):
                for child in node.handlers + node.orelse + node.finalbody:
                    stack.append((child, in_try))
                for child in node.body:
                    stack.append((child, True))
            else:
                for child in ast.iter_child_nodes(node):
                    stack.append((child, in_try))
        _check(
            (maxmin, cert_dict_eq, unguarded_json) == (0, 0, 0),
            f"residual unguarded exposure in verify.py: maxmin={maxmin} dict_eq={cert_dict_eq} json={unguarded_json}",
        )
        islinks = [
            n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and n.attr == "islink"
        ]
        isdigits = [
            n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and n.attr == "isdigit"
        ]
        _check(not islinks and not isdigits, f"forbidden islink={islinks} or isdigit={isdigits} in verify.py")

    def test_02_strict_finite_nan_swallowing_and_public_api_guards(self) -> None:
        v = _load_verify_module()

        def expect_fail(fn, label: str, *, substr: str | None = None) -> None:
            try:
                fn()
            except v.CheckFailure as exc:
                if substr is not None:
                    _check(substr in str(exc), f"{label}: expected {substr!r} in {str(exc)!r}")
                return
            raise RuntimeError(f"Expected CheckFailure for {label}")

        # strict_finite exact type, overflow, finiteness, positive/non_negative boundaries
        _check(v.strict_finite(0, "x", non_negative=True) == 0.0, "0 non_negative failed")
        _check(v.strict_finite(1.5, "x", positive=True) == 1.5, "1.5 positive failed")
        _check(v.strict_finite(np.float64(2.5), "x", positive=True) == 2.5, "np.float64 failed")
        _check(v.strict_finite(np.int64(3), "x", positive=True) == 3.0, "np.int64 failed")
        expect_fail(lambda: v.strict_finite(True, "x"), "bool True")
        expect_fail(lambda: v.strict_finite(False, "x"), "bool False")
        expect_fail(lambda: v.strict_finite("1.0", "x"), "str input")
        expect_fail(lambda: v.strict_finite(10**400, "x"), "int overflow")
        expect_fail(lambda: v.strict_finite(float("nan"), "x"), "NaN")
        expect_fail(lambda: v.strict_finite(float("inf"), "x"), "+Inf")
        expect_fail(lambda: v.strict_finite(float("-inf"), "x"), "-Inf")
        expect_fail(lambda: v.strict_finite(0.0, "x", positive=True), "0.0 positive")
        expect_fail(lambda: v.strict_finite(-0.1, "x", positive=True), "-0.1 positive")
        expect_fail(lambda: v.strict_finite(-0.1, "x", non_negative=True), "-0.1 non_negative")

        # strict_finite_max: closes fibral-rigidity NaN swallowing (verify.py:206-207)
        expect_fail(lambda: v.strict_finite_max(name="empty"), "empty strict_finite_max")
        _check(v.strict_finite_max(1.25, name="one") == 1.25, "single-arg strict_finite_max")
        _check(v.strict_finite_max(1.0, 3.0, 2.0, name="multi") == 3.0, "multi strict_finite_max")
        _check(v.strict_finite_max(3.0, 1.0, name="desc") == 3.0, "desc strict_finite_max")
        _check(v.strict_finite_max(2.0, 2.0, name="tie") == 2.0, "tie strict_finite_max")
        expect_fail(lambda: v.strict_finite_max(1.63e-16, float("nan"), 0.0, name="err"), "NaN in 2nd arg")
        expect_fail(lambda: v.strict_finite_max(float("nan"), 0.0, name="err"), "NaN in 1st arg")
        expect_fail(lambda: v.strict_finite_max(0.0, float("inf"), name="err"), "Inf in max")

        # sanitize: -0.0 normalization, bounded int, np.int64, non-finite/huge-int/cyclic/bad-key rejection
        max_ok_int = (1 << v.MAX_INT_BITS) - 1
        san = v.sanitize({"a": [-0.0, 1.5, np.float64(-0.0), np.int64(7), True, None, max_ok_int], "s": "ok"})
        _check(san["a"][0] == 0.0 and math.copysign(1.0, san["a"][0]) > 0.0, "-0.0 not normalized")
        _check(san["a"][2] == 0.0 and math.copysign(1.0, san["a"][2]) > 0.0, "np -0.0 not normalized")
        _check(san["a"][1] == 1.5 and san["a"][3] == 7 and type(san["a"][3]) is int and san["s"] == "ok", "sanitize altered valid values")
        expect_fail(lambda: v.sanitize({"bad": float("nan")}), "sanitize NaN")
        expect_fail(lambda: v.sanitize([float("inf")]), "sanitize Inf")
        expect_fail(lambda: v.sanitize({"huge": 1 << v.MAX_INT_BITS}), "sanitize huge int")
        expect_fail(lambda: v.sanitize({"huge": 10**4300}), "sanitize 10**4300")
        expect_fail(lambda: v.sanitize({1: "int_key"}), "sanitize non-str key")
        expect_fail(lambda: v.sanitize({"bad": "\ud800"}), "sanitize lone surrogate val")
        expect_fail(lambda: v.sanitize({"\ud800": 1}), "sanitize lone surrogate key")
        expect_fail(lambda: v.sanitize((1, 2)), "sanitize unsupported tuple")
        cyc_d: dict[str, object] = {}
        cyc_d["self"] = cyc_d
        expect_fail(lambda: v.sanitize(cyc_d), "sanitize cyclic dict")
        # Exact MAX_JSON_DEPTH boundary in sanitize (64 passes, 65 fails -> kills M010!)
        san_ok: Any = 1
        for _ in range(v.MAX_JSON_DEPTH):
            san_ok = [san_ok]
        v.sanitize(san_ok)
        expect_fail(lambda: v.sanitize([san_ok]), "sanitize depth 65")
        expect_fail(lambda: v.canonical_data_sha256({"big": 10**4300}), "canonical_data_sha256 10**4300")
        expect_fail(lambda: v.canonical_data_sha256(cyc_d), "canonical_data_sha256 cycle")
        expect_fail(lambda: v.canonical_data_sha256([]), "canonical_data_sha256 non-dict")
        expect_fail(lambda: v.canonical_data_sha256({1: "bad"}), "canonical_data_sha256 int key")
        expect_fail(lambda: v.canonical_data_sha256({"bad": (1, 2)}), "canonical_data_sha256 tuple")
        expect_fail(lambda: v.canonical_data_sha256({"f": float("nan")}), "canonical_data_sha256 NaN")
        expect_fail(lambda: v.render_canonical_cert_bytes([]), "render_canonical_cert_bytes non-dict")
        expect_fail(lambda: v.render_canonical_cert_bytes({"big": 10**4300}), "render_canonical_cert_bytes 10**4300")
        _check(len(v.canonical_data_sha256({"ok": [max_ok_int, True, None, 1.5, "s"]})) == 64, "canonical_data_sha256 valid")

        # _strict_parse_int explicit digit cap (D5)
        _check(v._strict_parse_int("9" * v.MAX_INT_DIGITS) == int("9" * v.MAX_INT_DIGITS), "exact MAX_INT_DIGITS failed")
        _check(v._strict_parse_int("-42") == -42, "negative int failed")
        for bad_tok in ("", "-", "9" * (v.MAX_INT_DIGITS + 1), "9" * 4200, "1\u0662", 123):
            expect_fail(lambda tok=bad_tok: v._strict_parse_int(tok), f"_strict_parse_int({bad_tok!r})")

        # build_fn_bars boundaries & NaN rejection (incl. float n=2.0 killing type(n) is int BoolOp!)
        c2, b2 = v.build_fn_bars(2, 0.5, 1.0, 1.0, 1.0, A_diag=0.0)
        _check(c2.shape == (4, 2) and len(b2) == 6, "build_fn_bars n=2 failed")
        expect_fail(lambda: v.build_fn_bars(1, 0.5, 1.0, 1.0, 1.0), "n=1")
        expect_fail(lambda: v.build_fn_bars(2.0, 0.5, 1.0, 1.0, 1.0), "float n=2.0")
        expect_fail(lambda: v.build_fn_bars(True, 0.5, 1.0, 1.0, 1.0), "bool n")
        expect_fail(lambda: v.build_fn_bars(3, 1.0, 1.0, 1.0, 1.0), "t == 1.0")
        expect_fail(lambda: v.build_fn_bars(3, 0.0, 1.0, 1.0, 1.0), "t == 0.0")
        expect_fail(lambda: v.build_fn_bars(3, float("nan"), 1.0, 1.0, 1.0), "t == nan")
        expect_fail(lambda: v.build_fn_bars(3, 0.5, 0.0, 1.0, 1.0), "E1 == 0")
        expect_fail(lambda: v.build_fn_bars(3, 0.5, 1.0, 0.0, 1.0), "A_intra == 0")
        expect_fail(lambda: v.build_fn_bars(3, 0.5, 1.0, 1.0, 0.0), "A_cross == 0")
        expect_fail(lambda: v.build_fn_bars(3, 0.5, 1.0, 1.0, 1.0, A_diag=-0.1), "A_diag < 0")

        # homogenize_periodic_truss guards (coords shape/type, bars, node indices, shift shape, NaN, zero-length, non-finite C_out)
        expect_fail(lambda: v.homogenize_periodic_truss(1, np.zeros((2, 1)), [(0, 1, np.ones(1), 1.0)]), "homogenize n=1")
        expect_fail(lambda: v.homogenize_periodic_truss(2.0, c2, b2), "homogenize float n=2.0")
        expect_fail(lambda: v.homogenize_periodic_truss(True, c2, b2), "homogenize bool n")
        expect_fail(lambda: v.homogenize_periodic_truss(2, c2, b2, cell_edge=0.0), "cell_edge == 0")
        expect_fail(lambda: v.homogenize_periodic_truss(2, [[0.0, 0.0], [1.0, 0.0]], b2), "non-ndarray coords")
        expect_fail(lambda: v.homogenize_periodic_truss(2, np.zeros((4,)), b2), "1D coords")
        expect_fail(lambda: v.homogenize_periodic_truss(2, np.zeros((4, 3)), b2), "wrong cols coords")
        expect_fail(lambda: v.homogenize_periodic_truss(2, np.zeros((1, 2)), [(0, 0, np.array([2.0, 0.0]), 1.0)]), "1-node coords")
        expect_fail(lambda: v.homogenize_periodic_truss(2, c2, []), "empty bars")
        expect_fail(lambda: v.homogenize_periodic_truss(2, c2, _SubList(b2)), "sublist bars")
        expect_fail(lambda: v.homogenize_periodic_truss(2, c2, [(0, 1, np.zeros(2))]), "short bar tuple")
        expect_fail(lambda: v.homogenize_periodic_truss(2, c2, [_SubTuple(b2[0])]), "subtuple bar entry")
        expect_fail(lambda: v.homogenize_periodic_truss(2, c2, [(True, 1, np.zeros(2), 1.0)]), "bool bar u")
        expect_fail(lambda: v.homogenize_periodic_truss(2, c2, [(9, 1, np.zeros(2), 1.0)]), "oob bar u")
        expect_fail(lambda: v.homogenize_periodic_truss(2, c2, [(0, True, np.zeros(2), 1.0)]), "bool bar v")
        expect_fail(lambda: v.homogenize_periodic_truss(2, c2, [(0, 9, np.zeros(2), 1.0)]), "oob bar v")
        expect_fail(lambda: v.homogenize_periodic_truss(2, c2, [(0, 1, [0.0, 0.0], 1.0)]), "non-ndarray shift")
        expect_fail(lambda: v.homogenize_periodic_truss(2, c2, [(0, 1, np.zeros(3), 1.0)]), "wrong shift shape")
        c_nan = c2.copy(); c_nan[0, 0] = float("nan")
        expect_fail(lambda: v.homogenize_periodic_truss(2, c_nan, b2), "NaN coords", substr="coords must contain only finite entries")
        b_nan_ea = list(b2); b_nan_ea[0] = (b2[0][0], b2[0][1], b2[0][2], float("nan"))
        expect_fail(lambda: v.homogenize_periodic_truss(2, c2, b_nan_ea), "NaN bar EA", substr="bar_EA must be finite")
        b_nan_sh = list(b2); sh_nan = b2[0][2].copy(); sh_nan[0] = float("nan"); b_nan_sh[0] = (b2[0][0], b2[0][1], sh_nan, b2[0][3])
        expect_fail(lambda: v.homogenize_periodic_truss(2, c2, b_nan_sh), "NaN bar shift", substr="bar shift must contain only finite entries")
        b_zero_len = list(b2); b_zero_len[0] = (0, 0, np.zeros(2), 1.0)
        expect_fail(lambda: v.homogenize_periodic_truss(2, c2, b_zero_len), "zero-length bar", substr="bar_length must be positive")
        orig_lstsq = np.linalg.lstsq
        try:
            np.linalg.lstsq = lambda a, b, rcond=None: (np.full((a.shape[1], b.shape[1]), float("nan")), np.array([]), 0, np.array([]))
            expect_fail(
                lambda: v.homogenize_periodic_truss(2, c2, b2),
                "non-finite C_out",
                substr="homogenized stiffness matrix contains non-finite values",
            )
        finally:
            np.linalg.lstsq = orig_lstsq

        # require_finite_stiffness_matrix guards
        expect_fail(lambda: v.require_finite_stiffness_matrix(np.zeros((1, 1)), 1, "m"), "stiff n=1")
        expect_fail(lambda: v.require_finite_stiffness_matrix(np.zeros((6, 6)), 3.0, "m"), "stiff float n")
        expect_fail(lambda: v.require_finite_stiffness_matrix([[0.0] * 6] * 6, 3, "m"), "stiff list")
        expect_fail(lambda: v.require_finite_stiffness_matrix(np.zeros((6,)), 3, "m"), "stiff 1D")
        expect_fail(lambda: v.require_finite_stiffness_matrix(np.zeros((5, 5)), 3, "m"), "stiff wrong shape")
        m_nan = np.zeros((6, 6)); m_nan[0, 2] = float("nan")
        expect_fail(lambda: v.require_finite_stiffness_matrix(m_nan, 3, "m"), "stiff NaN")

        # hashin_shtrikman_upper_bounds guards & boundaries
        k_hs, g_hs, e_hs = v.hashin_shtrikman_upper_bounds(100.0, 0.25, 1.0)
        _check(abs(e_hs - 100.0) < 1e-12 and k_hs > 0 and g_hs > 0, "HS phi=1.0 exact solid")
        expect_fail(lambda: v.hashin_shtrikman_upper_bounds(0.0, 0.25, 0.05), "Es == 0")
        expect_fail(lambda: v.hashin_shtrikman_upper_bounds(100.0, -1.0, 0.05), "nu_s == -1")
        expect_fail(lambda: v.hashin_shtrikman_upper_bounds(100.0, 0.5, 0.05), "nu_s == 0.5")
        expect_fail(lambda: v.hashin_shtrikman_upper_bounds(100.0, float("nan"), 0.05), "nu_s == nan")
        expect_fail(lambda: v.hashin_shtrikman_upper_bounds(100.0, 0.25, 0.0), "phi == 0")
        expect_fail(lambda: v.hashin_shtrikman_upper_bounds(100.0, 0.25, 1.01), "phi > 1")

        # D2: shear_within_bound checks ALL shear & normal-shear entries (C44, C55, C66, C34, C03, C30) & full 6x6 finiteness
        for (ri, ci) in ((3, 3), (4, 4), (5, 5), (3, 4), (0, 3), (3, 0)):
            mat = np.zeros((6, 6), dtype=float)
            mat[ri, ci] = v.NUMERICAL_BOUND
            _check(v.shear_within_bound(mat) is True, f"exact boundary at ({ri},{ci}) failed")
            mat[ri, ci] = 1.0
            _check(v.shear_within_bound(mat) is False, f"planted shear 1.0 at ({ri},{ci}) accepted")
            mat[ri, ci] = float("nan")
            expect_fail(lambda m=mat: v.shear_within_bound(m), f"shear_within_bound(NaN at ({ri},{ci}))")
        mat_n02 = np.zeros((6, 6), dtype=float); mat_n02[0, 2] = float("nan")
        expect_fail(lambda: v.shear_within_bound(mat_n02), "shear_within_bound NaN in normal block")
        mat0 = np.zeros((6, 6), dtype=float)
        _check(v.shear_within_bound(mat0, bound=0.0) is True, "zero bound exact zero failed")
        expect_fail(lambda: v.shear_within_bound(mat0, bound=-1e-9), "negative bound")

        # D1 & D2 live tamper injection into run_numerical_truss_checks & run_must_fail_controls!
        orig_hom = v.homogenize_periodic_truss
        try:
            for (ri, ci, bad_val, lbl) in (
                (0, 1, float("nan"), "D1 NaN C_mat[0,1]"),
                (0, 2, float("nan"), "D1 NaN C_mat[0,2]"),
                (0, 2, 1.0e6, "D1 1e6 C_mat[0,2]"),
                (0, 3, float("nan"), "D1 NaN C_mat[0,3]"),
                (0, 3, 1.0e6, "D1 1e6 C_mat[0,3]"),
            ):
                def _bad_unbraced(n, coords, bars, cell_edge=2.0, _r=ri, _c=ci, _v=bad_val):
                    C_m, ea = orig_hom(n, coords, bars, cell_edge=cell_edge)
                    if n >= 3:
                        C_m = C_m.copy()
                        C_m[_r, _c] = _v
                    return C_m, ea

                v.homogenize_periodic_truss = _bad_unbraced
                expect_fail(lambda: v.run_numerical_truss_checks(), lbl)

            # Braced C44==0, exact 1e-12 boundary, & off-diagonal shear tamper on BOTH braced & isotropic (D1)
            braced_call = [0]

            def _zero_braced_c44(n, coords, bars, cell_edge=2.0):
                C_m, ea = orig_hom(n, coords, bars, cell_edge=cell_edge)
                if n == 3 and len(bars) > 15:
                    braced_call[0] += 1
                    if braced_call[0] == 1:
                        C_m = C_m.copy()
                        C_m[3, 3] = 0.0
                        C_m[4, 4] = 0.0
                        C_m[5, 5] = 0.0
                return C_m, ea

            v.homogenize_periodic_truss = _zero_braced_c44
            expect_fail(lambda: v.run_numerical_truss_checks(), "D1 braced C44==0", substr="braced C44 must be positive")

            braced_call[0] = 0

            def _bad_braced_only(n, coords, bars, cell_edge=2.0):
                C_m, ea = orig_hom(n, coords, bars, cell_edge=cell_edge)
                if n == 3 and len(bars) > 15:
                    braced_call[0] += 1
                    if braced_call[0] == 1:
                        C_m = C_m.copy()
                        C_m[3, 4] = 1.0e6
                return C_m, ea

            v.homogenize_periodic_truss = _bad_braced_only
            expect_fail(lambda: v.run_numerical_truss_checks(), "D1 braced C_mat[3,4]=1e6", substr="braced stiffness matrix broke cubic symmetry")

            # Exact 1e-12 boundary on braced & isotropic MUST PASS (kills <= vs < mutants!)
            braced_call[0] = 0

            def _exact_1e12_boundary(n, coords, bars, cell_edge=2.0):
                C_m, ea = orig_hom(n, coords, bars, cell_edge=cell_edge)
                if n == 3 and len(bars) > 15:
                    braced_call[0] += 1
                    if braced_call[0] in (1, 74):  # 1=first braced, 74=final C_iso after 3 braced + 70 bisection
                        C_m = C_m.copy()
                        C_m[3, 4] = 1.0e-12
                return C_m, ea

            v.homogenize_periodic_truss = _exact_1e12_boundary
            v.run_numerical_truss_checks()

            # Isotropic-only C_iso[3,4]=1e6 tamper
            braced_call[0] = 0

            def _bad_iso_only(n, coords, bars, cell_edge=2.0):
                C_m, ea = orig_hom(n, coords, bars, cell_edge=cell_edge)
                if n == 3 and len(bars) > 15:
                    braced_call[0] += 1
                    if braced_call[0] == 74:
                        C_m = C_m.copy()
                        C_m[3, 4] = 1.0e6
                return C_m, ea

            v.homogenize_periodic_truss = _bad_iso_only
            expect_fail(lambda: v.run_numerical_truss_checks(), "D1 C_iso[3,4]=1e6", substr="isotropic stiffness matrix broke cubic symmetry")

            v.homogenize_periodic_truss = orig_hom
            num_ok = v.run_numerical_truss_checks()
            for (ri, ci) in ((3, 3), (4, 4), (5, 5), (3, 4), (0, 3)):
                def _planted_shear(n, coords, bars, cell_edge=2.0, _r=ri, _c=ci):
                    C_m, ea = orig_hom(n, coords, bars, cell_edge=cell_edge)
                    C_m = C_m.copy()
                    C_m[_r, _c] = 1.0
                    return C_m, ea

                v.homogenize_periodic_truss = _planted_shear
                expect_fail(lambda: v.run_must_fail_controls(num_ok), f"D2 planted shear C_eq[{ri},{ci}]=1.0")
        finally:
            v.homogenize_periodic_truss = orig_hom

    def test_03_zone_b_io_json_symlinks_fifo_cfrp_and_main_parity(self) -> None:
        v = _load_verify_module()
        disk_cert = json.loads(CERT_PATH.read_text(encoding="utf-8"))
        raw_data = v.load_strict_json_file(DATA_PATH, "data/carbon_fibers.json")
        _check(
            v.canonical_data_sha256(raw_data) == disk_cert["inputs"]["data/carbon_fibers.json_sha256"],
            "canonical_data_sha256 drifted from committed certificate",
        )

        # Live NumPy homogenization, CFRP, and controls match PR #3's committed certificate.json byte-for-byte
        num_res = v.run_numerical_truss_checks()
        cfrp_res = v.run_cfrp_benchmark_checks(num_res)
        ctrl_res = v.run_must_fail_controls(num_res)
        _check(num_res == disk_cert["numerical_homogenization"], "num_res != disk_cert")
        _check(cfrp_res == disk_cert["cfrp_material_benchmarks"], "cfrp_res != disk_cert")
        _check(ctrl_res == disk_cert["must_fail_controls"], "ctrl_res != disk_cert")

        def expect_fail(fn, label: str, *, substr: str | None = None) -> None:
            try:
                fn()
            except v.CheckFailure as exc:
                if substr is not None:
                    _check(substr in str(exc), f"{label}: expected {substr!r} in {str(exc)!r}")
                return
            raise RuntimeError(f"Expected CheckFailure for {label}")

        # CFRP dataset & num_res schema, NaN/bool & BoolOp SUBCOND_DEL killers
        expect_fail(lambda: v.run_cfrp_benchmark_checks([], raw_data=raw_data), "non-dict num_res")
        expect_fail(lambda: v.run_cfrp_benchmark_checks({"isotropic_braced_point": []}, raw_data=raw_data), "bad iso_pt in cfrp")
        expect_fail(lambda: v.run_must_fail_controls([]), "non-dict num_res in controls")
        expect_fail(lambda: v.run_must_fail_controls({"isotropic_braced_point": []}), "bad iso_pt in controls")
        expect_fail(lambda: v.run_cfrp_benchmark_checks(num_res, raw_data=["matrix_reference"]), "non-dict raw_data with key")
        bad = copy.deepcopy(raw_data); del bad["matrix_reference"]
        expect_fail(lambda: v.run_cfrp_benchmark_checks(num_res, raw_data=bad), "missing matrix_reference")
        bad = copy.deepcopy(raw_data); bad["matrix_reference"] = []
        expect_fail(lambda: v.run_cfrp_benchmark_checks(num_res, raw_data=bad), "non-dict matrix_reference", substr="matrix_reference must be a JSON object")
        for key in ("E_GPa", "rho_g_cm3", "fiber_volume_fraction_Vf"):
            bad = copy.deepcopy(raw_data); del bad["matrix_reference"][key]
            expect_fail(lambda: v.run_cfrp_benchmark_checks(num_res, raw_data=bad), f"missing matrix_reference.{key}")
        bad = copy.deepcopy(raw_data); bad["matrix_reference"]["fiber_volume_fraction_Vf"] = 1.0
        expect_fail(lambda: v.run_cfrp_benchmark_checks(num_res, raw_data=bad), "Vf == 1.0")
        bad = copy.deepcopy(raw_data); bad["reference_isotropic_solids"] = []
        expect_fail(lambda: v.run_cfrp_benchmark_checks(num_res, raw_data=bad), "empty metals")
        bad = copy.deepcopy(raw_data); bad["reference_isotropic_solids"] = tuple(raw_data["reference_isotropic_solids"])
        expect_fail(lambda: v.run_cfrp_benchmark_checks(num_res, raw_data=bad), "tuple metals")
        bad = copy.deepcopy(raw_data); bad["reference_isotropic_solids"][0] = ["id", "Es_GPa", "nu_s", "rho_s_g_cm3"]
        expect_fail(lambda: v.run_cfrp_benchmark_checks(num_res, raw_data=bad), "non-dict metal entry with keys")
        bad = copy.deepcopy(raw_data); del bad["reference_isotropic_solids"][0]["Es_GPa"]
        expect_fail(lambda: v.run_cfrp_benchmark_checks(num_res, raw_data=bad), "metal missing Es_GPa")
        bad = copy.deepcopy(raw_data); bad["reference_isotropic_solids"][0]["id"] = ""
        expect_fail(lambda: v.run_cfrp_benchmark_checks(num_res, raw_data=bad), "empty metal id")
        bad = copy.deepcopy(raw_data); bad["reference_isotropic_solids"][0]["id"] = 123
        expect_fail(lambda: v.run_cfrp_benchmark_checks(num_res, raw_data=bad), "non-str metal id")
        bad = copy.deepcopy(raw_data); bad["reference_isotropic_solids"][0]["Es_GPa"] = float("nan")
        expect_fail(lambda: v.run_cfrp_benchmark_checks(num_res, raw_data=bad), "metal NaN Es_GPa")
        bad = copy.deepcopy(raw_data); bad["carbon_fibers"] = []
        expect_fail(lambda: v.run_cfrp_benchmark_checks(num_res, raw_data=bad), "empty fibers")
        bad = copy.deepcopy(raw_data); bad["carbon_fibers"] = tuple(raw_data["carbon_fibers"])
        expect_fail(lambda: v.run_cfrp_benchmark_checks(num_res, raw_data=bad), "tuple fibers")
        bad = copy.deepcopy(raw_data); bad["carbon_fibers"][0] = ["id", "Ef_GPa", "sigma_ult_MPa", "rho_f_g_cm3"]
        expect_fail(lambda: v.run_cfrp_benchmark_checks(num_res, raw_data=bad), "non-dict fiber entry with keys")
        bad = copy.deepcopy(raw_data); del bad["carbon_fibers"][0]["Ef_GPa"]
        expect_fail(lambda: v.run_cfrp_benchmark_checks(num_res, raw_data=bad), "fiber missing Ef_GPa")
        bad = copy.deepcopy(raw_data); bad["carbon_fibers"][0]["id"] = ""
        expect_fail(lambda: v.run_cfrp_benchmark_checks(num_res, raw_data=bad), "empty fiber id")
        bad = copy.deepcopy(raw_data); bad["carbon_fibers"][0]["id"] = 123
        expect_fail(lambda: v.run_cfrp_benchmark_checks(num_res, raw_data=bad), "non-str fiber id")
        bad = copy.deepcopy(raw_data); bad["carbon_fibers"][0]["Ef_GPa"] = float("nan")
        expect_fail(lambda: v.run_cfrp_benchmark_checks(num_res, raw_data=bad), "fiber NaN Ef_GPa")

        # Zone-B file I/O, JSON, symlink ancestor, hardlink, NUL/surrogate, streaming, concurrency & canonical cert checks
        with tempfile.TemporaryDirectory(prefix="fibral_zb_") as td:
            root = Path(td)
            real_cert = root / "certificate.json"
            expected_bytes = CERT_PATH.read_bytes()
            v.write_canonical_cert_bytes(real_cert, expected_bytes)
            _check(stat.S_IMODE(real_cert.stat().st_mode) == 0o600, "certificate temp/final mode must be 0o600")
            # Overwriting existing single-link target (st_nlink == 1) succeeds cleanly
            v.write_canonical_cert_bytes(real_cert, expected_bytes)
            v.verify_canonical_cert_bytes(real_cert, expected_bytes)

            # D3: Hardlink target (st_nlink > 1) MUST be refused without clobbering original!
            hl_cert = root / "hl_cert.json"
            os.link(real_cert, hl_cert)
            expect_fail(
                lambda: v.write_canonical_cert_bytes(hl_cert, b'{"tampered":true}\n'),
                "D3 hardlink write refusal",
                substr="hardlinked certificate path",
            )
            _check(real_cert.read_bytes() == expected_bytes, "D3 hardlink clobbered original")
            os.unlink(hl_cert)

            # D3: Non-bytes payload rejected BEFORE opening dir; verify_canonical_cert_bytes rejects non-bytes
            for bad_payload in ("str", bytearray(b"{}")):
                expect_fail(
                    lambda bp=bad_payload: v.write_canonical_cert_bytes(root / "no_dir" / "c.json", bp),
                    f"non-bytes write payload {type(bad_payload).__name__}",
                )
            expect_fail(lambda: v.verify_canonical_cert_bytes(real_cert, "not_bytes"), "non-bytes expected_bytes")

            # D3: _next_tmp_name monotonic counter, EEXIST retry + exhaustion, pre-replace st_nlink!=1 & 16T stress
            tn1 = v._next_tmp_name("cert.json")
            tn2 = v._next_tmp_name("cert.json")
            _check(tn1 != tn2 and int(tn2.split(".")[-2]) == int(tn1.split(".")[-2]) + 1, "_next_tmp_name counter")
            col_target = root / "col.json"
            (root / ".col.json.frtmp.fixed0").write_bytes(b"occ")
            with mock.patch.object(v, "_next_tmp_name", side_effect=[".col.json.frtmp.fixed0", ".col.json.frtmp.fixed1"]):
                v.write_canonical_cert_bytes(col_target, b"{}\n")
            _check(col_target.read_bytes() == b"{}\n", "EEXIST retry failed")
            (root / ".col2.json.frtmp.occ").write_bytes(b"x")
            with mock.patch.object(v, "_next_tmp_name", return_value=".col2.json.frtmp.occ"):
                expect_fail(lambda: v.write_canonical_cert_bytes(root / "col2.json", b"{}\n"), "EEXIST exhaustion")

            real_fstat = os.fstat

            def _fake_fstat(fd: int) -> os.stat_result:
                st = real_fstat(fd)
                return os.stat_result((st.st_mode, st.st_ino, st.st_dev, 2, st.st_uid, st.st_gid, st.st_size, st.st_atime, st.st_mtime, st.st_ctime))

            with mock.patch.object(v.os, "fstat", side_effect=_fake_fstat):
                expect_fail(
                    lambda: v.write_canonical_cert_bytes(root / "hltmp.json", b"{}\n"),
                    "pre-replace temp hardlink guard",
                    substr="hardlinked temp file",
                )
            _check(list(root.glob(".hltmp.json.frtmp.*")) == [], "leftover temp after st_nlink != 1")

            conc_target = root / "conc.json"
            conc_errs: list[str] = []
            conc_lock = threading.Lock()
            conc_ok = 0

            def _cw(tid: int) -> None:
                nonlocal conc_ok
                b_out = f'{{"tid":{tid}}}\n'.encode("ascii")
                for _ in range(2):
                    try:
                        v.write_canonical_cert_bytes(conc_target, b_out)
                        with conc_lock:
                            conc_ok += 1
                    except Exception as exc:  # pragma: no cover
                        with conc_lock:
                            conc_errs.append(f"{tid}:{exc}")

            ths = [threading.Thread(target=_cw, args=(i,)) for i in range(16)]
            for th in ths:
                th.start()
            for th in ths:
                th.join()
            _check(conc_errs == [] and conc_ok == 32 and list(root.glob(".conc.json.frtmp.*")) == [], f"16T write failed: {conc_errs}")

            # D4: Embedded NUL byte & lone UTF-16 surrogate in path raise CheckFailure (zero raw ValueError!)
            for bad_p in (Path(f"{td}/a\x00b.json"), Path(f"{td}/a\ud800b.json")):
                expect_fail(lambda bp=bad_p: v.read_regular_bytes(bp, "nul"), "D4 read bad path")
                expect_fail(lambda bp=bad_p: v.write_canonical_cert_bytes(bp, b"{}\n"), "D4 write bad path")

            # Exact size cap boundary, >64 KiB streaming read, & max_bytes cap rejection (incl. float 2.0 on 2-byte file!)
            cap_file = root / "cap.json"
            cap_file.write_bytes(b"{}")
            _check(v.read_regular_bytes(cap_file, "cap", max_bytes=2) == b"{}", "exact max_bytes failed")
            expect_fail(lambda: v.read_regular_bytes(cap_file, "cap", max_bytes=1), "max_bytes + 1")
            expect_fail(lambda: v.read_regular_bytes(cap_file, "cap", max_bytes=0), "max_bytes == 0")
            expect_fail(lambda: v.read_regular_bytes(cap_file, "cap", max_bytes=1 << 100), "max_bytes == 1<<100")
            expect_fail(lambda: v.read_regular_bytes(cap_file, "cap", max_bytes=2.0), "max_bytes == float 2.0")
            expect_fail(lambda: v.read_regular_bytes(cap_file, "cap", max_bytes=True), "max_bytes == True")
            # Missing parent dir on read MUST NOT create it (kills L148 create_missing BoolOp!)
            no_dir = root / "must_not_create"
            expect_fail(lambda: v.read_regular_bytes(no_dir / "c.json", "nodir"), "missing parent dir read", substr="missing or unreadable")
            _check(not no_dir.exists(), "read_regular_bytes created missing parent dir")
            stream_file = root / "stream140k.bin"
            blob140 = b"x" * 140_000
            stream_file.write_bytes(blob140)
            _check(v.read_regular_bytes(stream_file, "stream", max_bytes=140_000) == blob140, "140 KiB streaming read failed")
            expect_fail(lambda: v.read_regular_bytes(stream_file, "stream", max_bytes=139_999), "140 KiB overflow")

            # Concrete Path rejection (_EvilPath & str) and root Path("/")
            expect_fail(lambda: v.read_regular_bytes(_EvilPath(real_cert), "evil"), "_EvilPath read")
            expect_fail(lambda: v.read_regular_bytes(str(real_cert), "str"), "str read")
            expect_fail(lambda: v.write_canonical_cert_bytes(_EvilPath(real_cert), expected_bytes), "_EvilPath write")
            expect_fail(lambda: v.read_regular_bytes(Path("/"), "root"), "root dir read")
            expect_fail(lambda: v.write_canonical_cert_bytes(Path("/"), expected_bytes), "root dir write")

            # Missing file & directory (ZB15) & FIFO (D2 non-blocking!)
            expect_fail(lambda: v.read_regular_bytes(root / "missing.json", "missing"), "missing file", substr="missing or unreadable")
            dpath = root / "adir"
            dpath.mkdir()
            expect_fail(lambda: v.verify_canonical_cert_bytes(dpath, expected_bytes), "ZB15 directory read", substr="not a regular file")
            expect_fail(lambda: v.write_canonical_cert_bytes(dpath, expected_bytes), "ZB15 directory write", substr="non-regular certificate path")
            if hasattr(os, "mkfifo"):
                fifo = root / "pipe.json"
                os.mkfifo(fifo)
                expect_fail(lambda: v.read_regular_bytes(fifo, "fifo"), "FIFO read", substr="not a regular file")
                expect_fail(lambda: v.write_canonical_cert_bytes(fifo, expected_bytes), "FIFO write", substr="non-regular certificate path")

            # Symlink target (ZB14), symlink parent, deep symlink grandparent, & ..-collapsed symlink (S21C-BUG-01/D1)
            sym_file = root / "sym.json"
            os.symlink(real_cert, sym_file)
            expect_fail(lambda: v.verify_canonical_cert_bytes(sym_file, expected_bytes), "ZB14 symlink cert", substr="symlink path or ancestor")
            expect_fail(lambda: v.write_canonical_cert_bytes(sym_file, expected_bytes), "symlink write", substr="symlink path or ancestor")

            real_sub = root / "real_sub" / "nested"
            real_sub.mkdir(parents=True)
            (real_sub / "c.json").write_bytes(expected_bytes)
            sym_parent = root / "sym_parent"
            os.symlink(root / "real_sub", sym_parent)
            expect_fail(lambda: v.verify_canonical_cert_bytes(sym_parent / "nested" / "c.json", expected_bytes), "deep symlink ancestor", substr="symlink path or ancestor")
            expect_fail(lambda: v.write_canonical_cert_bytes(sym_parent / "nested" / "c.json", expected_bytes), "write deep symlink ancestor", substr="symlink path or ancestor")
            expect_fail(lambda: v.verify_canonical_cert_bytes(sym_parent / ".." / "real_sub" / "nested" / "c.json", expected_bytes), "dotdot collapsed symlink", substr="symlink path or ancestor")

            # Relative path with symlinked PWD, stale missing PWD, normal PWD, relative/unset PWD & relative write (len(parts)==1!)
            orig_cwd = os.getcwd()
            orig_pwd = os.environ.get("PWD")
            try:
                os.chdir(real_sub)
                os.environ["PWD"] = str(sym_parent / "nested")
                expect_fail(lambda: v.verify_canonical_cert_bytes(Path("c.json"), expected_bytes), "symlink PWD relative path")
                os.environ["PWD"] = str(root / "stale_missing_pwd")
                v.verify_canonical_cert_bytes(Path("c.json"), expected_bytes)
                os.environ["PWD"] = "relative/pwd"
                v.verify_canonical_cert_bytes(Path("c.json"), expected_bytes)
                os.environ["PWD"] = str(real_sub)
                v.verify_canonical_cert_bytes(Path("c.json"), expected_bytes)
                v.write_canonical_cert_bytes(Path("rel_written.json"), expected_bytes)
                _check((real_sub / "rel_written.json").read_bytes() == expected_bytes, "relative write failed")
                os.environ.pop("PWD", None)
                v.verify_canonical_cert_bytes(Path("c.json"), expected_bytes)
            finally:
                os.chdir(orig_cwd)
                if orig_pwd is None:
                    os.environ.pop("PWD", None)
                else:
                    os.environ["PWD"] = orig_pwd

            # Zone-B canonical cert mismatches: ZB06 (true->1), ZB07 (dup keys), ZB08 (int->.0), ZB10 (CRLF)
            crlf_cert = root / "crlf.json"
            crlf_cert.write_bytes(expected_bytes.replace(b"\n", b"\r\n"))
            expect_fail(lambda: v.verify_canonical_cert_bytes(crlf_cert, expected_bytes), "ZB10 CRLF cert")
            zb06 = root / "zb06.json"
            zb06.write_bytes(expected_bytes.replace(b'"rejected": true', b'"rejected": 1', 1))
            expect_fail(lambda: v.verify_canonical_cert_bytes(zb06, expected_bytes), "ZB06 true->1")
            zb08 = root / "zb08.json"
            zb08.write_bytes(expected_bytes.replace(b'"coordinate_homogenization_cases_checked": 60', b'"coordinate_homogenization_cases_checked": 60.0', 1))
            expect_fail(lambda: v.verify_canonical_cert_bytes(zb08, expected_bytes), "ZB08 int->float")

            # Strict JSON loader: ZB03 (5000-digit int), ZB07 (dup key), ZB16 (bad UTF-8 / truncated),
            # NaN/Inf/1e999, non-dict root, lone surrogates (key & value), depth 64 pass vs depth 65 fail
            for lbl, raw_b in (
                ("ZB16 invalid utf8", b"\xff\xfe"),
                ("ZB16 truncated", b'{"a": '),
                ("D5 1001-digit int", ('{"n": ' + "9" * 1001 + "}").encode("ascii")),
                ("D5 4200-digit int", ('{"n": ' + "9" * 4200 + "}").encode("ascii")),
                ("ZB03 5000-digit int", ('{"n": ' + "9" * 5000 + "}").encode("ascii")),
                ("ZB07 duplicate key", b'{"a": 1, "a": 2}'),
                ("NaN constant", b'{"a": NaN}'),
                ("1e999 float overflow", b'{"a": 1e999}'),
                ("list root", b"[1, 2, 3]"),
                ("lone surrogate value", b'{"a": "\\ud800"}'),
                ("lone surrogate key", b'{"\\ud800": 1}'),
            ):
                p = root / "bad.json"
                p.write_bytes(raw_b)
                expect_fail(lambda p=p: v.load_strict_json_file(p, "bad"), lbl)

            # Exact MAX_JSON_DEPTH boundary (64 passes, 65 fails)
            ok_depth: Any = 1
            for _ in range(v.MAX_JSON_DEPTH - 1):
                ok_depth = [ok_depth]
            p_ok = root / "d64.json"
            p_ok.write_text(json.dumps({"x": ok_depth}), encoding="utf-8")
            _check(type(v.load_strict_json_file(p_ok, "d64")) is dict, "depth == 64 failed")
            p_deep = root / "d65.json"
            p_deep.write_text(json.dumps({"x": [ok_depth]}), encoding="utf-8")
            expect_fail(lambda: v.load_strict_json_file(p_deep, "d65"), "depth == 65")

            # Unreadable file / write OSError (if non-root)
            unreadable = root / "ro.json"
            unreadable.write_text("{}", encoding="utf-8")
            os.chmod(unreadable, 0o000)
            try:
                if os.geteuid() != 0:
                    expect_fail(lambda: v.read_regular_bytes(unreadable, "unreadable"), "unreadable file")
            finally:
                os.chmod(unreadable, 0o600)

            ro_dir = root / "ro_dir"
            ro_dir.mkdir()
            os.chmod(ro_dir, 0o500)
            try:
                if os.geteuid() != 0:
                    expect_fail(lambda: v.write_canonical_cert_bytes(ro_dir / "c.json", expected_bytes), "read-only parent write")
            finally:
                os.chmod(ro_dir, 0o700)

            # Full main() end-to-end check (including --write + committed cert parity + invalid argv + CheckFailure exit 1)
            orig_sym = v.run_symbolic_checks
            orig_cert_path = v.CERT_PATH
            try:
                if not HAS_SYMPY:
                    v.run_symbolic_checks = lambda: copy.deepcopy(disk_cert["symbolic_verification"])

                # 1. Clean verify against real committed results/certificate.json
                out_buf, err_buf = io.StringIO(), io.StringIO()
                with redirect_stdout(out_buf), redirect_stderr(err_buf):
                    rc = v.main([])
                n_ctrl = len(disk_cert["must_fail_controls"])
                _check(rc == 0 and "[PASS] Symbolic (n=2..7)" in out_buf.getvalue() and f"Controls ({n_ctrl}/{n_ctrl}): OK" in out_buf.getvalue(),
                       f"clean main([]) failed: rc={rc} out={out_buf.getvalue()!r} err={err_buf.getvalue()!r}")

                # 2. --write to temp path produces byte-identical certificate to committed results/certificate.json
                v.CERT_PATH = root / "written" / "certificate.json"
                out_buf, err_buf = io.StringIO(), io.StringIO()
                with redirect_stdout(out_buf), redirect_stderr(err_buf):
                    rc_w = v.main(["--write"])
                _check(rc_w == 0 and v.CERT_PATH.read_bytes() == expected_bytes, "--write bytes drifted from committed cert")

                # 3. Tampered certificate returns rc=1 with [FAIL] CheckFailure (zero raw traceback)
                v.CERT_PATH.write_bytes(expected_bytes.replace(b"\n", b"\r\n"))
                out_buf, err_buf = io.StringIO(), io.StringIO()
                with redirect_stdout(out_buf), redirect_stderr(err_buf):
                    rc_bad = v.main([])
                _check(rc_bad == 1 and "[FAIL] CheckFailure:" in err_buf.getvalue(), f"tampered main rc={rc_bad} err={err_buf.getvalue()!r}")

                # 4. Bad argv types & lone surrogates fail closed with rc=1 (zero TypeError traceback)
                for bad_argv in ("--write", [123], [_StrSub("--write")], ["bad\ud800"]):
                    err_buf = io.StringIO()
                    with redirect_stderr(err_buf):
                        rc_a = v.main(bad_argv)
                    _check(rc_a == 1 and "[FAIL] CheckFailure:" in err_buf.getvalue(), f"bad argv {bad_argv!r} rc={rc_a}")
            finally:
                v.run_symbolic_checks = orig_sym
                v.CERT_PATH = orig_cert_path

    def test_04_isolated_5mode_subprocess_planted_shear_and_deleted_cwd(self) -> None:
        """Subprocess regression across 5 modes incl. -I -O -B -W error & PYTHONHASHSEED=42 -B (without -I)."""
        runner = (
            "import sys, types\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "import importlib.util\n"
            "if importlib.util.find_spec('sympy') is None:\n"
            "    sys.modules['sympy'] = types.ModuleType('sympy')\n"
            "import verify\n"
            "num = verify.run_numerical_truss_checks()\n"
            "orig = verify.homogenize_periodic_truss\n"
            "for (r, c) in ((3, 3), (4, 4), (5, 5), (3, 4)):\n"
            "    def planted(n, coords, bars, cell_edge=2.0, _r=r, _c=c):\n"
            "        C, ea = orig(n, coords, bars, cell_edge=cell_edge)\n"
            "        C = C.copy(); C[_r, _c] = 1.0\n"
            "        return C, ea\n"
            "    verify.homogenize_periodic_truss = planted\n"
            "    try:\n"
            "        verify.run_must_fail_controls(num)\n"
            "        raise SystemExit(2)\n"
            "    except verify.CheckFailure:\n"
            "        pass\n"
            "print('REJECTED_ALL_PLANTED_SHEARS')\n"
            "raise SystemExit(0)\n"
        )
        mode_specs = [
            (["-I", "-B"], None),
            (["-I", "-O", "-B"], None),
            (["-I", "-B", "-W", "error"], None),
            (["-I", "-O", "-B", "-W", "error"], None),
            (["-B"], {**os.environ, "PYTHONHASHSEED": "42"}),
        ]
        for flags, env in mode_specs:
            proc = subprocess.run([sys.executable, *flags, "-c", runner], env=env, capture_output=True, text=True, timeout=60)
            _check(
                proc.returncode == 0 and "REJECTED_ALL_PLANTED_SHEARS" in proc.stdout,
                f"planted shear subprocess failed under {flags}: rc={proc.returncode} out={proc.stdout!r} err={proc.stderr!r}",
            )

        # Deleted-CWD fail-closed check (zero raw FileNotFoundError!)
        with tempfile.TemporaryDirectory(prefix="fibral_delcwd_") as td:
            vanish = Path(td) / "vanishing"
            vanish.mkdir()
            del_script = (
                "import os, sys, types; from pathlib import Path\n"
                f"sys.path.insert(0, {str(ROOT)!r})\n"
                "import importlib.util\n"
                "if importlib.util.find_spec('sympy') is None:\n"
                "    sys.modules['sympy'] = types.ModuleType('sympy')\n"
                "import verify\n"
                f"os.chdir({str(vanish)!r}); os.rmdir({str(vanish)!r})\n"
                "try:\n"
                "    verify.read_regular_bytes(Path('rel.json'), 'rel')\n"
                "    raise SystemExit(2)\n"
                "except verify.CheckFailure as exc:\n"
                "    print('DELCWD_CAUGHT:', exc)\n"
                "    raise SystemExit(0)\n"
            )
            proc2 = subprocess.run([sys.executable, "-B", "-c", del_script], capture_output=True, text=True, timeout=15)
            _check(
                proc2.returncode == 0 and "CWD unavailable" in proc2.stdout,
                f"deleted CWD check failed: rc={proc2.returncode} out={proc2.stdout!r} err={proc2.stderr!r}",
            )

    def test_05_darwin_system_root_alias_normalization(self) -> None:
        v = _load_verify_module()

        root_lnk = types.SimpleNamespace(st_mode=stat.S_IFLNK | 0o777, st_uid=0)
        user_lnk = types.SimpleNamespace(st_mode=stat.S_IFLNK | 0o777, st_uid=1000)

        def _fake_readlink(p: str) -> str:
            if p == "/var":
                return "private/var"
            if p == "/tmp":
                return "private/tmp"
            if p == "/etc":
                return "private/etc"
            raise OSError("unexpected readlink")

        with (
            mock.patch.object(v.sys, "platform", "darwin"),
            mock.patch.object(v.os, "lstat", return_value=root_lnk),
            mock.patch.object(v.os, "readlink", side_effect=_fake_readlink),
        ):
            _check(
                v._validate_path_and_split(Path("/var/folders/xx/cert.json"), "t")
                == ("/", ["private", "var", "folders", "xx", "cert.json"]),
                "darwin /var normalization failed",
            )
            _check(
                v._validate_path_and_split(Path("/tmp/cert.json"), "t")
                == ("/", ["private", "tmp", "cert.json"]),
                "darwin /tmp normalization failed",
            )
            _check(
                v._validate_path_and_split(Path("/var/folders/xx/evil_link/cert.json"), "t")
                == ("/", ["private", "var", "folders", "xx", "evil_link", "cert.json"]),
                "nested planted symlink must remain in parts for O_NOFOLLOW",
            )
            _check(
                v._normalize_darwin_system_prefix(Path("/tmp"), allow_exact=True) == Path("/private/tmp"),
                "darwin exact PWD /tmp normalization failed",
            )
            _check(
                v._validate_path_and_split(Path("/tmp"), "t") == ("/", ["tmp"]),
                "exact /tmp leaf must not normalize",
            )
            _check(
                v._validate_path_and_split(Path("/tmp/../etc/passwd"), "t")
                == ("/", ["tmp", "..", "etc", "passwd"]),
                "parent escape after /tmp must not normalize",
            )

        with (
            mock.patch.object(v.sys, "platform", "darwin"),
            mock.patch.object(v.os, "lstat", return_value=user_lnk),
            mock.patch.object(v.os, "readlink", side_effect=_fake_readlink),
        ):
            _check(
                v._validate_path_and_split(Path("/var/folders/xx/cert.json"), "t")
                == ("/", ["var", "folders", "xx", "cert.json"]),
                "non-root /var symlink must not normalize",
            )
        with (
            mock.patch.object(v.sys, "platform", "darwin"),
            mock.patch.object(v.os, "lstat", return_value=root_lnk),
            mock.patch.object(v.os, "readlink", return_value="private/evil"),
        ):
            _check(
                v._validate_path_and_split(Path("/var/folders/xx/cert.json"), "t")
                == ("/", ["var", "folders", "xx", "cert.json"]),
                "unexpected readlink target must not normalize",
            )
        with mock.patch.object(v.sys, "platform", "linux"):
            _check(
                v._validate_path_and_split(Path("/var/folders/xx/cert.json"), "t")
                == ("/", ["var", "folders", "xx", "cert.json"]),
                "non-darwin must not normalize",
            )


if __name__ == "__main__":
    unittest.main()
