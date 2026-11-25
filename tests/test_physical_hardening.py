"""Physical compressive kink-band & NASA SP-8007 shell-buckling tests for aska-physical/fibral-rigidity.

Covers:
  - Pin-jointed (K=1, r_g = r/2) compressive envelope capped by min(sigma_Euler, sigma_kink, gamma_SP8007*sigma_shell)
  - Honest separation of tensile rule-of-mixtures strength from representative UD compressive kink-band limits
    (e.g. K13D2U 350 MPa compressive vs 2,225.54 MPa ROM tensile; effective tube gain saturates at 1.0x at 5% rho_bar)
  - NASA SP-8007 orthotropic local shell buckling dominance at ultra-thin eta=0.99 (R_mid/t=99.5),
    capping allowable stress below sigma_kink and effective tube gain below 1.0
  - Zero bare asserts and zero raw builtin max/min in verify.py across -I -B, -I -O -B, -I -B -W error.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if importlib.util.find_spec("sympy") is None and "sympy" not in sys.modules:
    sys.modules["sympy"] = types.ModuleType("sympy")

import verify


class TestFibralRigidityPhysicalHardening(unittest.TestCase):
    def test_kink_band_and_sp8007_cap_cfrp_compression_in_benchmarks(self) -> None:
        raw_data = verify.load_strict_json_file(verify.DATA_PATH, "data/carbon_fibers.json")
        raw_by_id = {row["id"]: row for row in raw_data["carbon_fibers"]}
        num = verify.run_numerical_truss_checks()
        cfrp = verify.run_cfrp_benchmark_checks(num, raw_data=raw_data)
        by_id = {row["fiber_id"]: row for row in cfrp["cfrp_truss_benchmarks"]}

        # Every grade reads its hashed compressive allowable from data/carbon_fibers.json and caps strictly below ROM tensile
        self.assertEqual(set(by_id.keys()), {"T300", "T700S", "T1100G", "M55J", "K13D2U"})
        for fid, row in by_id.items():
            expected_kink = raw_by_id[fid]["sigma_kink_comp_Vf60_MPa"]
            env = row["buckling_vs_tensile_envelope"]
            self.assertEqual(row["sigma_kink_comp_CFRP_Vf60_MPa"], expected_kink)
            self.assertLess(row["sigma_kink_comp_CFRP_Vf60_MPa"], row["sigma_tensile_CFRP_Vf60_MPa"])
            self.assertEqual(env["K_eff_pin_jointed"], 1.0)
            self.assertEqual(env["at_5pct_governing_compressive_mode"], "kink_band_compression")
            self.assertEqual(env["at_5pct_sigma_allow_tube_MPa"], expected_kink)
            self.assertLessEqual(env["at_5pct_effective_tube_gain"], 1.0)
            self.assertAlmostEqual(env["hollow_tube_eta08_R_over_t_mid"], 4.5, places=4)
            self.assertFalse(env["hollow_tube_eta08_sp8007_thin_shell_valid"])
            self.assertLess(
                env["hollow_tube_eta08_critical_rho_bar_compressive"],
                env["hollow_tube_eta08_critical_rho_bar"],
            )

        k13 = by_id["K13D2U"]
        env_k13 = k13["buckling_vs_tensile_envelope"]
        self.assertAlmostEqual(k13["sigma_tensile_CFRP_Vf60_MPa"], 2225.54, places=2)
        self.assertLess(k13["sigma_kink_comp_CFRP_Vf60_MPa"], 0.20 * k13["sigma_tensile_CFRP_Vf60_MPa"])
        self.assertGreater(env_k13["at_5pct_sigma_euler_tube_MPa"], 10000.0)
        self.assertAlmostEqual(env_k13["hollow_tube_eta08_critical_rho_bar_compressive"], 0.000253, places=6)

        t11 = by_id["T1100G"]
        self.assertAlmostEqual(t11["sigma_tensile_CFRP_Vf60_MPa"], 4230.25, places=2)
        self.assertLess(t11["sigma_kink_comp_CFRP_Vf60_MPa"], 0.40 * t11["sigma_tensile_CFRP_Vf60_MPa"])

    def test_control8_and_benchmark_reject_tensile_mirrored_caps_and_missing_field(self) -> None:
        raw_data = verify.load_strict_json_file(verify.DATA_PATH, "data/carbon_fibers.json")
        num = verify.run_numerical_truss_checks()

        # Mutant M1/M2: replacing K13D2U compressive cap with ROM tensile 2225.54 MPa fails BOTH benchmark and Control #8
        m_k13 = copy.deepcopy(raw_data)
        for fib in m_k13["carbon_fibers"]:
            if fib["id"] == "K13D2U":
                fib["sigma_kink_comp_Vf60_MPa"] = 2225.54
        with self.assertRaises(verify.CheckFailure):
            verify.run_cfrp_benchmark_checks(num, raw_data=m_k13)
        with self.assertRaises(verify.CheckFailure):
            verify.run_must_fail_controls(num, raw_data=m_k13)

        # Mutant M6: replacing T300 compressive cap with ROM tensile 2139.5 MPa fails BOTH benchmark and Control #8
        m_t300 = copy.deepcopy(raw_data)
        for fib in m_t300["carbon_fibers"]:
            if fib["id"] == "T300":
                fib["sigma_kink_comp_Vf60_MPa"] = 2139.5
        with self.assertRaises(verify.CheckFailure):
            verify.run_cfrp_benchmark_checks(num, raw_data=m_t300)
        with self.assertRaises(verify.CheckFailure):
            verify.run_must_fail_controls(num, raw_data=m_t300)

        # Missing sigma_kink_comp_Vf60_MPa fails closed (no arbitrary fallback constant)
        m_missing = copy.deepcopy(raw_data)
        del m_missing["carbon_fibers"][0]["sigma_kink_comp_Vf60_MPa"]
        with self.assertRaises(verify.CheckFailure):
            verify.run_cfrp_benchmark_checks(num, raw_data=m_missing)
        with self.assertRaises(verify.CheckFailure):
            verify.run_must_fail_controls(num, raw_data=m_missing)

    def test_ultra_thin_eta099_shell_buckling_and_slender_euler_regime(self) -> None:
        L_bar = 2.0 * (math.sqrt(2.0) - 1.0)
        thin = verify.evaluate_tubular_compressive_limit(562.4, 350.0, 0.99, L_bar, 0.05)
        self.assertTrue(thin["sp8007_thin_shell_valid_R_over_t_ge_10"])
        self.assertAlmostEqual(thin["R_over_t_mid"], 99.5, places=4)
        self.assertEqual(thin["governing_mode_5pct"], "sp8007_local_shell_buckling")
        self.assertLess(thin["sigma_allow_tube_5pct_MPa"], 350.0)
        self.assertLess(thin["effective_tube_gain_5pct"], 1.0)

        # Very low relative density rho_bar=0.0005 puts T1100G genuinely in Euler regime where hollow gain > 1
        slender = verify.evaluate_tubular_compressive_limit(195.8, 1500.0, 0.80, L_bar, 0.0005)
        self.assertEqual(slender["governing_mode_5pct"], "euler_column_buckling")
        self.assertAlmostEqual(slender["effective_tube_gain_5pct"], (1.0 + 0.8**2) / (1.0 - 0.8**2), places=3)

    def test_invalid_inputs_and_ast_hygiene(self) -> None:
        L_bar = 2.0 * (math.sqrt(2.0) - 1.0)
        for bad in (0.0, -1.0, float("nan"), float("inf"), True):
            with self.assertRaises(verify.CheckFailure):
                verify.evaluate_tubular_compressive_limit(bad, 350.0, 0.80, L_bar, 0.05)
            with self.assertRaises(verify.CheckFailure):
                verify.evaluate_tubular_compressive_limit(562.4, bad, 0.80, L_bar, 0.05)
        for bad_eta in (-0.1, 1.0, 1.5, float("nan"), True):
            with self.assertRaises(verify.CheckFailure):
                verify.evaluate_tubular_compressive_limit(562.4, 350.0, bad_eta, L_bar, 0.05)

        for rel in ("verify.py", "tests/test_physical_hardening.py"):
            tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
            self.assertEqual([n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assert)], [])
        v_tree = ast.parse((ROOT / "verify.py").read_text(encoding="utf-8"))
        maxmin = [
            n.lineno
            for n in ast.walk(v_tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("max", "min")
        ]
        self.assertEqual(maxmin, [])


if __name__ == "__main__":
    unittest.main()
