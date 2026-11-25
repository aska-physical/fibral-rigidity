#!/usr/bin/env python3
"""Corner checks miss an interior pair of bar lengths.

Two elastic bars join pins at -1 and +1. A stable shape is an equilibrium
whose Hessian is positive definite. Each corner of the natural-length box
[0.7, 1.3] by [0.9, 1.5] has one. The interior pair (0.9, 1.1) does not.

Off the pin line, the two bar forces are not collinear, so they balance
only when both are zero. Both are zero only at a circle intersection.
Those circles do not meet when the lengths sum to the pin span.
On the pin line, the only root of the axial force for (0.9, 1.1) is the
point where both extensions are zero, and the transverse stiffness there is zero.

The bulk-modulus identity of the cross-polytope cell is a different statement.
It holds for every strut length once the area ratio is fixed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

EA = 1.0
PINS = (np.array([-1.0, 0.0]), np.array([1.0, 0.0]))
SPAN = 2.0
OUT = Path(__file__).resolve().parent / "results" / "length_box_certificate.json"


class CheckFailure(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise CheckFailure(message)


def energy_grad_hess(point: np.ndarray, lengths: tuple[float, float]):
    gradient = np.zeros(2)
    hessian = np.zeros((2, 2))
    for pin, length0 in zip(PINS, lengths):
        delta = point - pin
        length = float(np.linalg.norm(delta))
        require(length > 1e-12, "joint landed on a pin")
        extension = length - length0
        direction = delta / length
        scale = EA / length0
        gradient += scale * extension * direction
        hessian += scale * (
            np.outer(direction, direction)
            + (extension / length) * (np.eye(2) - np.outer(direction, direction))
        )
    return gradient, hessian


def newton(lengths: tuple[float, float], start: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    point = np.array(start, dtype=float)
    for _ in range(40):
        gradient, hessian = energy_grad_hess(point, lengths)
        step = np.linalg.solve(hessian, gradient)
        point = point - step
        if float(np.linalg.norm(step)) < 1e-12:
            break
    gradient, hessian = energy_grad_hess(point, lengths)
    require(float(np.linalg.norm(gradient)) < 1e-8, f"no equilibrium from {start} at {lengths}")
    return point, np.linalg.eigvalsh(hessian)


def min_eigenvalue(lengths: tuple[float, float], start: tuple[float, float]) -> float:
    _, eigenvalues = newton(lengths, start)
    return float(eigenvalues[0])


def axial_force(x: float, lengths: tuple[float, float]) -> float:
    total = 0.0
    for pin, length0 in zip((-1.0, 1.0), lengths):
        delta = x - pin
        length = abs(delta)
        if length < 1e-14:
            return math.nan
        extension = length - length0
        total += -(EA / length0) * extension * math.copysign(1.0, delta)
    return total


def on_line_roots(lengths: tuple[float, float]) -> list[float]:
    """Bracket sign changes of the axial force. The force is smooth on each open side of a pin."""
    roots: list[float] = []
    for left, right in ((-3.0, -1.0 - 1e-6), (-1.0 + 1e-6, 1.0 - 1e-6), (1.0 + 1e-6, 3.0)):
        xs = np.linspace(left, right, 4001)
        previous_x = float(xs[0])
        previous_force = axial_force(previous_x, lengths)
        for x in xs[1:]:
            force = axial_force(float(x), lengths)
            if math.isnan(previous_force) or math.isnan(force):
                previous_x, previous_force = float(x), force
                continue
            if previous_force == 0.0 or previous_force * force < 0.0:
                lo, hi = previous_x, float(x)
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    if axial_force(lo, lengths) * axial_force(mid, lengths) <= 0.0:
                        hi = mid
                    else:
                        lo = mid
                root = 0.5 * (lo + hi)
                if not any(abs(root - kept) < 1e-6 for kept in roots):
                    roots.append(root)
            previous_x, previous_force = float(x), force
    return roots


def transverse_stiffness(x: float, lengths: tuple[float, float]) -> float:
    """Curvature in the off-line direction at an on-line joint."""
    total = 0.0
    for pin, length0 in zip((-1.0, 1.0), lengths):
        length = abs(x - pin)
        extension = length - length0
        # N/L with N = EA * extension / length0
        total += (EA / length0) * extension / length
    return total


def interior_has_no_positive_definite_equilibrium() -> dict[str, float]:
    lengths = (0.9, 1.1)
    require(abs(sum(lengths) - SPAN) < 1e-15, "interior pair must sum to the pin span")
    # Circles of radii 0.9 and 1.1 centered 2 apart are tangent.
    require(abs(0.9 + 1.1 - SPAN) < 1e-15, "circles are not tangent")
    roots = on_line_roots(lengths)
    require(len(roots) >= 1, "missed the flat root")
    flat = min(roots, key=lambda value: abs(value + 0.1))
    require(abs(flat + 0.1) < 1e-8, f"flat root moved to {flat}")
    transverse = transverse_stiffness(flat, lengths)
    require(abs(transverse) < 1e-8, f"flat transverse stiffness is {transverse}")
    for root in roots:
        transverse_root = transverse_stiffness(root, lengths)
        require(transverse_root <= 1e-6, f"on-line root at {root} is transversely stiff")
    return {"flat_root": flat, "transverse_stiffness": transverse, "on_line_roots": len(roots)}


def main() -> int:
    corners = [
        ((0.7, 0.9), (-0.125, 0.0)),
        ((0.7, 1.5), (-0.44, 0.42)),
        ((1.3, 0.9), (0.22, 0.45)),
        ((1.3, 1.5), (-0.14, 0.97)),
    ]
    corner_rows = []
    for lengths, start in corners:
        smallest = min_eigenvalue(lengths, start)
        require(smallest > 1e-3, f"corner {lengths} is not positive definite ({smallest})")
        corner_rows.append(
            {
                "lengths": list(lengths),
                "min_eigenvalue_above_1e-3": True,
            }
        )

    interior = interior_has_no_positive_definite_equilibrium()
    # A length pair off the flat line, inside the same box, is positive definite again.
    neighbor = min_eigenvalue((0.9, 1.2), (-0.15, 0.5))
    require(neighbor > 1e-3, "a neighbor of the flat pair lost its stable shape")

    certificate = {
        "repository": "aska-physical/fibral-rigidity",
        "claim_rejected": "every length pair in the box has a positive-definite equilibrium",
        "corners": corner_rows,
        "interior_lengths": [0.9, 1.1],
        "interior_positive_definite_equilibrium": False,
        "interior_on_line_roots": interior["on_line_roots"],
        "neighbor_lengths": [0.9, 1.2],
        "neighbor_positive_definite": True,
    }
    rendered = json.dumps(certificate, indent=2, sort_keys=True) + "\n"
    parser = argparse.ArgumentParser(description="Check the two-bar length box.")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(rendered, encoding="utf-8")
    if not OUT.exists() or json.loads(OUT.read_text(encoding="utf-8")) != certificate:
        print("[FAIL] Certificate mismatch or missing.", flush=True)
        return 1
    print("[PASS] Four corners stable, flat interior pair not positive definite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
