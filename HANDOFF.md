# Handoff

2026-11-20. A clean `python3 verify.py` exited 1. The closed forms passed. The certificate stored raw residuals near 1e-16, and a second evaluation produced different digits below 1e-15.

The rejection rows were the literal `rejected: true`, guarded by `assert`. Under `python3 -O`, a planted shear stiffness of 1 still returned `rejected: true`.

Checks now raise through `require`, which `python3 -O` does not remove. The certificate records whether the numerical discrepancy is under 1e-9. The run still rejects a discrepancy of 1e-11 or more. Do not put the raw residual back into the certificate.

The certificate still names the repository Foundation-0/fibral-rigidity.
