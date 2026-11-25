#!/usr/bin/env python3
"""Deterministic verification suite for Foundation-0/fibral-rigidity (SymPy + NumPy)."""

from __future__ import annotations

import argparse
import errno
import hashlib
import hmac
import json
import math
import os
import secrets
import stat
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "carbon_fibers.json"
CERT_PATH = ROOT / "results" / "certificate.json"
MAX_JSON_BYTES = 1_048_576
MAX_ALLOWED_CAP_BYTES = 64 * 1024 * 1024
_READ_CHUNK_BYTES = 65_536
MAX_JSON_DEPTH = 64
MAX_INT_DIGITS = 1000
MAX_INT_BITS = 3322
_ASCII_DIGITS = frozenset("0123456789")
_EXACT_PATH_TYPES = (Path, type(Path()))
_DIR_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_TMP_LOCK = threading.Lock()
_TMP_COUNTER = 0


def _next_tmp_name(leaf: str) -> str:
    global _TMP_COUNTER
    with _TMP_LOCK:
        _TMP_COUNTER += 1
        seq = _TMP_COUNTER
    return f".{leaf}.frtmp.{os.getpid()}.{threading.get_ident()}.{seq}.{secrets.token_hex(8)}"


class CheckFailure(RuntimeError):
    """A failed check. Raised as a normal exception so python -O cannot strip it."""


def require(ok: bool, message: str) -> None:
    if not ok:
        raise CheckFailure(message)


def strict_finite(
    value: Any,
    name: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    """Validate that value is a finite non-bool real number (rejecting NaN, Inf, bool, overflow)."""
    require(
        type(value) is not bool and isinstance(value, (int, float, np.integer, np.floating)),
        f"{name} must be a finite number, got {type(value).__name__}",
    )
    try:
        val = float(value)
    except OverflowError as exc:
        raise CheckFailure(f"{name} overflows float: {exc}") from exc
    require(math.isfinite(val), f"{name} must be finite, got {value!r}")
    if positive:
        require(val > 0.0, f"{name} must be positive, got {val!r}")
    if non_negative:
        require(val >= 0.0, f"{name} must be non-negative, got {val!r}")
    return val


def strict_finite_max(*values: Any, name: str) -> float:
    """NaN-safe maximum: rejects any non-finite argument instead of silently dropping NaN."""
    require(len(values) >= 1, f"{name} requires at least one value")
    best = strict_finite(values[0], name)
    for item in values[1:]:
        cur = strict_finite(item, name)
        if cur > best:
            best = cur
    return best


def strict_finite_min(*values: Any, name: str) -> float:
    """NaN-safe minimum: rejects any non-finite argument instead of silently dropping NaN."""
    require(len(values) >= 1, f"{name} requires at least one value")
    best = strict_finite(values[0], name)
    for item in values[1:]:
        cur = strict_finite(item, name)
        if cur < best:
            best = cur
    return best


def _require_utf8_no_surrogate(text: str, label: str) -> None:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CheckFailure(f"{label}: lone UTF-16 surrogate forbidden") from exc


def sanitize(value: Any, _depth: int = 0) -> Any:
    """Drop the sign on a zero and reject non-finite floats, oversized ints, or deep/cyclic structures."""
    require(_depth <= MAX_JSON_DEPTH, f"sanitize: nesting depth exceeds {MAX_JSON_DEPTH}")
    t = type(value)
    if t is dict:
        out: dict[str, Any] = {}
        for key, item in value.items():
            require(type(key) is str, "sanitize: dict keys must be exact str")
            _require_utf8_no_surrogate(key, "sanitize")
            out[key] = sanitize(item, _depth + 1)
        return out
    if t is list:
        return [sanitize(item, _depth + 1) for item in value]
    if t is bool or value is None:
        return value
    if t is int:
        require(value.bit_length() <= MAX_INT_BITS, f"sanitize: integer exceeds {MAX_INT_BITS}-bit cap")
        return value
    if t is str:
        _require_utf8_no_surrogate(value, "sanitize")
        return value
    if isinstance(value, (float, np.floating)):
        val = strict_finite(value, "certificate_float")
        return 0.0 if val == 0.0 else val
    if isinstance(value, np.integer):
        return int(value)
    raise CheckFailure(f"sanitize: unsupported certificate value type {t.__name__}")


def _require_concrete_path(path: Any, label: str) -> Path:
    require(type(path) in _EXACT_PATH_TYPES, f"{label}: path must be concrete pathlib.Path")
    raw_s = str(path)
    require("\x00" not in raw_s, f"{label}: NUL byte forbidden in path")
    _require_utf8_no_surrogate(raw_s, label)
    return path


def _open_dir_chain(anchor: str, parts: list[str], label: str, *, create_missing: bool = False) -> int:
    """Walk directory components atomically via openat(..., O_DIRECTORY | O_NOFOLLOW)."""
    try:
        dfd = os.open(anchor, _DIR_OPEN_FLAGS)
    except OSError as exc:
        raise CheckFailure(f"{label}: cannot open anchor directory ({exc})") from exc

    for part in parts:
        try:
            nxt = os.open(part, _DIR_OPEN_FLAGS, dir_fd=dfd)
        except OSError as exc:
            if create_missing and exc.errno == errno.ENOENT:
                try:
                    os.mkdir(part, 0o700, dir_fd=dfd)
                except FileExistsError:
                    pass
                except OSError as mk_exc:
                    os.close(dfd)
                    raise CheckFailure(f"{label}: cannot create directory ({mk_exc})") from mk_exc
                try:
                    nxt = os.open(part, _DIR_OPEN_FLAGS, dir_fd=dfd)
                except OSError as re_exc:
                    os.close(dfd)
                    raise CheckFailure(f"{label}: cannot open created directory ({re_exc})") from re_exc
            else:
                os.close(dfd)
                if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise CheckFailure(f"{label}: symlink path or ancestor forbidden") from exc
                if exc.errno == errno.ENOENT:
                    raise CheckFailure(f"{label}: missing or unreadable ({exc})") from exc
                raise CheckFailure(f"{label}: cannot traverse directory ({exc})") from exc
        os.close(dfd)
        dfd = nxt
    return dfd


_DARWIN_SYSTEM_ROOT_ALIASES: tuple[tuple[str, str], ...] = (
    ("/var", "/private/var"),
    ("/tmp", "/private/tmp"),
    ("/etc", "/private/etc"),
)


def _normalize_darwin_system_prefix(path: Path, *, allow_exact: bool = False) -> Path:
    """Map verified root-owned macOS /var, /tmp, /etc aliases to /private/* without resolving user symlinks."""
    if sys.platform != "darwin" or not path.is_absolute():
        return path
    parts = path.parts
    min_parts = 2 if allow_exact else 3
    if len(parts) < min_parts or parts[0] != "/" or ".." in parts:
        return path
    top = "/" + parts[1]
    for alias, canonical in _DARWIN_SYSTEM_ROOT_ALIASES:
        if top != alias:
            continue
        try:
            st = os.lstat(alias)
            if not stat.S_ISLNK(st.st_mode) or st.st_uid != 0:
                return path
            target = os.readlink(alias)
            if target not in (canonical, canonical.lstrip("/")):
                return path
        except OSError:
            return path
        return Path(canonical, *parts[2:])
    return path


def _check_pwd_no_symlinks(label: str) -> None:
    pwd = os.environ.get("PWD")
    if pwd and os.path.isabs(pwd):
        _require_utf8_no_surrogate(pwd, f"{label} PWD")
        pwd_path = _normalize_darwin_system_prefix(Path(pwd), allow_exact=True)
        pwd_parts = [p for p in pwd_path.parts[1:] if p not in ("", ".")]
        try:
            pwd_fd = _open_dir_chain(pwd_path.anchor, pwd_parts, f"{label} PWD")
        except CheckFailure as exc:
            if "missing or unreadable" not in str(exc):
                raise
        else:
            os.close(pwd_fd)


def _validate_path_and_split(path: Any, label: str) -> tuple[str, list[str]]:
    target = _require_concrete_path(path, label)
    if not target.is_absolute():
        try:
            Path.cwd()
        except OSError as exc:
            raise CheckFailure(f"{label}: CWD unavailable ({exc})") from exc
        _check_pwd_no_symlinks(label)
        rel_parts = [p for p in target.parts if p not in ("", ".")]
        return ".", rel_parts
    norm = _normalize_darwin_system_prefix(target)
    abs_parts = [p for p in norm.parts[1:] if p not in ("", ".")]
    return norm.anchor, abs_parts


def read_regular_bytes(path: Path, label: str, max_bytes: int = MAX_JSON_BYTES) -> bytes:
    """Read raw bytes from a non-symlink regular file via atomic dir_fd under a strict streaming byte cap."""
    require(
        type(max_bytes) is int and 1 <= max_bytes <= MAX_ALLOWED_CAP_BYTES,
        f"{label}: max_bytes must be int in [1, {MAX_ALLOWED_CAP_BYTES}]",
    )
    anchor, parts = _validate_path_and_split(path, label)
    require(len(parts) >= 1, f"{label}: not a regular file")
    dfd = _open_dir_chain(anchor, parts[:-1], label)
    try:
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        try:
            fd = os.open(parts[-1], flags, dir_fd=dfd)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise CheckFailure(f"{label}: symlink path or ancestor forbidden") from exc
            raise CheckFailure(f"{label}: missing or unreadable ({exc})") from exc
    finally:
        os.close(dfd)

    try:
        st = os.fstat(fd)
        require(stat.S_ISREG(st.st_mode), f"{label}: not a regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            rem = max_bytes + 1 - total
            to_read = _READ_CHUNK_BYTES if _READ_CHUNK_BYTES < rem else rem
            chunk = os.read(fd, to_read)
            if not chunk:
                break
            total += len(chunk)
            require(total <= max_bytes, f"{label}: exceeds size cap ({max_bytes} bytes)")
            chunks.append(chunk)
        return b"".join(chunks)
    except CheckFailure:
        raise
    except OSError as exc:
        raise CheckFailure(f"{label}: read failed ({exc})") from exc
    finally:
        os.close(fd)


def _reject_dup_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in pairs:
        require(key not in out, f"duplicate JSON key: {key!r}")
        out[key] = val
    return out


def _reject_nonfinite_constant(token: str) -> Any:
    raise CheckFailure(f"non-finite JSON constant forbidden: {token!r}")


def _strict_parse_float(token: str) -> float:
    return strict_finite(float(token), "json_float")


def _strict_parse_int(token: str) -> int:
    """Parse strict ASCII JSON integer under explicit digit cap (closing Python's implicit 4,300-digit boundary)."""
    require(type(token) is str, "json_int must be exact str")
    neg = token.startswith("-")
    digits = token[1:] if neg else token
    require(len(digits) >= 1, "json_int has no digits")
    require(len(digits) <= MAX_INT_DIGITS, f"json_int exceeds {MAX_INT_DIGITS}-digit limit")
    for ch in digits:
        require(ch in _ASCII_DIGITS, "json_int contains non-ASCII digit")
    acc = 0
    for ch in digits:
        acc = acc * 10 + (ord(ch) - 48)
    return -acc if neg else acc


def _validate_json_tree(root: Any, label: str) -> None:
    stack: list[tuple[Any, int]] = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        require(depth <= MAX_JSON_DEPTH, f"{label}: JSON nesting depth exceeds {MAX_JSON_DEPTH}")
        t = type(node)
        if t is bool or node is None:
            continue
        if t is int:
            require(node.bit_length() <= MAX_INT_BITS, f"{label}: integer exceeds {MAX_INT_BITS}-bit cap")
            continue
        if t is float:
            strict_finite(node, f"{label}_float")
            continue
        if t is str:
            _require_utf8_no_surrogate(node, label)
            continue
        if t is dict:
            for k, v in node.items():
                require(type(k) is str, f"{label}: JSON dict keys must be exact str")
                stack.append((k, depth + 1))
                stack.append((v, depth + 1))
            continue
        if t is list:
            for item in node:
                stack.append((item, depth + 1))
            continue
        raise CheckFailure(f"{label}: unsupported JSON value type {t.__name__}")


def load_strict_json_file(path: Path, label: str, max_bytes: int = MAX_JSON_BYTES) -> dict[str, Any]:
    """Fail-closed strict JSON loader closing duplicate keys, NaN/Inf, huge ints, surrogates, and raw I/O tracebacks."""
    raw = read_regular_bytes(path, label, max_bytes=max_bytes)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CheckFailure(f"{label}: invalid UTF-8 ({exc})") from exc
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_reject_dup_keys,
            parse_constant=_reject_nonfinite_constant,
            parse_float=_strict_parse_float,
            parse_int=_strict_parse_int,
        )
    except CheckFailure:
        raise
    except (ValueError, RecursionError) as exc:
        raise CheckFailure(f"{label}: invalid JSON ({exc})") from exc
    require(type(parsed) is dict, f"{label}: JSON root must be an object")
    _validate_json_tree(parsed, label)
    return parsed


def canonical_data_sha256(data: dict[str, Any]) -> str:
    """Compute canonical compact sorted-key SHA-256 of the verified fiber dataset."""
    require(type(data) is dict, "canonical_data_sha256: data must be dict")
    _validate_json_tree(data, "canonical_data")
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def render_canonical_cert_bytes(cert: dict[str, Any]) -> bytes:
    """Render sanitized LF-terminated canonical JSON bytes, forbidding NaN/Infinity/huge ints."""
    require(type(cert) is dict, "render_canonical_cert_bytes: cert must be dict")
    clean = sanitize(cert)
    _validate_json_tree(clean, "certificate")
    return (json.dumps(clean, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def verify_canonical_cert_bytes(cert_path: Path, expected_bytes: bytes) -> None:
    """Verify results/certificate.json matches rendered canonical bytes via constant-time compare."""
    require(type(expected_bytes) is bytes, "expected_bytes must be exact bytes")
    raw = read_regular_bytes(cert_path, "results/certificate.json")
    require(
        hmac.compare_digest(raw, expected_bytes),
        "results/certificate.json canonical byte mismatch",
    )


def write_canonical_cert_bytes(cert_path: Path, payload: bytes) -> None:
    """Atomically write canonical certificate bytes via dir_fd 0o600 CSPRNG temp, fsync, st_nlink==1 & os.replace."""
    require(type(payload) is bytes, "results/certificate.json: payload must be exact bytes")
    anchor, parts = _validate_path_and_split(cert_path, "results/certificate.json")
    require(len(parts) >= 1, "refusing to overwrite non-regular certificate path")
    leaf = parts[-1]
    dfd = _open_dir_chain(anchor, parts[:-1], "results/certificate.json", create_missing=True)
    try:
        try:
            st = os.lstat(leaf, dir_fd=dfd)
        except FileNotFoundError:
            st = None
        except OSError as exc:
            raise CheckFailure(f"failed to stat results/certificate.json ({exc})") from exc
        if st is not None:
            require(not stat.S_ISLNK(st.st_mode), "results/certificate.json: symlink path or ancestor forbidden")
            require(stat.S_ISREG(st.st_mode), "refusing to overwrite non-regular certificate path")
            require(st.st_nlink <= 1, "refusing to write through hardlinked certificate path")

        tmp_fd: int | None = None
        tmp_name = ""
        for _ in range(32):
            cand = _next_tmp_name(leaf)
            try:
                tmp_fd = os.open(
                    cand,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_NONBLOCK,
                    0o600,
                    dir_fd=dfd,
                )
                tmp_name = cand
                break
            except FileExistsError:
                continue
            except OSError as exc:
                raise CheckFailure(f"failed to create temp certificate file ({exc})") from exc
        require(tmp_fd is not None, "temp certificate file collision limit exceeded")

        try:
            try:
                view = memoryview(payload)
                written = 0
                while written < len(payload):
                    written += os.write(tmp_fd, view[written:])  # type: ignore[arg-type]
                os.fsync(tmp_fd)  # type: ignore[arg-type]
                require(os.fstat(tmp_fd).st_nlink == 1, "refusing to replace hardlinked temp file")  # type: ignore[arg-type]
            finally:
                os.close(tmp_fd)  # type: ignore[arg-type]
            os.replace(tmp_name, leaf, src_dir_fd=dfd, dst_dir_fd=dfd)
            os.fsync(dfd)
        except CheckFailure:
            try:
                os.unlink(tmp_name, dir_fd=dfd)
            except OSError:
                pass
            raise
        except OSError as exc:
            try:
                os.unlink(tmp_name, dir_fd=dfd)
            except OSError:
                pass
            raise CheckFailure(f"failed to write results/certificate.json ({exc})") from exc
    finally:
        os.close(dfd)


NUMERICAL_BOUND = 1e-9


def require_finite_stiffness_matrix(stiffness: Any, n: int, name: str) -> np.ndarray:
    """Validate that stiffness is a finite 2D NumPy matrix of exact Kelvin/Voigt shape (n_strain, n_strain)."""
    require(type(n) is int and n >= 2, f"{name}: n must be int >= 2")
    n_strain = n + (n * (n - 1)) // 2
    require(isinstance(stiffness, np.ndarray), f"{name} must be a numpy.ndarray")
    require(stiffness.shape == (n_strain, n_strain), f"{name} must have shape ({n_strain}, {n_strain})")
    require(bool(np.all(np.isfinite(stiffness))), f"{name} contains non-finite values")
    return stiffness


def _cubic_stiffness_template(n: int, c11: float, c12: float, c44: float = 0.0) -> np.ndarray:
    n_strain = n + (n * (n - 1)) // 2
    mat = np.zeros((n_strain, n_strain), dtype=float)
    for i in range(n):
        for j in range(n):
            mat[i, j] = c11 if i == j else c12
    for s_idx in range(n, n_strain):
        mat[s_idx, s_idx] = c44
    return mat


def build_fn_bars(
    n: int, t: float, E1: float, A_intra: float, A_cross: float, A_diag: float = 0.0
) -> tuple[np.ndarray, list[tuple[int, int, np.ndarray, float]]]:
    """Build nodal coordinates and periodic bars on Q = [-1, 1]^n (volume 2^n)."""
    require(type(n) is int and n >= 2, f"n must be int >= 2, got {n!r}")
    t = strict_finite(t, "t", positive=True)
    require(t < 1.0, f"t must be < 1.0, got {t!r}")
    E1 = strict_finite(E1, "E1", positive=True)
    A_intra = strict_finite(A_intra, "A_intra", positive=True)
    A_cross = strict_finite(A_cross, "A_cross", positive=True)
    A_diag = strict_finite(A_diag, "A_diag", non_negative=True)

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
    require(type(n) is int and n >= 2, f"n must be int >= 2, got {n!r}")
    cell_edge = strict_finite(cell_edge, "cell_edge", positive=True)
    require(isinstance(coords, np.ndarray) and coords.ndim == 2 and coords.shape[1] == n and coords.shape[0] >= 2, "coords must be a 2D array with shape (>= 2, n)")
    require(bool(np.all(np.isfinite(coords))), "coords must contain only finite entries")
    require(type(bars) in (list, tuple) and len(bars) >= 1, "bars must be a non-empty list/tuple")
    n_nodes = coords.shape[0]
    ndof = n_nodes * n
    shear_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    n_strain = n + len(shear_pairs)
    cell_vol = cell_edge ** n

    B_u = np.zeros((len(bars), ndof), dtype=float)
    B_e = np.zeros((len(bars), n_strain), dtype=float)
    k_vec = np.zeros(len(bars), dtype=float)
    ea_vol = 0.0

    for b_idx, bar in enumerate(bars):
        require(type(bar) in (list, tuple) and len(bar) == 4, "each bar must be a 4-tuple (u, v, shift, EA)")
        u, v, shift, EA = bar
        require(type(u) is int and 0 <= u < n_nodes, "bar start node out of range")
        require(type(v) is int and 0 <= v < n_nodes, "bar end node out of range")
        EA_val = strict_finite(EA, "bar_EA", positive=True)
        require(isinstance(shift, np.ndarray) and shift.shape == (n,), "bar shift must be a 1D array of length n")
        require(bool(np.all(np.isfinite(shift))), "bar shift must contain only finite entries")
        vec = (coords[v] + shift) - coords[u]
        length = strict_finite(float(np.linalg.norm(vec)), "bar_length", positive=True)
        n_hat = vec / length
        k_vec[b_idx] = EA_val / length
        ea_vol += EA_val * length
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
    C_out = 0.5 * (C_cell + C_cell.T) / cell_vol
    require(bool(np.all(np.isfinite(C_out))), "homogenized stiffness matrix contains non-finite values")
    ea_density = strict_finite(ea_vol / cell_vol, "ea_vol_density", non_negative=True)
    return C_out, ea_density


def hashin_shtrikman_upper_bounds(Es: float, nu_s: float, phi: float) -> tuple[float, float, float]:
    """Exact 3D Hashin-Shtrikman isotropic solid-void upper bounds (K_HS, G_HS, E_HS)."""
    Es = strict_finite(Es, "Es", positive=True)
    nu_s = strict_finite(nu_s, "nu_s")
    require(-1.0 < nu_s < 0.5, f"nu_s must lie in (-1, 0.5), got {nu_s!r}")
    phi = strict_finite(phi, "phi", positive=True)
    require(phi <= 1.0, f"phi must be <= 1.0, got {phi!r}")
    Ks = Es / (3.0 * (1.0 - 2.0 * nu_s))
    Gs = Es / (2.0 * (1.0 + nu_s))
    K_hs = Ks * (4.0 * Gs * phi) / (4.0 * Gs + 3.0 * Ks * (1.0 - phi))
    G_hs = Gs * phi * (9.0 * Ks + 8.0 * Gs) / (9.0 * Ks + 8.0 * Gs + 6.0 * (Ks + 2.0 * Gs) * (1.0 - phi))
    E_hs = 9.0 * K_hs * G_hs / (3.0 * K_hs + G_hs)
    return (
        strict_finite(K_hs, "K_hs", positive=True),
        strict_finite(G_hs, "G_hs", positive=True),
        strict_finite(E_hs, "E_hs", positive=True),
    )


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
                C_mat = require_finite_stiffness_matrix(C_mat, n, "C_mat")
                rho_bar = strict_finite(ea_vol / E1, "rho_bar", positive=True)
                k1 = E1 * A1 / (t * math.sqrt(2.0))
                k2 = E1 * (alpha * A1) / (2.0 * (1.0 - t))
                lam = k2 / k1
                denom = (n - 2 + 2.0 * lam) * (n - 1 + lam)
                scale = (2.0 ** (2 - n)) * k1
                c11_th = scale * lam * (n - 1) * (n - 2 + lam) / denom
                c12_th = scale * (lam**2) / denom
                C_th = _cubic_stiffness_template(n, c11_th, c12_th, 0.0)
                err = strict_finite_max(*np.abs(C_mat - C_th).ravel(), name="homogenization_err")
                max_err = strict_finite_max(max_err, err, name="max_homogenization_err")
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
        C_mat = require_finite_stiffness_matrix(C_mat, 3, "braced_C_mat")
        C11, C12, C44 = float(C_mat[0, 0]), float(C_mat[0, 1]), float(C_mat[3, 3])
        C_cubic = _cubic_stiffness_template(3, C11, C12, C44)
        cubic_err = strict_finite_max(*np.abs(C_mat - C_cubic).ravel(), name="braced_cubic_err")
        require(C44 > 0.0, "braced C44 must be positive")
        require(cubic_err <= 1e-12, "braced stiffness matrix broke cubic symmetry")
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
        C_m = require_finite_stiffness_matrix(C_m, 3, "bisection_C_m")
        if 2.0 * C_m[3, 3] - (C_m[0, 0] - C_m[0, 1]) < 0.0:
            lo = mid
        else:
            hi = mid
    beta_iso = 0.5 * (lo + hi)
    coords, bars = build_fn_bars(3, c_inc, 1.0, 1.0, alpha_opt_3d, A_diag=beta_iso)
    C_iso, rho_iso = homogenize_periodic_truss(3, coords, bars)
    C_iso = require_finite_stiffness_matrix(C_iso, 3, "C_iso")
    C11_i, C12_i, C44_i = float(C_iso[0, 0]), float(C_iso[0, 1]), float(C_iso[3, 3])
    C_iso_cubic = _cubic_stiffness_template(3, C11_i, C12_i, C44_i)
    iso_sym_err = strict_finite_max(*np.abs(C_iso - C_iso_cubic).ravel(), name="iso_cubic_err")
    require(iso_sym_err <= 1e-12, "isotropic stiffness matrix broke cubic symmetry")
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


def evaluate_tubular_compressive_limit(
    E1_GPa: float,
    sig_kink_MPa: float,
    eta_tube: float,
    L_bar: float,
    rho_bar: float = 0.05,
    *,
    E2_GPa: float = 8.0,
    nu12: float = 0.30,
    K_eff: float = 1.0,
) -> dict[str, Any]:
    """Compute pin-jointed (K=1) CFRP compressive strut limit across Euler, kink-band, and NASA SP-8007 shell buckling.

    For a solid circular rod of outer radius r, radius of gyration is r_g = r / 2, so
      sigma_Euler = pi^2 * E1 * (r_g / (K*L))^2 = pi^2 * E1 * r^2 / (4 * K^2 * L^2),
    giving transition radius r/L = (2*K/pi) * sqrt(sigma_cap / E1) at K = 1 (pin-jointed truss strut).
    Hollow tubes increase Euler buckling by (1 + eta^2)/(1 - eta^2) ONLY while axial stress stays
    below min(sigma_kink, gamma_SP8007 * sigma_shell_ortho).

    Thin-shell validity note: classical axisymmetric thin-shell buckling and the NASA SP-8007
    knockdown factor assume R_mid / t >= 10 (eta >= 19/21 ~= 0.905). For thicker tubes (e.g.
    eta = 0.80, R_mid / t = 4.5), `sp8007_thin_shell_valid_R_over_t_ge_10` is False and
    `sigma_shell_sp8007_MPa` is reported solely as an upper reference scale (kink-band governs by >20x).
    """
    E1 = strict_finite(E1_GPa, "E1_GPa", positive=True)
    sig_kink = strict_finite(sig_kink_MPa, "sig_kink_MPa", positive=True)
    eta = strict_finite(eta_tube, "eta_tube", non_negative=True)
    require(eta < 1.0, f"eta_tube must be < 1.0, got {eta!r}")
    Lb = strict_finite(L_bar, "L_bar", positive=True)
    rho = strict_finite(rho_bar, "rho_bar", positive=True)
    E2 = strict_finite(E2_GPa, "E2_GPa", positive=True)
    nu_12 = strict_finite(nu12, "nu12", positive=True)
    require(nu_12 < 0.5, f"nu12 must be < 0.5, got {nu_12!r}")
    K = strict_finite(K_eff, "K_eff", positive=True)

    tube_gain = (1.0 + eta**2) / (1.0 - eta**2)
    if eta == 0.0:
        R_over_t: float | None = None
        thin_shell_valid = False
        gamma_sp = 1.0
        sig_shell_kd_MPa = 1.0e12
        sig_comp_cap_MPa = sig_kink
    else:
        R_over_t = (1.0 + eta) / (2.0 * (1.0 - eta))
        thin_shell_valid = bool(R_over_t >= 10.0)
        gamma_sp = 1.0 - 0.901 * (1.0 - math.exp(-math.sqrt(R_over_t) / 16.0))
        nu_21 = nu_12 * (E2 / E1)
        require(nu_12 * nu_21 < 1.0, "orthotropic Poisson product nu12*nu21 must be < 1")
        sig_shell_ortho_MPa = math.sqrt(E1 * E2) * (1.0 / R_over_t) / math.sqrt(3.0 * (1.0 - nu_12 * nu_21)) * 1000.0
        sig_shell_kd_MPa = gamma_sp * sig_shell_ortho_MPa
        sig_comp_cap_MPa = strict_finite_min(sig_kink, sig_shell_kd_MPa, name="sig_comp_cap_MPa")

    bar_geom_prefactor = (12.0 + 6.0 * math.sqrt(2.0)) * math.pi * (Lb**3) / 8.0
    r_over_L_solid_comp = (2.0 * K / math.pi) * math.sqrt((sig_kink / 1000.0) / E1)
    rho_crit_solid_comp = bar_geom_prefactor * (r_over_L_solid_comp**2)
    r_over_L_tube_equiv = (2.0 * K / math.pi) * math.sqrt((sig_comp_cap_MPa / 1000.0) / E1)
    rho_crit_tube_comp = bar_geom_prefactor * (r_over_L_tube_equiv**2) / tube_gain

    # Actual compressive allowable stress at benchmark relative density rho_bar (unit cell volume = 8)
    A_dimless = (8.0 * rho) / ((12.0 + 6.0 * math.sqrt(2.0)) * Lb)
    I_solid = (A_dimless**2) / (4.0 * math.pi)
    I_tube = I_solid * tube_gain
    sig_euler_solid_MPa = (math.pi**2 * (E1 * 1000.0) * I_solid) / (A_dimless * ((K * Lb) ** 2))
    sig_euler_tube_MPa = (math.pi**2 * (E1 * 1000.0) * I_tube) / (A_dimless * ((K * Lb) ** 2))
    sig_allow_solid_MPa = strict_finite_min(sig_euler_solid_MPa, sig_kink, name="sig_allow_solid_MPa")
    sig_allow_tube_MPa = strict_finite_min(sig_euler_tube_MPa, sig_kink, sig_shell_kd_MPa, name="sig_allow_tube_MPa")
    if sig_allow_tube_MPa == sig_kink:
        mode = "kink_band_compression"
    elif sig_allow_tube_MPa == sig_shell_kd_MPa:
        mode = "sp8007_local_shell_buckling"
    else:
        mode = "euler_column_buckling"

    return {
        "K_eff_pin_jointed": K,
        "tube_euler_gain": tube_gain,
        "R_over_t_mid": R_over_t,
        "sp8007_thin_shell_valid_R_over_t_ge_10": thin_shell_valid,
        "sp8007_knockdown_gamma": gamma_sp,
        "sigma_kink_MPa": sig_kink,
        "sigma_shell_sp8007_MPa": sig_shell_kd_MPa,
        "sigma_comp_cap_MPa": sig_comp_cap_MPa,
        "r_over_L_solid_comp": r_over_L_solid_comp,
        "rho_crit_solid_comp": rho_crit_solid_comp,
        "rho_crit_tube_comp": rho_crit_tube_comp,
        "sigma_euler_solid_5pct_MPa": sig_euler_solid_MPa,
        "sigma_euler_tube_5pct_MPa": sig_euler_tube_MPa,
        "sigma_allow_solid_5pct_MPa": sig_allow_solid_MPa,
        "sigma_allow_tube_5pct_MPa": sig_allow_tube_MPa,
        "effective_tube_gain_5pct": sig_allow_tube_MPa / sig_allow_solid_MPa,
        "governing_mode_5pct": mode,
    }


def run_cfrp_benchmark_checks(num_res: dict[str, Any], raw_data: dict[str, Any] | None = None) -> dict[str, Any]:
    require(type(num_res) is dict and type(num_res.get("isotropic_braced_point")) is dict, "num_res missing isotropic_braced_point")
    raw = load_strict_json_file(DATA_PATH, "data/carbon_fibers.json") if raw_data is None else raw_data
    require(type(raw) is dict and "matrix_reference" in raw, "data/carbon_fibers.json missing matrix_reference")
    mat = raw["matrix_reference"]
    require(type(mat) is dict, "matrix_reference must be a JSON object")
    for req_k in ("E_GPa", "rho_g_cm3", "fiber_volume_fraction_Vf"):
        require(req_k in mat, f"matrix_reference missing {req_k}")
    Em = strict_finite(mat["E_GPa"], "matrix_reference.E_GPa", positive=True)
    rho_m = strict_finite(mat["rho_g_cm3"], "matrix_reference.rho_g_cm3", positive=True)
    Vf = strict_finite(mat["fiber_volume_fraction_Vf"], "matrix_reference.fiber_volume_fraction_Vf", positive=True)
    require(Vf < 1.0, "fiber_volume_fraction_Vf must be < 1.0")
    rho_bar_ref = 0.05
    c_val = 2.0 - math.sqrt(2.0)
    L_bar = 2.0 * (math.sqrt(2.0) - 1.0)

    coords_opt, bars_opt = build_fn_bars(3, c_val, 1.0, 1.0, 2.0 * math.sqrt(2.0), A_diag=0.0)
    C_opt, rho_opt = homogenize_periodic_truss(3, coords_opt, bars_opt)
    C_opt = require_finite_stiffness_matrix(C_opt, 3, "C_opt")
    norm_K_opt = float((C_opt[0, 0] + 2.0 * C_opt[0, 1]) / (3.0 * rho_opt))
    norm_E100_opt = float((C_opt[0, 0] - 2.0 * (C_opt[0, 1] ** 2) / (C_opt[0, 0] + C_opt[0, 1])) / rho_opt)
    iso_pt = num_res["isotropic_braced_point"]
    norm_E_iso = strict_finite(iso_pt.get("E_iso_over_rhoE1"), "E_iso_over_rhoE1", positive=True)
    norm_K_iso = strict_finite(iso_pt.get("K_iso_over_rhoE1"), "K_iso_over_rhoE1", positive=True)

    solids = raw.get("reference_isotropic_solids")
    require(type(solids) is list and len(solids) >= 1, "reference_isotropic_solids must be a non-empty list")
    metal_hs: list[dict[str, Any]] = []
    max_m_E, max_m_K = 0.0, 0.0
    for m in solids:
        require(type(m) is dict and all(k in m for k in ("id", "Es_GPa", "nu_s", "rho_s_g_cm3")), "invalid metal entry")
        require(type(m["id"]) is str and len(m["id"]) >= 1, "metal id must be non-empty str")
        K_hs, _, E_hs = hashin_shtrikman_upper_bounds(
            strict_finite(m["Es_GPa"], "Es_GPa", positive=True),
            strict_finite(m["nu_s"], "nu_s"),
            rho_bar_ref,
        )
        rho_lat = rho_bar_ref * strict_finite(m["rho_s_g_cm3"], "rho_s_g_cm3", positive=True)
        s_E, s_K = E_hs / rho_lat, K_hs / rho_lat
        max_m_E = strict_finite_max(max_m_E, s_E, name="max_metal_specific_E")
        max_m_K = strict_finite_max(max_m_K, s_K, name="max_metal_specific_K")
        metal_hs.append({
            "id": m["id"],
            "rho_lattice_g_cm3": round(rho_lat, 5),
            "K_HS_GPa": round(K_hs, 6),
            "E_HS_GPa": round(E_hs, 6),
            "specific_K_HS_MPa_per_kg_m3": round(s_K, 4),
            "specific_E_HS_MPa_per_kg_m3": round(s_E, 4),
        })

    fiber_rows = raw.get("carbon_fibers")
    require(type(fiber_rows) is list and len(fiber_rows) >= 1, "carbon_fibers must be a non-empty list")
    fibers: list[dict[str, Any]] = []
    eta_tube = 0.80
    tube_gain = (1.0 + eta_tube**2) / (1.0 - eta_tube**2)
    for fib in fiber_rows:
        require(
            type(fib) is dict and all(k in fib for k in ("id", "Ef_GPa", "sigma_ult_MPa", "sigma_kink_comp_Vf60_MPa", "rho_f_g_cm3")),
            "invalid fiber entry",
        )
        require(type(fib["id"]) is str and len(fib["id"]) >= 1, "fiber id must be non-empty str")
        Ef = strict_finite(fib["Ef_GPa"], "Ef_GPa", positive=True)
        sig_f = strict_finite(fib["sigma_ult_MPa"], "sigma_ult_MPa", positive=True)
        sig_kink_MPa = strict_finite(fib["sigma_kink_comp_Vf60_MPa"], "sigma_kink_comp_Vf60_MPa", positive=True)
        rho_f = strict_finite(fib["rho_f_g_cm3"], "rho_f_g_cm3", positive=True)
        E1_c = Vf * Ef + (1.0 - Vf) * Em
        rho_c = Vf * rho_f + (1.0 - Vf) * rho_m
        sig_c_MPa = E1_c * ((sig_f / 1000.0) / Ef) * 1000.0
        require(
            sig_kink_MPa < 0.80 * sig_c_MPa,
            f"fiber {fib['id']} compressive kink allowable ({sig_kink_MPa} MPa) must be strictly below 80% of ROM tensile strength ({sig_c_MPa:.2f} MPa)",
        )
        rho_lat = rho_bar_ref * rho_c
        K_g, E100_g = norm_K_opt * rho_bar_ref * E1_c, norm_E100_opt * rho_bar_ref * E1_c
        E_i, K_i = norm_E_iso * rho_bar_ref * E1_c, norm_K_iso * rho_bar_ref * E1_c
        s_K, s_E100, s_Ei = K_g / rho_lat, E100_g / rho_lat, E_i / rho_lat
        r_over_L = math.sqrt(4.0 * (sig_c_MPa / 1000.0) / (math.pi**2 * E1_c))
        rho_crit_solid = (12.0 + 6.0 * math.sqrt(2.0)) * (math.pi * (L_bar**2) * (r_over_L**2)) * L_bar / 8.0
        comp_env = evaluate_tubular_compressive_limit(E1_c, sig_kink_MPa, eta_tube, L_bar, rho_bar_ref)
        fibers.append({
            "fiber_id": fib["id"],
            "Ef_GPa": Ef,
            "E1_CFRP_Vf60_GPa": round(E1_c, 4),
            "rho_CFRP_Vf60_g_cm3": round(rho_c, 4),
            "sigma_tensile_CFRP_Vf60_MPa": round(sig_c_MPa, 2),
            "sigma_kink_comp_CFRP_Vf60_MPa": round(sig_kink_MPa, 2),
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
                "K_eff_pin_jointed": comp_env["K_eff_pin_jointed"],
                "solid_rod_critical_r_over_L": round(r_over_L, 6),
                "solid_rod_critical_rho_bar": round(rho_crit_solid, 6),
                "hollow_tube_eta08_buckling_gain": round(tube_gain, 6),
                "hollow_tube_eta08_critical_rho_bar": round(rho_crit_solid / tube_gain, 6),
                "hollow_tube_eta08_R_over_t_mid": round(comp_env["R_over_t_mid"], 4),
                "hollow_tube_eta08_sp8007_thin_shell_valid": comp_env["sp8007_thin_shell_valid_R_over_t_ge_10"],
                "hollow_tube_eta08_sp8007_gamma": round(comp_env["sp8007_knockdown_gamma"], 6),
                "hollow_tube_eta08_sp8007_shell_MPa": round(comp_env["sigma_shell_sp8007_MPa"], 2),
                "compressive_cap_MPa": round(comp_env["sigma_comp_cap_MPa"], 2),
                "solid_rod_critical_rho_bar_compressive": round(comp_env["rho_crit_solid_comp"], 6),
                "hollow_tube_eta08_critical_rho_bar_compressive": round(comp_env["rho_crit_tube_comp"], 6),
                "at_5pct_sigma_euler_tube_MPa": round(comp_env["sigma_euler_tube_5pct_MPa"], 2),
                "at_5pct_sigma_allow_tube_MPa": round(comp_env["sigma_allow_tube_5pct_MPa"], 2),
                "at_5pct_effective_tube_gain": round(comp_env["effective_tube_gain_5pct"], 4),
                "at_5pct_governing_compressive_mode": comp_env["governing_mode_5pct"],
            },
        })

    return {
        "benchmark_relative_density": rho_bar_ref,
        "reference_metal_HS_upper_bounds": metal_hs,
        "cfrp_truss_benchmarks": fibers,
    }


def shear_within_bound(stiffness: np.ndarray, bound: float = NUMERICAL_BOUND) -> bool:
    """Validate entire 6x6 stiffness matrix finiteness and bound ALL shear & normal-shear components."""
    bound_val = strict_finite(bound, "bound", non_negative=True)
    mat = require_finite_stiffness_matrix(stiffness, 3, "stiffness")
    shear_max = strict_finite_max(
        *np.abs(mat[3:, :]).ravel(),
        *np.abs(mat[:, 3:]).ravel(),
        name="shear_stiffness",
    )
    return shear_max <= bound_val


def run_must_fail_controls(
    num_res: dict[str, Any],
    raw_data: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    require(type(num_res) is dict and type(num_res.get("isotropic_braced_point")) is dict, "num_res missing isotropic_braced_point")
    c_val = 2.0 - math.sqrt(2.0)
    coords, bars = build_fn_bars(3, c_val, 1.0, 1.0, 1.0)
    C_eq, rho_eq = homogenize_periodic_truss(3, coords, bars)
    C_eq = require_finite_stiffness_matrix(C_eq, 3, "C_eq")
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
        C_t = require_finite_stiffness_matrix(C_t, 3, "C_t")
        within = bool((C_t[0, 0] + 2.0 * C_t[0, 1]) / 3.0 <= r_t / 9.0 + 1e-12)
        require(within, "pin-jointed truss exceeded the bulk bound")

    e_iso = strict_finite(num_res["isotropic_braced_point"].get("E_iso_over_rhoE1"), "E_iso_over_rhoE1", positive=True)
    _, _, E_hs = hashin_shtrikman_upper_bounds(1.0, 0.25, 0.05)
    below_same_material_plate = e_iso < (E_hs / 0.05) - 0.1
    require(below_same_material_plate, "isotropic truss met the same-material plate bound")

    C_cut, _ = homogenize_periodic_truss(3, coords, [b for idx, b in enumerate(bars) if idx != 12])
    C_cut = require_finite_stiffness_matrix(C_cut, 3, "C_cut")
    cut_breaks_axial = bool(abs(C_cut[0, 0] - C_cut[1, 1]) > 1e-2)
    require(cut_breaks_axial, "cutting a cross bar preserved axial stiffness")

    c_off, b_off = build_fn_bars(3, 0.35, 1.0, 1.0, 1.0)
    C_off, _ = homogenize_periodic_truss(3, c_off, b_off)
    C_off = require_finite_stiffness_matrix(C_off, 3, "C_off")
    nu_off = float(C_off[0, 1] / (C_off[0, 0] + C_off[0, 1]))
    poisson_moves = abs(nu_off - 0.20) > 0.02
    require(poisson_moves, "equal-area Poisson ratio stayed at the unit-spring value")

    raw = load_strict_json_file(DATA_PATH, "data/carbon_fibers.json") if raw_data is None else raw_data
    require(type(raw) is dict and type(raw.get("matrix_reference")) is dict, "data/carbon_fibers.json missing matrix_reference")
    mat = raw["matrix_reference"]
    Em = strict_finite(mat.get("E_GPa"), "matrix_reference.E_GPa", positive=True)
    Vf = strict_finite(mat.get("fiber_volume_fraction_Vf"), "matrix_reference.fiber_volume_fraction_Vf", positive=True)
    require(Vf < 1.0, "fiber_volume_fraction_Vf must be < 1.0")
    fiber_rows = raw.get("carbon_fibers")
    require(type(fiber_rows) is list and len(fiber_rows) >= 1, "carbon_fibers must be a non-empty list")

    k13_E1_c: float | None = None
    k13_sig_tensile_MPa: float | None = None
    k13_sig_kink_MPa: float | None = None
    all_grades_kink_below_tensile = True
    for fib in fiber_rows:
        require(
            type(fib) is dict and all(k in fib for k in ("id", "Ef_GPa", "sigma_ult_MPa", "sigma_kink_comp_Vf60_MPa")),
            "invalid fiber entry in control #8",
        )
        Ef = strict_finite(fib["Ef_GPa"], "Ef_GPa", positive=True)
        sig_f = strict_finite(fib["sigma_ult_MPa"], "sigma_ult_MPa", positive=True)
        sig_k = strict_finite(fib["sigma_kink_comp_Vf60_MPa"], "sigma_kink_comp_Vf60_MPa", positive=True)
        E1_c = Vf * Ef + (1.0 - Vf) * Em
        sig_t = E1_c * ((sig_f / 1000.0) / Ef) * 1000.0
        if not (sig_k < 0.80 * sig_t):
            all_grades_kink_below_tensile = False
        if fib["id"] == "K13D2U":
            k13_E1_c = E1_c
            k13_sig_tensile_MPa = sig_t
            k13_sig_kink_MPa = sig_k
    require(
        k13_E1_c is not None and k13_sig_tensile_MPa is not None and k13_sig_kink_MPa is not None,
        "K13D2U entry missing from data/carbon_fibers.json",
    )

    L_bar_ctrl = 2.0 * (math.sqrt(2.0) - 1.0)
    k13_eta08 = evaluate_tubular_compressive_limit(k13_E1_c, k13_sig_kink_MPa, 0.80, L_bar_ctrl, 0.05)
    k13_eta08_tensile = evaluate_tubular_compressive_limit(k13_E1_c, k13_sig_tensile_MPa, 0.80, L_bar_ctrl, 0.05)
    k13_eta099 = evaluate_tubular_compressive_limit(k13_E1_c, k13_sig_kink_MPa, 0.99, L_bar_ctrl, 0.05)
    kink_and_shell_cap_tube_gain = bool(
        all_grades_kink_below_tensile
        and k13_eta08["governing_mode_5pct"] == "kink_band_compression"
        and k13_eta08["sigma_allow_tube_5pct_MPa"] < k13_eta08_tensile["sigma_allow_tube_5pct_MPa"]
        and k13_eta08["sigma_allow_tube_5pct_MPa"] < 0.5 * k13_sig_tensile_MPa
        and k13_eta08["effective_tube_gain_5pct"] <= 1.0
        and k13_eta099["sp8007_thin_shell_valid_R_over_t_ge_10"]
        and k13_eta099["governing_mode_5pct"] == "sp8007_local_shell_buckling"
        and k13_eta099["sigma_allow_tube_5pct_MPa"] < k13_sig_kink_MPa
        and k13_eta099["effective_tube_gain_5pct"] < 1.0
    )
    require(kink_and_shell_cap_tube_gain, "kink-band and SP-8007 shell limits failed to cap hollow-tube gain below ROM tensile")

    return [
        {"control": "equal_area_saturates_bulk_bound", "rejected": equal_area_misses_bound, "observed_K_over_Kmax": round(float(K_eq / K_max_eq), 8)},
        {"control": "unbraced_incentre_has_shear_rigidity", "rejected": unbraced_has_no_shear},
        {"control": "planted_shear_stiffness_is_accepted", "rejected": planted_shear_rejected},
        {"control": "pin_jointed_truss_exceeds_cauchy_maxwell_bulk_bound", "rejected": bound_holds},
        {"control": "isotropic_truss_exceeds_same_material_HS_plate_bound", "rejected": below_same_material_plate, "truss_E_over_rhoE1": round(e_iso, 8), "hs_plate_E_over_rhoEs_at_5pct": round(E_hs / 0.05, 8)},
        {"control": "cut_cross_bar_preserves_axial_stiffness", "rejected": cut_breaks_axial, "C11_after_cut": round(float(C_cut[0, 0]), 10), "C22_after_cut": round(float(C_cut[1, 1]), 10)},
        {"control": "unit_spring_poisson_holds_for_equal_area_bars_at_general_t", "rejected": poisson_moves, "nu_100_at_t035_equal_area": round(nu_off, 8), "unit_spring_nu_100": 0.2},
        {
            "control": "tensile_strength_and_uncapped_tube_gain_govern_cfrp_compression",
            "rejected": kink_and_shell_cap_tube_gain,
            "k13_rom_tensile_MPa": round(k13_sig_tensile_MPa, 2),
            "k13_eta08_allow_MPa_at_5pct": round(k13_eta08["sigma_allow_tube_5pct_MPa"], 2),
            "k13_eta099_sp8007_shell_MPa": round(k13_eta099["sigma_shell_sp8007_MPa"], 2),
        },
    ]


def main(argv: list[str] | None = None) -> int:
    if argv is not None:
        if type(argv) is not list or any(type(item) is not str for item in argv):
            print("[FAIL] CheckFailure: argv must be a list of exact str", file=sys.stderr)
            return 1
        try:
            for item in argv:
                _require_utf8_no_surrogate(item, "argv")
        except CheckFailure as exc:
            print(f"[FAIL] CheckFailure: {exc}", file=sys.stderr)
            return 1
    parser = argparse.ArgumentParser(description="Verify fibral-rigidity results.")
    parser.add_argument("--write", action="store_true", help="Write results/certificate.json")
    args = parser.parse_args(argv)

    try:
        raw_data = load_strict_json_file(DATA_PATH, "data/carbon_fibers.json")
        sym = run_symbolic_checks()
        num = run_numerical_truss_checks()
        cfrp = run_cfrp_benchmark_checks(num, raw_data=raw_data)
        controls = run_must_fail_controls(num, raw_data=raw_data)
        cert = {
            "repository": "Foundation-0/fibral-rigidity",
            "certificate_version": "1.0.0",
            "inputs": {"data/carbon_fibers.json_sha256": canonical_data_sha256(raw_data)},
            "symbolic_verification": sym,
            "numerical_homogenization": num,
            "cfrp_material_benchmarks": cfrp,
            "must_fail_controls": controls,
        }
        expected_bytes = render_canonical_cert_bytes(cert)

        if args.write:
            write_canonical_cert_bytes(CERT_PATH, expected_bytes)

        verify_canonical_cert_bytes(CERT_PATH, expected_bytes)
    except CheckFailure as exc:
        print(f"[FAIL] CheckFailure: {exc}", file=sys.stderr)
        return 1

    print(f"[PASS] Symbolic (n=2..7), Numerical ({num['coordinate_homogenization_cases_checked']} cases, discrepancy within 1e-9), CFRP (5 grades), Controls ({len(controls)}/{len(controls)}): OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
