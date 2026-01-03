from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np


@dataclass(frozen=True)
class AchTestInput:
    """
    Inputs for ACHTEST.

    nu_hz: dict band -> effective frequency in Hz
    O:     dict band -> observable value (e.g., Δt in days, Δm in mag, θ in arcsec)
    sigma: dict band -> statistical uncertainty (same unit as O)
    sys:   dict band -> systematic budget (same unit as O)

    controls_source: optional dict control_name -> dict band -> value
    controls_inst:   optional dict control_name -> dict band -> value
    """
    nu_hz: Dict[str, float]
    O: Dict[str, float]
    sigma: Dict[str, float]
    sys: Dict[str, float]
    controls_source: Optional[Dict[str, Dict[str, float]]] = None
    controls_inst: Optional[Dict[str, Dict[str, float]]] = None


@dataclass(frozen=True)
class AchTestResult:
    verdict: str  # "REJECT" or "NON_REJECT"
    k_sigma: float
    b: float
    b_se: float
    B_bound: float
    a: float
    a_se: float
    n_bands: int

    # diagnostics
    explained_by: str  # "none", "instrument", "source", "both"
    b_no_controls: float
    b_se_no_controls: float

    # robustness
    loo_b: Dict[str, float]               # leave-one-out slope estimates
    perm_p_value: Optional[float]         # permutation p-value (two-sided)


def _stack_vectors(
    bands: List[str],
    nu_hz: Dict[str, float],
    O: Dict[str, float],
    sigma: Dict[str, float],
    sys: Dict[str, float],
    controls: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Build y, x (ln nu), weights, and control matrix columns (if any).
    """
    lnnu = np.array([np.log(nu_hz[b]) for b in bands], dtype=float)
    y = np.array([O[b] for b in bands], dtype=float)

    # total variance (stat + sys) : conservative
    var = np.array([(sigma[b] ** 2) + (sys[b] ** 2) for b in bands], dtype=float)
    if np.any(var <= 0):
        raise ValueError("All bands must have positive (sigma^2 + sys^2).")
    w = 1.0 / var

    ctrl_names: List[str] = []
    C = None
    if controls:
        ctrl_names = sorted(list(controls.keys()))
        C = np.column_stack([[controls[name][b] for b in bands] for name in ctrl_names]).astype(float)

        # Standardize controls to prevent scale dominance (mean 0, std 1)
        C_mean = C.mean(axis=0)
        C_std = C.std(axis=0)
        C_std[C_std == 0.0] = 1.0
        C = (C - C_mean) / C_std

    return lnnu, y, w, (C, ctrl_names)


def _wls_fit(y: np.ndarray, X: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Weighted least squares:
      beta = (X' W X)^(-1) X' W y
    Returns beta and covariance matrix of beta.
    """
    W = np.diag(w)
    XtW = X.T @ W
    XtWX = XtW @ X

    # Solve robustly
    try:
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        XtWX_inv = np.linalg.pinv(XtWX)

    beta = XtWX_inv @ (XtW @ y)

    # Estimate residual variance (weighted) for covariance scaling
    yhat = X @ beta
    resid = y - yhat

    # Effective dof
    dof = max(1, X.shape[0] - X.shape[1])

    # Weighted residual sum of squares
    wrss = float(resid.T @ W @ resid)

    # Scale: sigma^2_hat
    s2 = wrss / dof
    cov = XtWX_inv * s2
    return beta, cov


def _center_over_bands(y: np.ndarray) -> np.ndarray:
    """
    r(ν) = O(ν) - mean_over_bands(O)
    """
    return y - float(np.mean(y))


def _fit_slope(
    bands: List[str],
    inp: AchTestInput,
    use_controls_source: bool,
    use_controls_inst: bool,
) -> Tuple[float, float, float, float, str]:
    """
    Fit centered residuals r vs lnnu plus optional standardized controls.
    Returns: b, b_se, a, a_se, explained_by_tag (not final)
    """
    lnnu, y_raw, w, (Csrc, src_names) = _stack_vectors(
        bands, inp.nu_hz, inp.O, inp.sigma, inp.sys, inp.controls_source if use_controls_source else None
    )
    _, _, _, (Cinst, inst_names) = _stack_vectors(
        bands, inp.nu_hz, inp.O, inp.sigma, inp.sys, inp.controls_inst if use_controls_inst else None
    )

    y = _center_over_bands(y_raw)

    # Design matrix: [1, lnnu, controls...]
    cols = [np.ones_like(lnnu), lnnu]

    if Csrc is not None:
        cols.append(Csrc)
    if Cinst is not None:
        cols.append(Cinst)

    # Flatten any appended matrices
    X_parts = []
    for c in cols:
        if c.ndim == 1:
            X_parts.append(c.reshape(-1, 1))
        else:
            X_parts.append(c)
    X = np.concatenate(X_parts, axis=1)

    beta, cov = _wls_fit(y, X, w)

    a = float(beta[0])
    b = float(beta[1])
    a_se = float(np.sqrt(max(0.0, cov[0, 0])))
    b_se = float(np.sqrt(max(0.0, cov[1, 1])))

    tag = f"src={len(src_names)} inst={len(inst_names)}"
    return b, b_se, a, a_se, tag


def _leave_one_out_slopes(inp: AchTestInput, k_sigma: float, include_controls: bool) -> Dict[str, float]:
    bands_all = sorted(inp.nu_hz.keys())
    loo = {}
    for drop in bands_all:
        bands = [b for b in bands_all if b != drop]
        if len(bands) < 3:
            loo[drop] = float("nan")
            continue

        b, b_se, *_ = _fit_slope(
            bands, inp,
            use_controls_source=include_controls,
            use_controls_inst=include_controls,
        )
        loo[drop] = b
    return loo


def _permutation_p_value(
    inp: AchTestInput,
    include_controls: bool,
    n_perm: int = 2000,
    seed: int = 0
) -> float:
    """
    Permute O values across bands (keeping nu fixed) to estimate how often |b_perm| >= |b_obs|.
    Two-sided p-value.
    """
    rng = np.random.default_rng(seed)
    bands = sorted(inp.nu_hz.keys())

    # observed
    b_obs, b_se_obs, *_ = _fit_slope(
        bands, inp,
        use_controls_source=include_controls,
        use_controls_inst=include_controls,
    )

    O_vals = np.array([inp.O[b] for b in bands], dtype=float)

    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(O_vals)
        O_perm = {b: float(perm[i]) for i, b in enumerate(bands)}
        inp_perm = AchTestInput(
            nu_hz=inp.nu_hz,
            O=O_perm,
            sigma=inp.sigma,
            sys=inp.sys,
            controls_source=inp.controls_source,
            controls_inst=inp.controls_inst,
        )
        b_p, _, *_ = _fit_slope(
            bands, inp_perm,
            use_controls_source=include_controls,
            use_controls_inst=include_controls,
        )
        if abs(b_p) >= abs(b_obs):
            count += 1

    # add-one smoothing
    return float((count + 1) / (n_perm + 1))


def run_achtest(
    inp: AchTestInput,
    k_sigma: float = 3.0,
    include_controls: bool = True,
    permutation_test: bool = True,
    n_perm: int = 2000,
    seed: int = 0,
) -> AchTestResult:
    """
    Run ACHTEST.

    - Fits centered residual r = O - mean(O) vs ln(nu).
    - Optionally includes standardized source/instrument controls.
    - Produces verdict and bound.

    Verdict rule (ErvraLab):
      REJECT if |b| > k * b_se
      else NON_REJECT, with bound B = k * b_se

    Attribution:
      Compare b without controls vs with controls:
        - if significant without controls but not with instrument controls => "instrument"
        - if significant without controls but not with source controls => "source"
        - if significant without controls but not with both => "both"
        - else "none"
    """
    bands = sorted(inp.nu_hz.keys())
    if set(bands) != set(inp.O.keys()) or set(bands) != set(inp.sigma.keys()) or set(bands) != set(inp.sys.keys()):
        raise ValueError("nu_hz, O, sigma, sys must have the same band keys.")

    if len(bands) < 3:
        raise ValueError("Need at least 3 bands for a meaningful slope test.")

    # Fit without any controls (baseline)
    b0, b0_se, a0, a0_se, _ = _fit_slope(
        bands, inp,
        use_controls_source=False,
        use_controls_inst=False,
    )
    sig0 = abs(b0) > (k_sigma * b0_se)

    # Fit with controls depending on include_controls
    b, b_se, a, a_se, _ = _fit_slope(
        bands, inp,
        use_controls_source=include_controls,
        use_controls_inst=include_controls,
    )
    sig = abs(b) > (k_sigma * b_se)
    verdict = "REJECT" if sig else "NON_REJECT"
    B_bound = float(k_sigma * b_se)

    # Attribution
    explained_by = "none"
    if sig0 and include_controls:
        # instrument-only
        bI, bI_se, *_ = _fit_slope(bands, inp, use_controls_source=False, use_controls_inst=True)
        sigI = abs(bI) > (k_sigma * bI_se)

        # source-only
        bS, bS_se, *_ = _fit_slope(bands, inp, use_controls_source=True, use_controls_inst=False)
        sigS = abs(bS) > (k_sigma * bS_se)

        # If baseline was significant but instrument-only fit is not => instrument explains
        inst_explains = (not sigI)
        src_explains = (not sigS)

        if inst_explains and src_explains:
            explained_by = "both"
        elif inst_explains:
            explained_by = "instrument"
        elif src_explains:
            explained_by = "source"
        else:
            explained_by = "none"

    loo_b = _leave_one_out_slopes(inp, k_sigma=k_sigma, include_controls=include_controls)

    p_val = None
    if permutation_test:
        p_val = _permutation_p_value(inp, include_controls=include_controls, n_perm=n_perm, seed=seed)

    return AchTestResult(
        verdict=verdict,
        k_sigma=float(k_sigma),
        b=float(b),
        b_se=float(b_se),
        B_bound=float(B_bound),
        a=float(a),
        a_se=float(a_se),
        n_bands=len(bands),
        explained_by=explained_by,
        b_no_controls=float(b0),
        b_se_no_controls=float(b0_se),
        loo_b=loo_b,
        perm_p_value=p_val,
    )
