from __future__ import annotations
import numpy as np
import mpmath
import scipy.optimize as opt

try:
    # Optional: user-provided equilibrium magnetization function
    from Functions_NMR import mag_sat
except Exception:
    mag_sat = None

__version__ = "1.6-clean-args"


def _g_eps(eps: float, kappa: float) -> float:
    """
    Root function for the spherical Brownstein–Tarr eigenvalue problem with Robin BC:

        g(ε) = 1 - ε cot(ε) - κ

    where κ = R ρ / D is the dimensionless surface relaxivity.
    """
    return 1.0 - eps * float(mpmath.cot(eps)) - kappa


def _find_roots_bracketed(kappa: float,
                          n_terms: int,
                          tol: float = 1e-10,
                          maxiter: int = 200) -> np.ndarray:
    """
    Find the first `n_terms` positive roots of g(ε)=0 within ((i-1)π, iπ), i=1..n_terms.
    Uses Brent's method with tiny endpoint shifts to avoid poles of cot(ε).
    Falls back to mpmath.findroot if initial bracketing fails.
    """
    roots = []
    for i in range(1, n_terms + 1):
        a = (i - 1) * np.pi + 1e-12   # avoid singularity at (i-1)π
        b = i * np.pi - 1e-12         # avoid singularity at iπ

        def g(x):  # lightweight local function using numpy trig
            return 1.0 - x / np.tan(x) - kappa

        fa, fb = g(a), g(b)
        if fa * fb > 0.0:
            # try the midpoint to re-bracket
            mid = 0.5 * (a + b)
            fm = g(mid)
            if fa * fm < 0:
                b, fb = mid, fm
            elif fm * fb < 0:
                a, fa = mid, fm
            else:
                # safeguarded Newton via mpmath
                r = float(mpmath.findroot(lambda x: _g_eps(x, kappa), (mid,)))
                roots.append(r)
                continue

        r = opt.brentq(lambda x: 1.0 - x / np.tan(x) - kappa, a, b, xtol=tol, maxiter=maxiter)
        roots.append(r)
    return np.array(roots, dtype=float)


def NMR_SemiA_sphere(
    radius=1.0,                      # [m] pore radius R > 0
    T2B=1.0,                         # [s] bulk transverse relaxation; set <=0 or None to disable
    diffusion=1.0,                   # [m^2/s] self/effective diffusion coefficient D > 0
    rho=1.0,                         # [m/s] surface relaxivity (transverse)
    B_0=0.05, Temp=303.15, fluid='water',  # used only if mag_sat() is available
    n_terms=50,                      # number of eigenmodes kept in the truncated series
    t_0=0.0, t_f=1.0, dt=1e-2,       # [s] time window and step (uniform)
    volume_=True,                    # if True: return M(t)=V⟨m⟩_V; else: volume-averaged ⟨m⟩_V
    normalize=False,                 # if True: scale output so that signal at t_0 equals 1
    return_data='all',               # 'all' | 'time-mag' | 'mag_amounts' | 'mag_assemble'
    progress=False                   # use tqdm if available (cosmetic only)
):
    """
    Semi-analytic Brownstein–Tarr solution for a spherical pore with Robin boundary
    condition and optional bulk relaxation T2B. Closed-form eigen-expansion (no mesh).

    Physics
    -------
    - κ = R ρ / D
    - Eigenvalues ε_n solve: 1 - ε_n cot(ε_n) - κ = 0, n=1..N
    - Modal decays: 1/T_n = D ε_n^2 / R^2 + 1/T2B  (with 1/T2B = 0 if T2B<=0 or None)
    - Spherical amplitudes:
        A_n = 6 * (sin ε_n - ε_n cos ε_n)^2 / [ ε_n^3 ( ε_n - sin ε_n cos ε_n ) ]
    - Volume-averaged magnetization (up to M0):
        ⟨m⟩_V(t) = Σ_n A_n exp( - (t - t_0) / T_n )

    Scaling & Output
    ----------------
    - If mag_sat() is available: M0 = mag_sat(B_0, Temp, fluid); else M0 = 1.
    - If volume_=True, return absolute amount:  M(t) = V ⟨m⟩_V(t),  V = 4πR^3/3
      Otherwise, return volume-averaged ⟨m⟩_V(t).
    - If normalize=True, rescale the computed time series so that m(t_0) = 1 (output in [0,1]).

    Returns
    -------
    If return_data == 'all':
        (t, mag_amounts, eps, Tn, An, mag_assemble)
        - t : time samples
        - mag_amounts : ⟨m⟩_V(t) or M(t) depending on `volume_` and `normalize`
        - eps : eigenvalues ε_n
        - Tn : modal times T_n
        - An : modal amplitudes (no ΣA_n renormalization)
        - mag_assemble : mag_amounts[-1]
    If return_data == 'time-mag': (t, mag_amounts)
    If return_data == 'mag_amounts': mag_amounts
    If return_data == 'mag_assemble': float
    """
    # --- Basic validation
    R = float(radius)
    if R <= 0.0:
        raise ValueError("`radius` must be > 0.")
    D = float(diffusion)
    if D <= 0.0:
        raise ValueError("`diffusion` must be > 0 for the BT spherical model.")
    invT2B = 0.0 if (T2B is None or T2B <= 0.0) else 1.0 / float(T2B)

    # --- Geometry and κ
    V = (4.0 / 3.0) * np.pi * R**3
    kappa = R * float(rho) / D

    # --- Equilibrium magnetization M0 (absolute scaling, optional)
    try:
        M0 = float(mag_sat(B_0, Temp, fluid)) if callable(mag_sat) else 1.0
    except Exception:
        M0 = 1.0

    # --- Eigenvalues ε_n
    eps = _find_roots_bracketed(kappa, n_terms=n_terms)

    # --- Modal times T_n
    Tn = 1.0 / (D * (eps**2) / R**2 + invT2B)

    # --- Modal amplitudes A_n
    sin_eps, cos_eps = np.sin(eps), np.cos(eps)
    num = (sin_eps - eps * cos_eps) ** 2
    denom = eps**3 * (eps - sin_eps * cos_eps)
    An = 6.0 * num / denom

    # --- Time grid
    nt = int(round((t_f - t_0) / dt)) + 1
    if nt < 2:
        raise ValueError("Time grid must contain at least 2 points. Check t_0, t_f, dt.")
    t = np.linspace(t_0, t_f, nt)

    # --- Time evaluation (with optional tqdm)
    use_tqdm = False
    if progress:
        try:
            from tqdm import tqdm
            use_tqdm = True
        except Exception:
            use_tqdm = False

    if use_tqdm:
        mag_amounts = np.empty(nt, dtype=float)
        iterator = tqdm(range(nt), desc="Semi-analytic", unit="step", total=nt)
        for i in iterator:
            mag_amounts[i] = M0 * np.sum(An * np.exp(-(t[i] - t_0) / Tn))
    else:
        decay = np.exp(-np.outer(t - t_0, 1.0 / Tn))
        mag_amounts = M0 * (decay @ An)

    # --- Absolute amount vs. volume-averaged
    if volume_:
        mag_amounts = V * mag_amounts  # M(t) = V ⟨m⟩_V(t)

    # --- Optional output normalization to enforce m(t_0)=1
    if normalize:
        m0 = mag_amounts[0]
        if m0 == 0.0:
            raise ZeroDivisionError("First sample is zero; cannot normalize to 1.")
        mag_amounts = mag_amounts / m0

    # Convenience scalar
    mag_assemble = float(mag_amounts[-1])

    # --- Return selector
    if return_data == 'all':
        return t, mag_amounts, eps, Tn, An, mag_assemble
    elif return_data == 'time-mag':
        return t, mag_amounts
    elif return_data == 'mag_amounts':
        return mag_amounts
    elif return_data == 'mag_assemble':
        return mag_assemble
    else:
        # Fallback to the richest option
        return t, mag_amounts, eps, Tn, An, mag_assemble


# Expose version for easy tracking
NMR_SemiA_sphere.__version__ = __version__