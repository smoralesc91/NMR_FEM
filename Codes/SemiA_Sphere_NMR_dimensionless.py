from __future__ import annotations
import numpy as np
import mpmath
import scipy.optimize as opt

__version__ = "1.1-dimless_flexible"


def _g_eps_dimless(eps: float, kappa: float) -> float:
    """
    Characteristic equation for the Brownstein–Tarr eigenvalue problem in spherical geometry:

        g(ε) = 1 − ε cot(ε) − κ

    where κ = R ρ / D is the classical dimensionless surface relaxivity parameter.
    Note: In the context of the paper, κ corresponds to the product α * φ_R.
    """
    return 1.0 - eps * float(mpmath.cot(eps)) - kappa


def _find_roots_bracketed_dimless(kappa: float,
                                  n_terms: int,
                                  tol: float = 1e-10,
                                  maxiter: int = 200) -> np.ndarray:
    """
    Computes the first `n_terms` positive roots of the Brownstein–Tarr characteristic equation
    g(ε) = 0 using Brent's method on the intervals ((i−1)π, iπ), i = 1..n_terms.

    Small offsets are applied to avoid the singularities of cot(ε) at integer multiples of π.
    If bracketing fails, mpmath.findroot is used as a fallback.
    """
    roots = []
    for i in range(1, n_terms + 1):
        a = (i - 1) * np.pi + 1e-12   # avoid singularity at (i−1)π
        b = i * np.pi - 1e-12         # avoid singularity at iπ

        def g_np(x):
            return 1.0 - x / np.tan(x) - kappa

        fa, fb = g_np(a), g_np(b)

        # Attempt midpoint re-bracketing if no sign change
        if fa * fb > 0.0:
            mid = 0.5 * (a + b)
            fm = g_np(mid)
            if fa * fm < 0:
                b, fb = mid, fm
            elif fm * fb < 0:
                a, fa = mid, fm
            else:
                # Fallback to safeguarded Newton via mpmath
                r = float(mpmath.findroot(
                    lambda x: _g_eps_dimless(x, kappa), (mid,)
                ))
                roots.append(r)
                continue

        r = opt.brentq(
            lambda x: 1.0 - x / np.tan(x) - kappa,
            a, b, xtol=tol, maxiter=maxiter
        )
        roots.append(r)

    return np.array(roots, dtype=float)


def NMR_SemiA_sphere_dimless(
    radius=1.0,                      # [m] pore radius R > 0
    T2B=1.0,                         # [s] bulk relaxation time T2B > 0 (used for φ_R)
    diffusion=1.0,                   # [m^2/s] diffusion coefficient D > 0
    rho=1.0,                         # [m/s] surface relaxivity ρ
    B_0=0.05, Temp=303.15, fluid='water',  # kept for compatibility; not used here
    n_terms=50,                      # number of eigenmodes in the truncated series
    t_0=0.0, t_f=1.0, dt=1e-2,       # Dimensionless time generation params
    tau_array=None,                  # *** NEW: Optional array of dimensionless time points
    volume_=True,                    # compatibility argument; no effect
    normalize=False,                 # optional renormalization by S(τ_0)
    return_data='all',               # output control
    progress=False                   # progress bar placeholder; unused
):
    """
    Dimensionless semi-analytical Brownstein–Tarr solution for a spherical pore
    with Robin boundary condition.

    ------------------------
    Dimensionless formulation
    ------------------------
    All variables are non-dimensionalized as:

        τ = t / T2B
        φ = r / sqrt(D T2B)
        μ = m / m_s

    The dimensionless pore radius is:

        φ_R = R / sqrt(D T2B)

    Characteristic BT parameters:
        κ = R ρ / D = α * φ_R        (Magnetic Damköhler number)
        ε_i                          eigenvalues from g(ε_i) = 0
        β_i = (ε_i² + φ_R²)/φ_R²     modal decay rates

    Dimensionless signal (Eq. 54 in paper):
        S(τ) = Σ I_i exp[ − τ β_i ]

    Modal intensities (Eq. 57 in paper):
        I_i = 12 [sin ε_i − ε_i cos ε_i]²
              -------------------------------------------
              ε_i³ [2 ε_i − sin(2 ε_i)]

    Intensities are normalized such that Σ I_i = 1, ensuring S(0) = 1.

    ------------------------
    Inputs
    ------------------------
    tau_array : array-like, optional
        If provided, the signal is computed exactly at these dimensionless time points.
        Overrides t_0, t_f, dt. Useful for direct comparison with FEM solvers.

    Other arguments match the dimensional version for API compatibility.

    ------------------------
    Outputs
    ------------------------
    return_data='all':
        (tau, S_tau, eps, tau_modes, I, S_assemble)
    """
    # --- Basic physical consistency checks
    R = float(radius)
    if R <= 0.0:
        raise ValueError("`radius` must be > 0.")

    D = float(diffusion)
    if D <= 0.0:
        raise ValueError("`diffusion` must be > 0 for Brownstein–Tarr model.")

    if T2B is None or T2B <= 0.0:
        raise ValueError("`T2B` must be > 0 to define dimensionless variables.")
    T2B = float(T2B)

    # --- Dimensionless parameters φ_R and κ
    # Matches Eq. 50: phi_R = R / sqrt(D*T2B)
    phi_R = R / np.sqrt(D * T2B)
    phi_R2 = phi_R**2

    # Matches Eq. 55 RHS: alpha * phi_R = (rho*sqrt(T2B/D)) * (R/sqrt(D*T2B)) = rho*R/D
    kappa = R * float(rho) / D

    # --- Compute BT eigenvalues ε_i (Eq. 55)
    eps = _find_roots_bracketed_dimless(kappa, n_terms=n_terms)

    # --- Dimensionless modal intensities (Eq. 57)
    sin_eps = np.sin(eps)
    cos_eps = np.cos(eps)
    num = 12.0 * (sin_eps - eps * cos_eps) ** 2
    denom = eps**3 * (2.0 * eps - np.sin(2.0 * eps))
    I = num / denom

    # Normalize intensities to ensure Σ I = 1
    sum_I = np.sum(I)
    if sum_I == 0.0:
        raise ZeroDivisionError(
            "Sum of modal intensities is zero. Check physical parameters or n_terms."
        )
    I = I / sum_I

    # --- Modal decay rates and characteristic times
    # Matches Eq. 54 exponent factor: (epsilon^2 + phi_R^2) / phi_R^2
    beta = (eps**2 + phi_R2) / phi_R2
    tau_modes = 1.0 / beta

    # --- Dimensionless time grid
    if tau_array is not None:
        # Use provided grid for direct validation
        tau = np.array(tau_array, dtype=float)
    else:
        # Generate grid based on parameters
        nt = int(round((t_f - t_0) / dt)) + 1
        if nt < 2:
            raise ValueError("Time grid must contain at least 2 points.")
        tau = np.linspace(t_0, t_f, nt)

    # --- Evaluate the modal sum S(τ)
    # S(τ) = Σ I_i exp(−τ β_i)
    # We use outer product to evaluate all modes at all time steps efficiently
    decay = np.exp(-np.outer(tau, beta))
    S_tau = decay @ I

    # Optional normalization (usually redundant as Sum(I)=1, but safe for numerics)
    if normalize:
        S0 = S_tau[0]
        if S0 == 0.0:
            raise ZeroDivisionError("S(τ_0) is zero; cannot normalize signal.")
        S_tau = S_tau / S0

    S_assemble = float(S_tau[-1])

    # --- Select output format
    if return_data == 'all':
        return tau, S_tau, eps, tau_modes, I, S_assemble
    elif return_data == 'time-mag':
        return tau, S_tau
    elif return_data == 'mag_amounts':
        return S_tau
    elif return_data == 'mag_assemble':
        return S_assemble
    else:
        # Default fallback
        return tau, S_tau, eps, tau_modes, I, S_assemble


NMR_SemiA_sphere_dimless.__version__ = __version__
