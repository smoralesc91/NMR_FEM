from __future__ import annotations
import numpy as np
import mpmath
import scipy.optimize as opt

try:
    from Functions_NMR import mag_sat
except Exception:
    mag_sat = None

__version__ = "1.3-progress-norm"

def _g_eps(eps: float, kappa: float) -> float:
    # g(ε) = 1 - ε cot ε - κ
    return 1.0 - eps*float(mpmath.cot(eps)) - kappa

def _find_roots_bracketed(kappa: float, n_terms: int,
                          tol: float = 1e-10, maxiter: int = 200) -> np.ndarray:
    """
    Busca n_terms raíces de g(ε)=0 en ((i-1)π, iπ).
    Ajusta extremos para evitar singulares y hace fallback si no hay cambio de signo claro.
    """
    roots = []
    for i in range(1, n_terms+1):
        a = (i-1)*np.pi + 1e-12
        b = i*np.pi - 1e-12

        def g(x): return 1.0 - x/np.tan(x) - kappa

        fa, fb = g(a), g(b)
        if fa*fb > 0.0:
            mid = 0.5*(a+b)
            fm = g(mid)
            if fa*fm < 0: b, fb = mid, fm
            elif fm*fb < 0: a, fa = mid, fm
            else:
                try:
                    r = float(mpmath.findroot(lambda x: _g_eps(x, kappa), (mid,)))
                    roots.append(r); continue
                except:
                    raise RuntimeError("No se pudo brackear ni converger con findroot.")
        r = opt.brentq(lambda x: 1.0 - x/np.tan(x) - kappa, a, b, xtol=tol, maxiter=maxiter)
        roots.append(r)
    return np.array(roots, dtype=float)

def NMR_SemiA_sphere(radius = 1.0,
                     T2B = 1.0,
                     diffusion = 1.0,
                     rho = 1.0,
                     B_0 = 0.05,
                     Temp = 303.15,
                     fluid = 'water',
                     n_terms = 50,
                     t_0 = 0.0,
                     t_f = 1.0,
                     dt = 1e-2,
                     volume_ = False,
                     return_data = 'all',
                     progress = True,
                     normalize = True):
    """
    Solución semianalítica (Brownstein–Tarr) para esfera con Robin y T2B.
    - volume_=False: ⟨m⟩_V(t)
    - volume_=True : M(t)=V⟨m⟩_V(t), V=4πR^3/3
    - normalize=True: reescala A_n para que Σ A_n = 1 con serie truncada.
    """
    # Parámetros
    R = float(radius)
    D = float(diffusion)
    invT2B = 0.0 if (T2B is None or T2B <= 0.0) else 1.0/float(T2B)
    V = (4.0/3.0)*np.pi*R**3
    kappa = R*float(rho)/D

    # Magnetización equilibrio
    try:
        M0 = float(mag_sat(B_0, Temp, fluid)) if callable(mag_sat) else 1.0
    except Exception:
        M0 = 1.0

    # Raíces ε_n
    eps = _find_roots_bracketed(kappa, n_terms=n_terms)

    # Tiempos modales
    Tn = 1.0 / (D*(eps**2)/R**2 + invT2B)

    # Coeficientes A_n (¡factor 6!)
    sin_eps, cos_eps = np.sin(eps), np.cos(eps)
    num = (sin_eps - eps*cos_eps)**2
    denom = eps**3 * (eps - sin_eps*cos_eps)
    An = 6.0 * num / denom

    # Renormalización opcional (para que m(0)=M0 con truncamiento)
    if normalize:
        S = An.sum()
        if S > 0:
            An = An / S

    # Tiempo
    nt = int(round((t_f - t_0)/dt)) + 1
    t = np.linspace(t_0, t_f, nt)

    # Evaluación temporal (tqdm opcional)
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
            mag_amounts[i] = M0 * np.sum(An * np.exp(- t[i] / Tn))
    else:
        decay = np.exp(-np.outer(t, 1.0/Tn))
        mag_amounts = M0 * (decay @ An)

    if volume_:
        mag_amounts = V * mag_amounts

    mag_assemble = mag_amounts[-1]

    if return_data == 'all':
        return t, mag_amounts, eps, Tn, An, mag_assemble
    elif return_data == 'time-mag':
        return t, mag_amounts
    elif return_data == 'mag_assemble':
        return mag_assemble
    elif return_data == 'mag_amounts':
        return mag_amounts

NMR_SemiA_sphere.__version__ = __version__
