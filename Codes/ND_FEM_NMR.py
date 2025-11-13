from dolfin import *
import numpy as np
import math

__nd_version__ = "0.1"

# Optional user utilities (same pattern as FEM_NMR)
try:
    from Functions_NMR import mag_sat, mesh_statistics
except Exception:
    mag_sat = None
    mesh_statistics = None

# Silence FEniCS logging by default (user may override)
set_log_level(LogLevel.ERROR)


def ND_BT_FEM(
    radius=1.0,                      # [m] physical pore radius R > 0
    mesh_res=300,                    # number of intervals on [0, φ_R]
    mesh_stats=False,                # call user mesh_statistics(mesh, T2B, rho) if available (for reference)
    T2B=1.0,                         # [s] bulk transverse relaxation
    diffusion=1.0,                   # [m^2/s] self/effective diffusion coefficient D > 0
    rho=1.0,                         # [m/s] surface relaxivity (transverse)
    B_0=0.05, Temp=303.15, fluid='water',  # used only if user wants to reconstruct dimensional M(t)
    t_0=0.0, t_f=1.0, dt=1e-2,       # physical time window and step in seconds
    volume_=True,                    # if True, return volume-weighted signal 4π∫ μ φ² dφ; else normalized average
    normalize=False,                 # if True, scale signal so that M(t_0) = 1
    return_data='all',               # 'all' | 'time-mag' | 'mag_amounts' | 'mag_assemble'
    linear_solver='mumps',           # 'mumps' | 'cg'
    progress=False,                  # use tqdm if available
    store_fields_every=None,         # None or integer N to store μ(φ,τ) every N steps
    V_degree=2,                      # polynomial degree for CG space
    comm=None                        # MPI communicator (None -> MPI.comm_world)
):
    r"""
    Finite-Element solver for the **non-dimensional** spherically symmetric Bloch–Torrey equation
    with surface relaxation, as developed in the non-dimensional model:

        ∂_τ μ = ∂_{φφ} μ + (2/φ) ∂_φ μ - μ,     φ ∈ (0, φ_R)
        μ(φ, 0) = 1,
        ∂_φ μ(0, τ) = 0,                        (symmetry at φ = 0)
        ∂_φ μ(φ_R, τ) + α μ(φ_R, τ) = 0.        (Robin at φ = φ_R)

    where:
        τ   = t / T2B                        (dimensionless time),
        φ   = r / sqrt(D * T2B)              (dimensionless radius),
        μ   = m / m_s                        (normalized magnetization),
        α   = ρ * sqrt(D * T2B) / D          (dimensionless surface relaxivity),
        φ_R = R / sqrt(D * T2B)              (dimensionless pore radius).

    The numerical scheme is backward Euler in (dimensionless) time τ and a standard CG FEM
    in the 1D radial coordinate φ, using the weak form:

        Find μ^n ∈ V such that for all w ∈ V:

            ∫_Ω μ^n w dφ
          + Δτ ∫_Ω (∂_φ μ^n)(∂_φ w) dφ
          + Δτ α ∫_{∂_R Ω} μ^n w dS
          + Δτ ∫_Ω (2/φ) (∂_φ μ^n) w dφ
          + Δτ ∫_Ω μ^n w dφ
          = ∫_Ω μ^{n-1} w dφ.

    Parameters
    ----------
    radius : float
        Physical pore radius R [m]. Used only to construct the non-dimensional radius φ_R.
    mesh_res : int
        Number of elements in the 1D mesh [0, φ_R].
    mesh_stats : bool
        If True and Functions_NMR.mesh_statistics is available, it is called for user diagnostics.
        Note: the mesh is built in the non-dimensional coordinate φ, but (R, T2B, rho) are passed.
    T2B : float
        Bulk transverse relaxation time [s]. Must be > 0 for a meaningful non-dimensionalization.
    diffusion : float
        Self/effective diffusion coefficient D [m^2/s]. Must be > 0.
    rho : float
        Surface relaxivity [m/s].
    B_0, Temp, fluid : float, float, str
        Reserved for potential post-processing of dimensional magnetization via mag_sat; they do
        not affect the non-dimensional PDE.
    t_0, t_f, dt : float
        Physical start time, final time, and time step [s]. Internal stepping is performed in τ
        using Δτ = dt / T2B. The returned `time_array` is in physical time units [s].
    volume_ : bool
        If True, the returned signal is the volume-weighted integral
            M̃(τ) = 4π ∫_0^{φ_R} μ(φ, τ) φ^2 dφ.
        If False, a normalized average is returned:
            M̃(τ) = (∫ μ φ^2 dφ) / (∫ φ^2 dφ).
    normalize : bool
        If True, the signal array is scaled so that M̃(t_0) = 1 (provided the initial signal ≠ 0).
    return_data : {'all', 'time-mag', 'mag_amounts', 'mag_assemble'}
        Controls what is returned:
            'all'          -> (time_array, mu_final, signal_array, signal_final[, snapshots])
            'time-mag'     -> (time_array, signal_array)
            'mag_amounts'  -> signal_array
            'mag_assemble' -> signal_final (last-time assembled signal)
    linear_solver : {'mumps', 'cg'}
        Linear solver choice. 'mumps' uses a direct LU solver; 'cg' uses a CG+AMG Krylov solver.
    progress : bool
        If True, a tqdm progress bar is used (if tqdm is installed).
    store_fields_every : int or None
        If not None and > 0, a list of snapshot Functions is returned, storing μ(φ,τ) every
        `store_fields_every` time steps (including step 0).
    V_degree : int
        Polynomial degree for the continuous Galerkin space.
    comm : MPI communicator or None
        MPI communicator for parallel runs. If None, MPI.comm_world is used.

    Returns
    -------
    Depends on `return_data`. For 'all':
        time_array : np.ndarray
            Physical time samples in seconds.
        mu_final : dolfin.Function
            Final non-dimensional magnetization field μ(φ, τ_f).
        signal_array : np.ndarray
            Time evolution of the non-dimensional volume-weighted signal M̃(τ).
        signal_final : float
            Final non-dimensional signal 4π ∫ μ(φ, τ_f) φ^2 dφ.
        snapshots : list of dolfin.Function (optional)
            Only if `store_fields_every` is not None.

    Notes
    -----
    The signal returned by this routine is **non-dimensional**. To reconstruct a dimensional
    magnetization M(t) one may multiply by m_s * (D*T2B)^{3/2}, where m_s is a characteristic
    magnetization (e.g. from mag_sat(B_0, Temp, fluid)).
    """
    # ------------------------
    # Input validation
    # ------------------------
    if radius <= 0.0:
        raise ValueError("`radius` must be strictly positive.")
    if diffusion <= 0.0:
        raise ValueError("`diffusion` must be strictly positive to define a diffusive length scale.")
    if T2B is None or T2B <= 0.0:
        raise ValueError("`T2B` must be strictly positive for the non-dimensional model.")
    if dt <= 0.0:
        raise ValueError("`dt` must be strictly positive.")
    if t_f <= t_0:
        raise ValueError("`t_f` must be greater than `t_0`.")
    if mesh_res < 2:
        raise ValueError("`mesh_res` must be at least 2 elements.")

    # MPI communicator
    if comm is None:
        comm = MPI.comm_world

    # ------------------------
    # Non-dimensional parameters
    # ------------------------
    # Non-dimensional radius φ_R = R / sqrt(D*T2B)
    phi_R = float(radius) / math.sqrt(float(diffusion) * float(T2B))

    # Non-dimensional surface relaxivity α = ρ sqrt(D*T2B) / D
    alpha_val = float(rho) * math.sqrt(float(diffusion) * float(T2B)) / float(diffusion)

    # Non-dimensional time step Δτ = dt / T2B
    dt_nd = float(dt) / float(T2B)

    # Non-dimensional initial and final times (not strictly needed for the solver,
    # but kept for completeness/documentation)
    tau_0 = float(t_0) / float(T2B)
    tau_f = float(t_f) / float(T2B)

    # ------------------------
    # 1D mesh in φ ∈ [0, φ_R]
    # ------------------------
    mesh = IntervalMesh(comm, mesh_res, 0.0, phi_R)

    # Optional mesh statistics (purely informational; uses physical parameters)
    if mesh_stats and callable(mesh_statistics):
        try:
            mesh_statistics(mesh, T2B, rho)
        except Exception as e:
            warning("mesh_statistics() raised and was ignored: %s" % str(e))

    # Mark outer boundary φ = φ_R with id = 1
    facet_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

    class Outer(SubDomain):
        def inside(self, x, on_boundary):
            # φ = x[0] in the non-dimensional domain
            return on_boundary and near(x[0], phi_R, DOLFIN_EPS)

    Outer().mark(facet_markers, 1)

    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh, subdomain_data=facet_markers)

    # ------------------------
    # Function space and fields
    # ------------------------
    V = FunctionSpace(mesh, "CG", int(V_degree))
    mu = TrialFunction(V)
    w = TestFunction(V)

    # Radial coordinate in non-dimensional space
    x = SpatialCoordinate(mesh)
    phi = x[0]
    phi2 = phi * phi

    # Initial condition: μ(φ, 0) = 1 (uniform normalized magnetization)
    mu_prev = Function(V)
    mu_prev.interpolate(Constant(1.0))

    # Physics and time coefficients (all are non-dimensional at this point)
    dtc = Constant(dt_nd)
    alpha_c = Constant(alpha_val)

    # ------------------------
    # Variational problem (non-dimensional weak form)
    # ------------------------
    # Weak form for:
    #   (μ^n - μ^{n-1})/Δτ - μ_{φφ}^n - (2/φ) μ_φ^n + μ^n = 0
    #
    # After multiplying by w, integrating over Ω and using Green's identity
    # (with Robin condition at φ=φ_R and symmetry at φ=0), we obtain:
    #
    #   ∫ μ^n w dφ
    # + Δτ ∫ (μ_φ^n)(w_φ) dφ
    # + Δτ α ∫_∂R μ^n w dS
    # + Δτ ∫ (2/φ) μ_φ^n w dφ
    # + Δτ ∫ μ^n w dφ
    # = ∫ μ^{n-1} w dφ.
    #
    Dmu_dphi = mu.dx(0)  # derivative ∂_φ μ in the 1D coordinate φ

    a = (mu * w * dx
         + dtc * dot(grad(mu), grad(w)) * dx
         + dtc * alpha_c * mu * w * ds(1)
         + dtc * (2.0 / phi) * Dmu_dphi * w * dx
         + dtc * mu * w * dx)

    L = mu_prev * w * dx

    # Assemble the system matrix once
    A = assemble(a)

    # Choose solver
    use_mumps = (str(linear_solver).lower() == "mumps")
    if use_mumps:
        solver = LUSolver(A, "mumps")
    else:
        # CG with algebraic multigrid preconditioner
        solver = KrylovSolver("cg", "hypre_amg")
        solver.set_operator(A)
        solver.parameters["relative_tolerance"] = 1e-10
        solver.parameters["absolute_tolerance"] = 0.0
        solver.parameters["maximum_iterations"] = 2000

    # ------------------------
    # Time discretization (physical time array)
    # ------------------------
    nt = int(np.floor((t_f - t_0) / dt)) + 1
    time_array = t_0 + np.arange(nt, dtype=float) * dt

    # Reference integral for volume-weighted averages: ∫ φ^2 dφ
    denom = assemble(phi2 * dx)

    # Storage for signal M̃(τ)
    mag_amounts = np.zeros(nt, dtype=float)

    # First sample at t_0
    mu_now = Function(V)
    if volume_:
        mag_amounts[0] = 4.0 * np.pi * assemble(mu_prev * phi2 * dx)
    else:
        mag_amounts[0] = assemble(mu_prev * phi2 * dx) / denom
    initial_signal = mag_amounts[0]

    # ------------------------
    # Progress bar support (optional)
    # ------------------------
    use_tqdm = False
    if progress:
        try:
            from tqdm import tqdm
            use_tqdm = True
        except Exception:
            use_tqdm = False

    iterator = range(1, nt)
    if use_tqdm:
        iterator = tqdm(iterator, desc="Time-stepping (ND BT)", unit="step",
                        total=nt - 1, leave=True)

    # Optional snapshots of the field μ(φ, τ)
    snapshots = [] if (store_fields_every is not None and store_fields_every > 0) else None
    if snapshots is not None and 0 % store_fields_every == 0:
        snapshots.append(mu_prev.copy(deepcopy=True))

    # ------------------------
    # Time loop in physical time (internally uses Δτ = dt/T2B)
    # ------------------------
    for i in iterator:
        # Right-hand side for the current step
        b = assemble(L)

        # Solve the linear system A mu_now = b
        if use_mumps:
            solver.solve(mu_now.vector(), b)
        else:
            solver.solve(mu_now.vector(), b)

        # Compute non-dimensional signal
        if volume_:
            mag_amounts[i] = 4.0 * np.pi * assemble(mu_now * phi2 * dx)
        else:
            mag_amounts[i] = assemble(mu_now * phi2 * dx) / denom

        # Advance in time: μ^{n-1} ← μ^n
        mu_prev.assign(mu_now)

        # Progress bar info
        if use_tqdm and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(t=time_array[i])

        # Optional snapshots
        if snapshots is not None and (i % store_fields_every == 0):
            snapshots.append(mu_now.copy(deepcopy=True))

    # Final assembly at t_f (non-dimensional volume signal)
    mag_assemble = 4.0 * np.pi * assemble(mu_now * phi2 * dx)

    # Optional normalization (scale so that initial signal is 1)
    if normalize:
        if abs(initial_signal) > 0.0:
            mag_amounts = mag_amounts / initial_signal

    # ------------------------
    # Return data according to user request
    # ------------------------
    if return_data == 'all':
        if snapshots is not None:
            return time_array, mu_now, mag_amounts, mag_assemble, snapshots
        else:
            return time_array, mu_now, mag_amounts, mag_assemble
    elif return_data == 'time-mag':
        return time_array, mag_amounts
    elif return_data == 'mag_assemble':
        return mag_assemble
    elif return_data == 'mag_amounts':
        return mag_amounts
    else:
        raise ValueError("Invalid `return_data` option in ND_BT_FEM.")


ND_BT_FEM.__version__ = __nd_version__
