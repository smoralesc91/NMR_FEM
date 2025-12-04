from dolfin import *
import numpy as np

__version__ = "1.5-spatial-profile"

# Optional user utilities
try:
    from Functions_NMR import mag_sat, mesh_statistics, T2star_conventional
except Exception:
    mag_sat = None
    mesh_statistics = None
    T2star_conventional = None

# Silence FEniCS logging (can be overridden by user)
set_log_level(LogLevel.ERROR)

# Global FEniCS parameters (legacy dolfin)
parameters["linear_algebra_backend"] = "PETSc"
parameters["form_compiler"]["representation"] = "uflacs"  # legacy-compatible
parameters["form_compiler"]["cpp_optimize"] = True


def FEM_NMR(
    radius=1.0,                      # [m] pore radius R > 0
    mesh_res=300,                    # number of intervals on [0, R]
    mesh_stats=False,                # call user mesh_statistics(mesh, T2B, rho) if available
    T2B=1.0,                         # [s] bulk transverse relaxation, set <=0 to disable
    diffusion=1.0,                   # [m^2/s] self/effective diffusion coefficient D >= 0
    rho=1.0,                         # [m/s] surface relaxivity (transverse)
    B_0=0.05, Temp=303.15, fluid='water',  # used only for initial magnetization via mag_sat()
    t_0=0.0, t_f=1.0, dt=1e-2,       # time window and step (uniform)
    volume_=True,                    # if True, return physical M(t) = 4π∫ m r^2 dr ; else weighted average
    normalize=False,                 # if True, scale output so that signal at t_0 equals 1
    return_data='all',               # 'all' | 'time-mag' | 'mag_amounts' | 'mag_assemble'
    linear_solver='mumps',           # 'mumps' | 'cg'
    progress=False,                   # use tqdm if available
    store_fields_every=None,         # None or integer N to store m(r,t) every N steps
    V_degree=2,                      # Lagrange degree for CG space
    comm=None                        # MPI communicator (None->MPI.comm_world; for serial use MPI.comm_self)
):
    """
    Finite-Element solver for the spherically symmetric Bloch–Torrey equation with surface relaxation:

        ∂t m = D (m_rr + (2/r) m_r) - m/T2B,   r ∈ (0, R)
        D m_r(R) + ρ m(R) = 0,                 (Robin at r=R)
        m_r(0) = 0                              (symmetry at r=0)

    Weak form weighted by r^2 (to avoid 1/r singularity near r=0):
        Find m^n ∈ V s.t.  ∫ ( m^n w r^2
                              + Δt D r^2 ∂_r m^n ∂_r w
                              + Δt (1/T2B) m^n w r^2 ) dr
                           + Δt ρ R^2 m^n(R) w(R)
                           = ∫ m^{n-1} w r^2 dr

    Signal definitions:
        - If volume_ is False: return ⟨m⟩_{r^2} = (∫ m r^2 dr) / (∫ r^2 dr)
        - If volume_ is True : return M(t) = 4π ∫ m r^2 dr
        - If normalize is True: scale signal by its initial value so that signal(t_0)=1

    Parameters
    ----------
    radius : float
        Pore radius R > 0 (meters).
    mesh_res : int
        Number of 1D elements; use >= 100 for smooth signals.
    mesh_stats : bool
        If True and user function available, prints mesh-related info.
    T2B : float
        Bulk transverse relaxation time [s]. If <= 0, bulk term disabled.
    diffusion : float
        Diffusion coefficient [m^2/s].
    rho : float
        Surface relaxivity [m/s].
    B_0, Temp, fluid : used only if mag_sat() is available to set m0.
    t_0, t_f, dt : floats
        Time window [t_0, t_f] with uniform step dt>0.
    volume_ : bool
        Toggle between physical M(t) and r^2-weighted average.
    normalize : bool
        Normalize output by initial signal value.
    return_data : str
        'all' -> (time_array, m_final, signal_array, M_final),
        'time-mag' -> (time_array, signal_array),
        'mag_amounts' -> signal_array,
        'mag_assemble' -> M_final (physical 4π∫ m r^2 dr at final time).
    linear_solver : str
        'mumps' (direct) or 'cg' (iterative with Hypre AMG).
    progress : bool
        Show a tqdm progress bar if installed.
    store_fields_every : int or None
        If int N>0, store a snapshot of m every N steps. Returned list in 'all'.
    V_degree : int
        Polynomial degree for the CG space.
    comm : MPI communicator
        Use MPI.comm_world for parallel runs; MPI.comm_self for forced serial.

    Returns
    -------
    Depends on `return_data` as described above. For 'all', if store_fields_every is not None,
    the tuple becomes (time_array, m_final, signal_array, M_final, snapshots_list).
    """
    # ------------------------
    # Input validation
    # ------------------------
    if radius <= 0.0:
        raise ValueError("`radius` must be > 0.")
    if mesh_res < 4:
        raise ValueError("`mesh_res` should be >= 4.")
    if dt <= 0.0:
        raise ValueError("`dt` must be > 0.")
    if t_f <= t_0:
        raise ValueError("Require t_f > t_0.")
    if diffusion < 0.0:
        raise ValueError("`diffusion` must be >= 0.")
    if rho < 0.0:
        raise ValueError("`rho` must be >= 0.")

    # ------------------------
    # MPI communicator
    # ------------------------
    if comm is None:
        comm = MPI.comm_world  # allow parallel runs by default

    # ------------------------
    # 1D mesh on [0, R]
    # ------------------------
    mesh = IntervalMesh(comm, mesh_res, 0.0, float(radius))

    # Optional user mesh stats
    if mesh_stats and callable(mesh_statistics):
        try:
            mesh_statistics(mesh, T2B, rho)
        except Exception as e:
            warning("mesh_statistics() raised and was ignored: %s" % str(e))

    # Mark outer boundary r=R with id=1
    facet_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
    class Outer(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], float(radius), DOLFIN_EPS)
    Outer().mark(facet_markers, 1)

    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh, subdomain_data=facet_markers)

    # ------------------------
    # Function space and fields
    # ------------------------
    V = FunctionSpace(mesh, "CG", int(V_degree))
    m = TrialFunction(V)
    w = TestFunction(V)

    # Radial coordinate and weights
    x = SpatialCoordinate(mesh)
    r = x[0]
    r2 = r*r
    R = Constant(float(radius))

    # Initial magnetization m0 (optional Curie-law saturation)
    try:
        m0_val = float(mag_sat(B_0, Temp, fluid)) if callable(mag_sat) else 1.0
    except Exception:
        m0_val = 1.0
    m_prev = Function(V)
    m_prev.interpolate(Constant(m0_val))

    # Physics and time coefficients
    D = Constant(float(diffusion))
    invT2B = Constant(0.0 if (T2B is None or T2B <= 0.0) else 1.0/float(T2B))
    dtc = Constant(float(dt))
    rho_c = Constant(float(rho))

    # ------------------------
    # Variational problem (r^2-weighted form)
    # ------------------------
    a = ( m*w*r2*dx
          + dtc*D*dot(grad(m), grad(w))*r2*dx
          + dtc*invT2B*m*w*r2*dx
          + dtc*rho_c*R*R*m*w*ds(1) )
    L = m_prev*w*r2*dx

    # Assemble system matrix once
    A = assemble(a)

    # Choose solver
    use_mumps = (str(linear_solver).lower() == "mumps")
    if use_mumps:
        solver = LUSolver(A, "mumps")
    else:
        solver = KrylovSolver("cg", "hypre_amg")
        solver.set_operator(A)
        # Reasonable PETSc KSP tolerances
        solver.parameters["relative_tolerance"] = 1e-10
        solver.parameters["absolute_tolerance"] = 0.0
        solver.parameters["maximum_iterations"] = 5000

    # ------------------------
    # Time discretization
    # ------------------------
    nt = int(np.floor((t_f - t_0)/dt)) + 1
    time_array = t_0 + np.arange(nt, dtype=float)*dt

    # Weighted integral denominator and initial signal
    denom = assemble(r2*dx)

    # Storage for signal
    mag_amounts = np.zeros(nt, dtype=float)

    # First sample at t_0
    m_now = Function(V)
    if volume_:
        mag_amounts[0] = 4.0*np.pi * assemble(m_prev*r2*dx)
    else:
        mag_amounts[0] = assemble(m_prev*r2*dx) / denom

    # Keep initial for normalization if requested
    initial_signal = float(mag_amounts[0])

    # Optional progress bar
    use_tqdm = False
    if progress:
        try:
            from tqdm import tqdm
            use_tqdm = True
        except Exception:
            use_tqdm = False

    iterator = range(1, nt)
    if use_tqdm:
        iterator = tqdm(iterator, desc="Time-stepping", unit="step", total=nt-1, leave=True)

    # Optional field snapshots
    snapshots = [] if (store_fields_every is not None and store_fields_every > 0) else None
    if snapshots is not None and 0 % store_fields_every == 0:
        snapshots.append(m_prev.copy(deepcopy=True))

    # ------------------------
    # Time loop
    # ------------------------
    b = None  # will reuse tensor for RHS
    for i in iterator:
        # Assemble RHS with updated m_prev (reuse tensor if possible)
        if b is None:
            b = assemble(L)
        else:
            assemble(L, tensor=b)

        # Solve the linear system
        if use_mumps:
            solver.solve(m_now.vector(), b)
        else:
            solver.solve(m_now.vector(), b)

        # Compute signal
        if volume_:
            mag_amounts[i] = 4.0*np.pi * assemble(m_now*r2*dx)
        else:
            mag_amounts[i] = assemble(m_now*r2*dx) / denom

        # Advance in time
        m_prev.assign(m_now)

        # Progress bar info
        if use_tqdm and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(t=time_array[i])

        # Optional snapshots
        if snapshots is not None and (i % store_fields_every == 0):
            snapshots.append(m_now.copy(deepcopy=True))

    # Final physical assembly at t_f (4π ∫ m r^2 dr)
    mag_assemble = 4.0*np.pi * assemble(m_now*r2*dx)

    # Optional normalization (scale so that signal at t_0 equals 1)
    if normalize:
        # Avoid division by zero if the initial signal is zero (degenerate case)
        if abs(initial_signal) > 0.0:
            mag_amounts = mag_amounts / initial_signal

    # ------------------------
    # Prepare returns
    # ------------------------
    if return_data == 'all':
        if snapshots is not None:
            return time_array, m_now, mag_amounts, mag_assemble, snapshots
        else:
            return time_array, m_now, mag_amounts, mag_assemble
    elif return_data == 'time-mag':
        return time_array, mag_amounts
    elif return_data == 'mag_assemble':
        return mag_assemble
    elif return_data == 'mag_amounts':
        return mag_amounts
    else:
        raise ValueError("Invalid `return_data` option.")



def profile_1d_dof(f):
    """
    Extracts and sorts 1D profile data from a dolfin.Function.

    This utility function retrieves the spatial coordinates and corresponding
    solution values from a FEniCS Function defined on a 1D (Interval) mesh.
    It accesses the coordinates of the degrees of freedom (DoFs) via
    `tabulate_dof_coordinates()` and the solution values via `vector().get_local()`.

    A critical step is sorting: the internal FEniCS ordering of DoFs
    (and thus the vectors) is not guaranteed to be spatially monotonic
    (e.g., from 0 to R). This function performs a sort operation based on the
    spatial coordinates (x) to ensure that the returned arrays are correctly
    ordered for plotting or spatial analysis.

    Parameters
    ----------
    f : dolfin.Function
        The 1D FEniCS function object (e.g., m_now, m_prev) from which
        to extract the profile data.

    Returns
    -------
    coords_sorted : numpy.ndarray
        A 1D array of the spatial coordinates (DoFs) sorted in
        ascending order.
    values_sorted : numpy.ndarray
        A 1D array of the corresponding function values, reordered
        to match `coords_sorted`.
    """
    V = f.function_space()
    # Get DoF coordinates (may be unsorted)
    x = V.tabulate_dof_coordinates().reshape((-1, 1))[:,0]
    # Get DoF values (in the same unsorted order as x)
    m = f.vector().get_local()
    
    # Get the indices that would sort the coordinate array
    idx = np.argsort(x)
    
    # Return both arrays, sorted by coordinate
    return x[idx], m[idx]

# Backward-compatibility alias
NMR_FEM = FEM_NMR
FEM_NMR.__version__ = __version__
NMR_FEM.__version__ = __version__