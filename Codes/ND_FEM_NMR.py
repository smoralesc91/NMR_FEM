from dolfin import *
import numpy as np
import math

__nd_version__ = "0.4_Rigorous_Analytic"

# Optional user utilities
try:
    from Functions_NMR import mag_sat, mesh_statistics
except Exception:
    mag_sat = None
    mesh_statistics = None

set_log_level(LogLevel.ERROR)

def ND_BT_FEM(
    radius=1.0,                      # [m] R
    mesh_res=300,                    # Mesh size
    mesh_stats=False,                # Optional stats
    T2B=1.0,                         # [s] Bulk T2
    diffusion=1.0,                   # [m^2/s] D
    rho=1.0,                         # [m/s] Surface relaxivity
    B_0=0.05, Temp=303.15, fluid='water',  # Unused in ND logic, kept for signature
    t_0=0.0, t_f=1.0, dt=1e-2,       # Time control
    volume_=False,                   # False = S(tau) (Eq. 52); True = M(tau) (Unnormalized)
    normalize=False,                 # Extra normalization check
    return_data='all',               # Return format
    linear_solver='mumps',           # Solver type
    progress=False,                  # Progress bar
    store_fields_every=None,         # Snapshots
    V_degree=2,                      # FEM degree
    comm=None                        # MPI
):
    r"""
    Finite-Element solver for the **non-dimensional** Bloch–Torrey equation.
    Strict implementation of the equations derived in the paper.

    **Governing Equation:**
        ∂_τ μ = ∇² μ - μ   (in spherical weighted form)

    **Signal Calculation (Eq. 52):**
        If volume_ is False:
            S(τ) = (3 / φ_R³) * ∫ μ φ² dφ
        
        This uses the EXACT analytical volume of the sphere for normalization,
        not the numerical approximation from the mesh.

    Parameters
    ----------
    (Matches original signature exactly)
    """
    
    # ------------------------
    # 1. Input Validation
    # ------------------------
    if radius <= 0.0 or diffusion <= 0.0 or T2B <= 0.0 or dt <= 0.0:
        raise ValueError("Physical parameters must be strictly positive.")
    if t_f <= t_0:
        raise ValueError("t_f must be > t_0.")
    if comm is None:
        comm = MPI.comm_world

    # ------------------------
    # 2. Exact Dimensionless Scaling (Eq. 50)
    # ------------------------
    # φ_R = R / sqrt(D*T2B)
    phi_R = float(radius) / math.sqrt(float(diffusion) * float(T2B))

    # α = ρ * sqrt(T2B / D)
    alpha_val = float(rho) * math.sqrt(float(T2B) / float(diffusion))

    dt_nd = float(dt) / float(T2B)

    # ------------------------
    # 3. Mesh & Measures
    # ------------------------
    mesh = IntervalMesh(comm, mesh_res, 0.0, phi_R)

    # Mark boundary φ = φ_R
    facet_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    class Outer(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], phi_R, DOLFIN_EPS)
    Outer().mark(facet_markers, 1)

    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh, subdomain_data=facet_markers)

    # ------------------------
    # 4. Variational Setup (Eq. 53)
    # ------------------------
    V = FunctionSpace(mesh, "CG", int(V_degree))
    mu = TrialFunction(V)
    w = TestFunction(V)
    mu_prev = Function(V)
    
    # Initial Condition: μ(φ, 0) = 1
    mu_prev.interpolate(Constant(1.0))

    # Geometric Weight: φ²
    x = SpatialCoordinate(mesh)
    phi = x[0]
    phi2 = phi * phi

    dtc = Constant(dt_nd)
    alpha_c = Constant(alpha_val)
    phi_R_sq = Constant(phi_R**2) 

    # Weighted Weak Form (Exact match to paper Eq. 53)
    a = (mu * w * phi2 * dx
         + dtc * dot(grad(mu), grad(w)) * phi2 * dx
         + dtc * mu * w * phi2 * dx
         + dtc * alpha_c * mu * w * phi_R_sq * ds(1))

    L = mu_prev * w * phi2 * dx

    A = assemble(a)

    if str(linear_solver).lower() == "mumps":
        solver = LUSolver(A, "mumps")
    else:
        solver = KrylovSolver("cg", "hypre_amg")
        solver.set_operator(A)
        solver.parameters["relative_tolerance"] = 1e-12
        solver.parameters["absolute_tolerance"] = 1e-14

    # ------------------------
    # 5. Signal Factors (Strict Implementation)
    # ------------------------
    # We define the geometric factors analytically to avoid mesh errors.
    
    # Factor for Eq. 52: S(τ) = (3/φ_R³) * Integral
    factor_normalized = 3.0 / (phi_R**3)
    
    # Factor for Unnormalized M(τ): 4π * Integral
    # (Matches M(t) definition in Section 4.3 of paper)
    factor_volume = 4.0 * np.pi

    # ------------------------
    # 6. Time Loop
    # ------------------------
    nt = int(np.floor((t_f - t_0) / dt)) + 1
    time_array = t_0 + np.arange(nt, dtype=float) * dt
    mag_amounts = np.zeros(nt, dtype=float)
    mu_now = Function(V)

    # --- Initial Step (t=0) ---
    # Compute integral of initial condition numerically
    integral_0 = assemble(mu_prev * phi2 * dx)
    
    if volume_:
        mag_amounts[0] = integral_0 * factor_volume
    else:
        # STRICT EQ. 52 IMPLEMENTATION:
        # Use analytical factor, not numerical denominator
        mag_amounts[0] = integral_0 * factor_normalized
        
    initial_signal = mag_amounts[0]

    # Progress bar setup
    iterator = range(1, nt)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="FEM Solver", unit="step", total=nt-1)
        except ImportError: pass
    
    # Snapshots setup
    snapshots = [] if (store_fields_every is not None and store_fields_every > 0) else None
    if snapshots is not None and 0 % store_fields_every == 0:
        snapshots.append(mu_prev.copy(deepcopy=True))

    # --- Stepping ---
    for i in iterator:
        b = assemble(L)
        solver.solve(mu_now.vector(), b)
        
        # Compute Integral: ∫ μ φ² dφ
        integral_now = assemble(mu_now * phi2 * dx)
        
        # Apply strict geometric definition
        if volume_:
            mag_amounts[i] = integral_now * factor_volume
        else:
            mag_amounts[i] = integral_now * factor_normalized
            
        mu_prev.assign(mu_now)
        
        if snapshots is not None and (i % store_fields_every == 0):
            snapshots.append(mu_now.copy(deepcopy=True))

    # Final assembly
    val_final_integral = assemble(mu_now * phi2 * dx)
    mag_assemble = val_final_integral * factor_volume # Always return 'M' here per signature

    # ------------------------
    # 7. Final Check & Return
    # ------------------------
    # Even with analytical factors, FEM has tiny interpolation errors.
    # 'normalize=True' is kept as a hard clamp if the user requests it.
    if normalize:
        if abs(initial_signal) > 1e-14:
            mag_amounts = mag_amounts / initial_signal

    if return_data == 'all':
        return time_array, mu_now, mag_amounts, mag_assemble, snapshots if snapshots else None
    elif return_data == 'time-mag':
        return time_array, mag_amounts
    elif return_data == 'mag_assemble':
        return mag_assemble
    elif return_data == 'mag_amounts':
        return mag_amounts
    else:
        raise ValueError("Invalid `return_data` option.")

ND_BT_FEM.__version__ = __nd_version__
