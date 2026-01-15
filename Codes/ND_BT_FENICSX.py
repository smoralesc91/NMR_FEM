# ==============================================================================
# 1. ENVIRONMENT SETUP
# ==============================================================================
import numpy as np
import time
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector


def BT_fenicsx_decay(R_phys=100.0e-6,
                     rho_phys=40.0e-6,
                     D_phys=2.30e-9,
                     T2B_phys=2.0,
                     t_start=0.0,
                     t_final=10.0,
                     dt_phys=0.01,
                     NUM_ELEMS=100,
                     DEGREE=2,
                     SOLVER_TYPE='mumps',
                     comm=MPI.COMM_WORLD,
                     verbose=False,
                     export_files=False):
    """
    Solve the transverse magnetization decay in spherical pores using the 
    dimensionless Bloch-Torrey equation via finite element method (FEM).
    
    This function implements a 1D radial model for NMR T2 relaxation in
    spherical pores with surface relaxation. The solution is obtained by
    solving the diffusion-reaction equation with Robin boundary conditions.
    
    Parameters
    ----------
    R_phys : float, optional
        Physical pore radius in meters. Default is 100.0e-6 (100 μm).
    rho_phys : float, optional
        Surface relaxivity in meters per second. Default is 40.0e-6 (40 μm/s).
    D_phys : float, optional
        Bulk diffusion coefficient in m²/s. Default is 2.30e-9 (water at 25°C).
    T2B_phys : float, optional
        Bulk relaxation time in seconds. Default is 2.0.
    t_start : float, optional
        Simulation start time in seconds. Default is 0.0.
    t_final : float, optional
        Simulation end time in seconds. Default is 10.0.
    dt_phys : float, optional
        Physical time step in seconds. Default is 0.01.
    NUM_ELEMS : int, optional
        Number of elements in the 1D radial mesh. Default is 100.
    DEGREE : int, optional
        Polynomial degree of the finite element basis functions. Default is 2.
    SOLVER_TYPE : str, optional
        Linear solver type. Options: 'mumps', 'umfpack', 'superlu', or 'default'.
        Default is 'mumps'.
    comm : MPI.Comm, optional
        MPI communicator for parallel execution. Default is MPI.COMM_WORLD.
    verbose : bool, optional
        If True, print simulation progress to console. Default is False.
    export_files : bool, optional
        If True, export the time and signal arrays to a text file. Default is False.
    
    Returns
    -------
    time_array : ndarray or None
        Array of physical time points in seconds. On rank 0, this is an ndarray.
        On other MPI ranks, returns None.
    signal_array : ndarray or None
        Array of normalized magnetization signal values (M(t)/M(0)). On rank 0,
        this is an ndarray. On other MPI ranks, returns None.
    
    Notes
    -----
    The dimensionless Bloch-Torrey equation in spherical coordinates is solved:
        ∂M/∂τ - ∇²M + M = 0
    with Robin boundary condition at the pore wall (r = φ_R):
        ∂M/∂n + αM = 0
    and initial condition M(r,0) = 1.
    
    Dimensionless parameters:
        φ_R = R / √(D·T2B)   (dimensionless radius)
        α   = ρ·√(T2B/D)     (dimensionless rho)
        τ   = t / T2B        (dimensionless time)
    
    The finite element formulation includes the spherical Jacobian (r²) in all
    volume integrals. The time discretization uses the implicit backward Euler
    method. The magnetization signal is computed by volume integration of the
    magnetization and normalized by its initial value.
    
    The function is optimized for Bayesian inference, with minimal console output
    and file I/O unless explicitly requested.

    **Algorithmic Optimizations:**

    1.  **Vectorized Integration (The "Dot Product" Trick):**
        Standard FEM integration ``assemble(u * dx)`` is computationally expensive 
        inside a loop ($O(N_{cells})$). This implementation pre-calculates a 
        static weight vector $\mathbf{w}$ representing the geometric integral 
        of the basis functions:
        $$ w_i = \int_{\Omega} \phi_i(x) r^2 dr $$
        The total magnetization at step $n$ is then computed via a BLAS-1 
        dot product, reducing complexity to $O(N_{dofs})$:
        $$ M(t) = \mathbf{w} \cdot \mathbf{u}_n $$

    2.  **LU Factorization Reuse:**
        For a constant time step $\Delta t$, the system stiffness matrix $\mathbf{A}$ 
        is time-invariant. The LU factorization is computed once during initialization 
        and reused via backward substitution for all subsequent steps.

    3.  **Static Memory Allocation:**
        NumPy arrays for output data are pre-allocated to prevent dynamic resizing 
        overhead. PETSc vectors are updated in-place.

    4.  **Pointer Localization:**
        Critical PETSc vector handles are dereferenced outside the time loop to 
        bypass Python attribute lookup latency.
    """
    # ==========================================================================
    # 1. PHYSICS & SCALING (Dimensionless Groups)
    # ==========================================================================
    diff_length = np.sqrt(D_phys * T2B_phys)
    PHI_R_VAL   = R_phys / diff_length
    ALPHA_VAL   = rho_phys * np.sqrt(T2B_phys / D_phys)
    
    # Time discretization (Dimensionless)
    TAU_START = t_start / T2B_phys
    TAU_FINAL = t_final / T2B_phys
    DT_VAL    = dt_phys / T2B_phys

    # ==========================================================================
    # 2. MESH GENERATION & FUNCTION SPACES
    # ==========================================================================
    # 1D Interval Mesh scaled to dimensionless radius [0, PHI_R]
    domain = mesh.create_unit_interval(comm, NUM_ELEMS)
    domain.geometry.x[:, 0] *= PHI_R_VAL
    
    # Function Space: Continuous Galerkin (Lagrange)
    V = fem.functionspace(domain, ("CG", DEGREE))
    
    # Boundary Marking (Surface of the sphere r = PHI_R)
    fdim = domain.topology.dim - 1
    def is_pore_wall(x):
        return np.isclose(x[0], PHI_R_VAL, rtol=1e-12)
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, is_pore_wall)
    boundary_tags = mesh.meshtags(domain, fdim, boundary_facets, 
                                  np.full_like(boundary_facets, 1))
    
    # Measures for integration
    dx = ufl.dx
    ds = ufl.Measure("ds", domain=domain, subdomain_data=boundary_tags)
    
    # ==========================================================================
    # 3. VARIATIONAL FORMULATION (Pre-Compiled)
    # ==========================================================================
    # Trial (u) and Test (v) functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Solution containers (Previous and Current steps)
    u_n    = fem.Function(V)
    u_curr = fem.Function(V)
    
    # Initial Condition: Uniform magnetization M(r,0) = 1.0
    u_n.interpolate(lambda x: np.ones_like(x[0]))
    
    # UFL Constants (Avoids recompilation if parameters change)
    dt    = fem.Constant(domain, PETSc.ScalarType(DT_VAL))
    alpha = fem.Constant(domain, PETSc.ScalarType(ALPHA_VAL))
    phi_R = fem.Constant(domain, PETSc.ScalarType(PHI_R_VAL))
    
    # Spherical Coordinate System (Jacobian r^2 term)
    x = ufl.SpatialCoordinate(domain)
    phi = x[0] # Radial coordinate r
    
    # --- Bilinear Form (LHS) ---
    # Time-independent matrix 'A' for Backward Euler scheme:
    # (u - dt*Laplacian(u) + dt*u)*v * r^2
    a = (u * v * phi**2 * dx +
         dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * phi**2 * dx +
         dt * u * v * phi**2 * dx +
         dt * alpha * phi_R**2 * u * v * ds(1))
    
    # --- Linear Form (RHS) ---
    # Time-dependent vector 'b': u_n * v * r^2
    L = u_n * v * phi**2 * dx
    
    # Compile UFL forms
    bilinear_form = fem.form(a)
    linear_form   = fem.form(L)
    
    # ==========================================================================
    # 4. ALGEBRAIC SETUP & SOLVER CONFIGURATION
    # ==========================================================================
    # A. Assemble Stiffness Matrix (Done ONCE)
    A = assemble_matrix(bilinear_form)
    A.assemble()
    
    # B. Create RHS Vector
    # Uses matrix structure to determine compatible vector layout (Robust API)
    b = A.createVecLeft()
    
    # C. Configure Linear Solver (KSP)
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY) # Direct solver mode
    solver.getPC().setType(PETSc.PC.Type.LU)
    
    # Set factorization package based on availability
    if SOLVER_TYPE == 'mumps':
        try:
            solver.getPC().setFactorSolverType("mumps")
        except PETSc.Error:
            if comm.rank == 0: print("WARNING: MUMPS not available, using default LU.")
    elif SOLVER_TYPE == 'superlu':
        solver.getPC().setFactorSolverType("superlu_dist")
    elif SOLVER_TYPE == 'umfpack':
        solver.getPC().setFactorSolverType("umfpack")
    
    solver.setFromOptions()
    
    # ==========================================================================
    # 5. OPTIMIZATION: PRE-CALCULATED WEIGHT VECTOR
    # ==========================================================================
    # Instead of assembling scalar integrals at every step, we compute a 
    # geometric weight vector. The integral becomes a dot product.
    # W_i = Integral(v_i * r^2 * dr)
    weight_form = fem.form(v * phi**2 * dx)
    weight_vec  = assemble_vector(weight_form) 
    
    # Finalize assembly in case of parallel distribution
    weight_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    
    # Analytical normalization factor for spherical volume (3/R^3)
    norm_factor = 3.0 / (PHI_R_VAL**3)
    
    # ==========================================================================
    # 6. TIME STEPPING LOOP (HIGH PERFORMANCE)
    # ==========================================================================
    duration_tau = TAU_FINAL - TAU_START
    steps = int(np.round(duration_tau / DT_VAL))
    
    # Static Memory Allocation (NumPy) for results
    time_array = np.zeros(steps + 1, dtype=np.float64)
    signal_raw = np.zeros(steps + 1, dtype=np.float64)
    
    # Initial State (t=0)
    # Note: u_n.x.petsc_vec contains the initial condition values
    signal_0 = norm_factor * weight_vec.dot(u_n.x.petsc_vec)
    
    time_array[0] = t_start
    signal_raw[0] = signal_0
    
    my_rank = comm.rank
    if verbose and my_rank == 0:
        start_time = time.perf_counter()
        print(f"-> Starting Simulation: {steps} steps")
    
    # --- Localizing references for speed (Avoid attribute lookups in loop) ---
    u_curr_vec = u_curr.x.petsc_vec
    u_n_array = u_n.x.array
    u_curr_array = u_curr.x.array
    
    # --- MAIN LOOP ---
    for step in range(steps):
        # 1. Update RHS Vector (b)
        #    Reset vector and re-assemble linear form (fast op)
        with b.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b, linear_form)
        
        # 2. Solve Linear System
        #    Uses pre-factored LU decomposition. Writes result to u_curr_vec.
        solver.solve(b, u_curr_vec)
        u_curr.x.scatter_forward() # Sync ghost values
        
        # 3. Compute Magnetization (Instantaneous Dot Product)
        #    Replaces expensive fem.assemble_scalar()
        vol_int = weight_vec.dot(u_curr_vec)
        
        # 4. Store Results (Direct Array Access)
        if my_rank == 0:
            signal_raw[step+1] = norm_factor * vol_int
            time_array[step+1] = (TAU_START + (step + 1) * DT_VAL) * T2B_phys
            
        # 5. Update Previous Step (Fast Memory Copy)
        u_n_array[:] = u_curr_array[:]

    # ==========================================================================
    # 7. FINALIZATION & OUTPUT
    # ==========================================================================
    if verbose and my_rank == 0:
        end_time = time.perf_counter()
        print(f"-> Simulation completed in {end_time - start_time:.4f} s")

    if my_rank == 0:
        # Signal Normalization M(t)/M(0)
        if signal_raw[0] > 1e-14:
            signal_norm = signal_raw / signal_raw[0]
        else:
            signal_norm = signal_raw

        if export_files:
            np.savetxt("nmr_decay.txt",
                       np.column_stack((time_array, signal_norm)),
                       header="Time_s Signal_Norm",
                       comments='')
            
        return time_array, signal_norm

    return None, None