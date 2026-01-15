# ==============================================================================
# 1. ENVIRONMENT SETUP
# ==============================================================================
import dolfinx
import numpy as np
import ufl
import time
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc
import warnings

def BT_fenicsx_decay_profile(R_phys=100.0e-6,
               rho_phys=40.0e-6,
               D_phys=2.30e-9,
               T2B_phys=2.0,
               t_start=0.0,
               t_final=10.0,
               dt_phys=0.01,
               PROFILE_SAVE_INTERVAL=None,
               NUM_ELEMS=100,
               DEGREE=2,
               SOLVER_TYPE='mumps',
               comm= MPI.COMM_WORLD,
               verbose=True,
               export_files=True,
               normalize_initial=True):
    """
    Solve the transverse magnetization decay in spherical pores using the 
    dimensionless Bloch-Torrey equation via finite element method (FEM).
    
    Implements the diffusion-reaction equation with Robin boundary condition
    for spherical symmetry in 1D radial coordinates. The model describes
    NMR T2 relaxation in porous media with surface relaxation at pore walls.
    
    Parameters
    ----------
    R_phys : float, optional
        Physical pore radius [m]. Default: 100.0e-6 (100 μm).
    rho_phys : float, optional
        Surface relaxivity [m/s]. Default: 40.0e-6 (40 μm/s).
    D_phys : float, optional
        Bulk diffusion coefficient [m²/s]. Default: 2.30e-9 (water at 25°C).
    T2B_phys : float, optional
        Bulk relaxation time [s]. Default: 2.0.
    t_start : float, optional
        Simulation start time [s]. Default: 0.0.
    t_final : float, optional
        Simulation end time [s]. Default: 10.0.
    dt_phys : float, optional
        Physical time step [s]. Default: 0.01.
    PROFILE_SAVE_INTERVAL : int or None, optional
        Spatial profile saving interval (time steps). If None or 0, profiles 
        are disabled. Default: None.
    NUM_ELEMS : int, optional
        Number of 1D radial elements. Default: 100.
    DEGREE : int, optional
        Polynomial degree for finite element basis. Default: 2 (quadratic).
    SOLVER_TYPE : str, optional
        Linear solver type: 'mumps', 'umfpack', 'superlu', or 'default'.
        Default: 'mumps'.
    comm : MPI.Comm, optional
        MPI communicator for parallel execution. Default: MPI.COMM_WORLD.
    verbose : bool, optional
        Enable console output for simulation progress. Default: True.
    export_files : bool, optional
        Export results to text files. Default: True.
    normalize_initial : bool, optional
        If True, normalizes the initial signal to exactly 1.0 by computing
        the numerical volume integral. Recommended for comparison with
        analytical solutions and Bayesian inference. Default: True.
    
    Returns
    -------
    dict
        Dictionary containing simulation results with keys:
        - 'time': Physical time array [s]
        - 'tau': Dimensionless time array
        - 'signal': Normalized magnetization signal
        - 'profiles': (Optional) List of (time, profile) tuples if profiling enabled
        - 'r_dimless': (Optional) Radial coordinates in dimensionless units
        - 'r_norm': (Optional) Radial coordinates normalized to pore radius
        - 'normalization_factor': Numerical normalization factor used
    
    Notes
    -----
    Dimensionless formulation:
    - Characteristic length: L = √(D·T2B)
    - Dimensionless radius: φ_R = R / L
    - Dimensionless rho: α = ρ·√(T2B/D)
    - Dimensionless time: τ = t / T2B
    
    Variational formulation includes spherical Jacobian (r²).
    Robin boundary condition: ∂u/∂n + α·u = 0 at r = R.
    
    The backward Euler scheme ensures unconditional stability.
    Spatial profiles are saved at intervals specified by PROFILE_SAVE_INTERVAL.
    
    Example
    -------
    >>> results = BT_fenicsx(R_phys=50e-6, T2B_phys=1.0, verbose=False)
    >>> signal = results['signal']
    >>> time = results['time']
    
    For Bayesian inference workflows, set verbose=False and export_files=False
    to minimize I/O overhead during MCMC sampling.
    """
    
    # ==========================================================================
    # 1. LINEAR SOLVER CONFIGURATION
    # ==========================================================================
    SOLVER_TYPE = SOLVER_TYPE

    def get_solver_options(solver_type):
        """
        Configure PETSc solver options for direct linear solvers.
        
        Parameters
        ----------
        solver_type : str
            Solver backend specification.
            
        Returns
        -------
        dict
            PETSc options dictionary.
        """
        opts = {"ksp_type": "preonly", "pc_type": "lu"}

        if solver_type == 'mumps':
            opts["pc_factor_mat_solver_type"] = "mumps"
        elif solver_type == 'umfpack':
            opts["pc_factor_mat_solver_type"] = "umfpack"
        elif solver_type == 'superlu':
            opts["pc_factor_mat_solver_type"] = "superlu_dist"
        # Default: PETSc internal LU factorization

        return opts

    # ==========================================================================
    # 2. PHYSICAL PARAMETERS
    # ==========================================================================
    # Pore system characteristics
    R_phys = R_phys      # [m] Pore radius
    rho_phys = rho_phys  # [m/s] Surface relaxivity
    D_phys = D_phys      # [m²/s] Diffusion coefficient
    T2B_phys = T2B_phys  # [s] Bulk relaxation time

    # Temporal domain specification
    t_start = t_start    # [s] Initial simulation time
    t_final = t_final    # [s] Final simulation time
    dt_phys = dt_phys    # [s] Time step size

    # Spatial profile output control
    PROFILE_SAVE_INTERVAL = PROFILE_SAVE_INTERVAL

    # FEM discretization parameters
    NUM_ELEMS = NUM_ELEMS  # Number of radial elements
    DEGREE = DEGREE        # Polynomial degree of basis functions

    # ==========================================================================
    # 3. DIMENSIONLESS SCALING
    # ==========================================================================
    # Characteristic diffusion length scale
    diff_length = np.sqrt(D_phys * T2B_phys)

    # Dimensionless parameters
    PHI_R_VAL = R_phys / diff_length                  # Dimensionless radius
    ALPHA_VAL = rho_phys * np.sqrt(T2B_phys / D_phys)  # Damköhler number

    # Dimensionless time parameters
    TAU_START = t_start / T2B_phys
    TAU_FINAL = t_final / T2B_phys
    DT_VAL = dt_phys / T2B_phys

    comm = comm

    # ==========================================================================
    # 4. FINITE ELEMENT METHOD SETUP (1D SPHERICAL COORDINATES)
    # ==========================================================================
    # Create 1D domain [0, φ_R] representing radial coordinate
    domain = mesh.create_unit_interval(comm, NUM_ELEMS)
    domain.geometry.x[:, 0] *= PHI_R_VAL

    # Continuous Galerkin function space
    V = fem.functionspace(domain, ("CG", DEGREE))

    # Identify pore wall boundary at r = φ_R
    def is_pore_wall(x):
        return np.isclose(x[0], PHI_R_VAL, rtol=1e-10)

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, is_pore_wall)
    boundary_tags = mesh.meshtags(domain, fdim, boundary_facets,
                                  np.full_like(boundary_facets, 1))

    # Integration measures
    dx = ufl.dx
    ds = ufl.Measure("ds", domain=domain, subdomain_data=boundary_tags)

    # Function space placeholders
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    u_n, u_curr = fem.Function(V), fem.Function(V)

    # Physical parameters as PETSc constants
    dt = fem.Constant(domain, PETSc.ScalarType(DT_VAL))
    alpha = fem.Constant(domain, PETSc.ScalarType(ALPHA_VAL))
    phi_R = fem.Constant(domain, PETSc.ScalarType(PHI_R_VAL))
    x = ufl.SpatialCoordinate(domain)
    phi = x[0]  # Radial coordinate

    # Initial condition: uniform magnetization M(r,0) = 1
    u_n.interpolate(lambda x: np.ones_like(x[0]))

    # ==========================================================================
    # 5. SPATIAL PROFILE EXTRACTION SETUP (CONDITIONAL)
    # ==========================================================================
    EXTRACT_PROFILES = False
    if PROFILE_SAVE_INTERVAL and PROFILE_SAVE_INTERVAL > 0:
        EXTRACT_PROFILES = True

    saved_profiles = []
    x_sorted = None
    r_normalized = None
    idx_sorted = None

    if EXTRACT_PROFILES:
        # Extract and sort DOF coordinates for consistent output
        x_dofs = V.tabulate_dof_coordinates()[:, 0]
        idx_sorted = np.argsort(x_dofs)
        x_sorted = x_dofs[idx_sorted]
        r_normalized = x_sorted / PHI_R_VAL

        # Save initial magnetization profile
        u_vals_init = u_n.x.array.copy()
        saved_profiles.append((t_start, u_vals_init[idx_sorted]))

    # ==========================================================================
    # 6. VOLUME NORMALIZATION FACTOR
    # ==========================================================================
    if normalize_initial:
        # Compute sphere volume using numerical integration (consistent with FEM)
        one_func = fem.Function(V)
        one_func.interpolate(lambda x: np.ones_like(x[0]))
        vol_numerical = fem.assemble_scalar(fem.form(one_func * phi**2 * dx))
        vol_numerical = comm.allreduce(vol_numerical, op=MPI.SUM)
        norm_factor = 1.0 / vol_numerical
        
        if verbose and comm.rank == 0:
            # Compare with analytical volume for validation
            vol_analytic = PHI_R_VAL**3 / 3.0
            rel_error = abs(vol_numerical - vol_analytic) / vol_analytic
            print(f"Normalization: Numerical volume integration")
            print(f"  Numerical volume: {vol_numerical:.10f}")
            print(f"  Analytical volume: {vol_analytic:.10f}")
            print(f"  Relative error: {rel_error:.2e}")
    else:
        # Original analytical normalization
        norm_factor = 3.0 / (PHI_R_VAL**3)
        if verbose and comm.rank == 0:
            print(f"Normalization: Analytical factor 3/R³ = {norm_factor:.6f}")

    # ==========================================================================
    # 7. VARIATIONAL FORMULATION (BACKWARD EULER TIME DISCRETIZATION)
    # ==========================================================================
    # Dimensionless Bloch-Torrey equation in spherical coordinates:
    # ∂M/∂τ - ∇²M + M = 0
    # Robin BC: ∂M/∂n + αM = 0 at r = φ_R
    # Spherical Jacobian: r² included in all volume integrals

    F_lhs = (u * v * phi**2 * dx +
             dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * phi**2 * dx +
             dt * u * v * phi**2 * dx +
             dt * alpha * phi_R**2 * u * v * ds(1))

    F_rhs = u_n * v * phi**2 * dx

    # Configure linear solver
    solver_options = get_solver_options(SOLVER_TYPE)
    problem = LinearProblem(F_lhs, F_rhs,
                            petsc_options_prefix="solver",
                            petsc_options=solver_options)

    # ==========================================================================
    # 8. TIME INTEGRATION LOOP
    # ==========================================================================
    duration_tau = TAU_FINAL - TAU_START
    steps = int(np.round(duration_tau / DT_VAL))

    results_time = []
    results_tau = []
    results_signal = []

    # Compute initial signal (integrated magnetization)
    vol_int_0 = fem.assemble_scalar(fem.form(u_n * phi**2 * dx))
    vol_int_0 = comm.allreduce(vol_int_0, op=MPI.SUM)
    signal_0 = norm_factor * vol_int_0

    # Verify initial normalization
    if comm.rank == 0 and abs(signal_0 - 1.0) > 1e-10:
        warnings.warn(f"Initial signal = {signal_0:.10f}, not 1.0. "
                     f"Difference: {signal_0-1.0:.2e}")

    results_time.append(t_start)
    results_tau.append(TAU_START)
    results_signal.append(signal_0)

    if verbose and comm.rank == 0:
        print("=" * 75)
        print(f"SIMULATION STARTED | Steps: {steps} | Element: P{DEGREE} "
              f"| Mesh: {NUM_ELEMS}")
        print(f"Solver: {SOLVER_TYPE.upper()} (via PETSc)")
        print(f"Physical Parameters:")
        print(f"  R    = {R_phys*1e6:.2f} μm")
        print(f"  ρ    = {rho_phys*1e6:.2f} μm/s")
        print(f"  D    = {D_phys*1e12:.2f} μm²/s")
        print(f"  T2B  = {T2B_phys:.2f} s")
        print(f"Dimensionless Parameters:")
        print(f"  φ_R  = R/√(D·T2B) = {PHI_R_VAL:.4f}")
        print(f"  α    = ρ√(T2B/D) = {ALPHA_VAL:.4f}")
        print(f"Time Configuration:")
        print(f"  t ∈ [{t_start:.2f}, {t_final:.2f}] s,  Δt = {dt_phys:.4f}")
        print(f"  τ ∈ [{TAU_START:.2f}, {TAU_FINAL:.2f}], Δτ = {DT_VAL:.4f}")

        if EXTRACT_PROFILES:
            print(f"Profile Mode: ENABLED (save every {PROFILE_SAVE_INTERVAL} steps)")
        else:
            print(f"Profile Mode: DISABLED (signal decay only)")
        print("=" * 75)
        start_time = time.perf_counter()

    # Main time stepping loop
    for step in range(steps):
        # Solve linear system for current time step
        u_curr = problem.solve()

        # Update dimensionless and physical time
        curr_tau = TAU_START + (step + 1) * DT_VAL
        curr_phys_time = curr_tau * T2B_phys

        # Compute normalized magnetization signal
        vol_int = fem.assemble_scalar(fem.form(u_curr * phi**2 * dx))
        vol_int = comm.allreduce(vol_int, op=MPI.SUM)
        signal = norm_factor * vol_int

        if comm.rank == 0:
            results_time.append(curr_phys_time)
            results_tau.append(curr_tau)
            results_signal.append(signal)

            # Conditional spatial profile saving
            if EXTRACT_PROFILES:
                is_last = (step + 1) == steps
                is_interval = ((step + 1) % PROFILE_SAVE_INTERVAL == 0)

                if is_last or is_interval:
                    u_snapshot = u_curr.x.array.copy()
                    saved_profiles.append((curr_phys_time,
                                          u_snapshot[idx_sorted]))

        # Update solution for next time step
        u_n.x.array[:] = u_curr.x.array[:]

    if verbose and comm.rank == 0:
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"\nSimulation completed in {elapsed:.2f} seconds "
              f"({elapsed/steps:.4f} s/step)")

    # ==========================================================================
    # 9. DATA EXPORT (CONDITIONAL)
    # ==========================================================================
    if export_files and comm.rank == 0:
        # Export physical time decay curve
        np.savetxt("nmr_decay_physical.txt",
                   np.column_stack((results_time, results_signal)),
                   header="Time_s Signal_Normalized", comments='')

        # Export dimensionless time decay curve
        np.savetxt("nmr_decay_dimensionless.txt",
                   np.column_stack((results_tau, results_signal)),
                   header="Tau_dimless Signal_Normalized", comments='')

        # Export spatial profiles if enabled
        if EXTRACT_PROFILES:
            header_str = "r_dimless r_norm " + \
                         " ".join([f"M_t={t:.4f}" for t, _ in saved_profiles])
            profile_data = [x_sorted, r_normalized] + \
                          [prof for _, prof in saved_profiles]
            export_matrix = np.column_stack(profile_data)

            np.savetxt("magnetization_profiles_evolution.txt",
                       export_matrix, header=header_str)
            if verbose:
                print("Data Export: Signal decay + spatial profiles saved.")
        elif verbose:
            print("Data Export: Signal decay saved (profiles disabled).")

    # ==========================================================================
    # 10. RESULTS ASSEMBLY
    # ==========================================================================
    results = {
        'time': np.array(results_time),
        'tau': np.array(results_tau),
        'signal': np.array(results_signal),
        'normalization_factor': norm_factor
    }

    if EXTRACT_PROFILES:
        results['profiles'] = saved_profiles
        results['r_dimless'] = x_sorted
        results['r_norm'] = r_normalized
    
    # Store dimensionless parameters for reference
    results['dimensionless_params'] = {
        'phi_R': PHI_R_VAL,
        'alpha': ALPHA_VAL,
        'dt_dimless': DT_VAL
    }

    return results