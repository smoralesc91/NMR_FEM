# ==============================================================================
# 1. ENVIRONMENT SETUP (Google Colab Support)
# ==============================================================================
import dolfinx
import numpy as np
import ufl
import time
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc

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
    """
    
    # ==========================================================================
    # 1. LINEAR SOLVER CONFIGURATION
    # ==========================================================================
    def get_solver_options(solver_type):
        """
        Returns PETSc solver options for the selected direct solver.
        
        Parameters
        ----------
        solver_type : str
            Solver type identifier.
            
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
        # For 'default' or any other, PETSc uses its internal LU.
        
        return opts

    # ==========================================================================
    # 2. DIMENSIONLESS SCALING
    # ==========================================================================
    # Compute characteristic diffusion length
    diff_length = np.sqrt(D_phys * T2B_phys)
    
    # Dimensionless parameters
    PHI_R_VAL = R_phys / diff_length                   # Dimensionless radius
    ALPHA_VAL = rho_phys * np.sqrt(T2B_phys / D_phys)  # Dimensionless rho
    
    # Dimensionless time parameters
    TAU_START = t_start / T2B_phys
    TAU_FINAL = t_final / T2B_phys
    DT_VAL    = dt_phys / T2B_phys
    
    # MPI communicator
    comm = comm
    
    # ==========================================================================
    # 3. FINITE ELEMENT MESH AND FUNCTION SPACE
    # ==========================================================================
    # Create 1D interval [0, PHI_R_VAL] representing the radial coordinate
    domain = mesh.create_unit_interval(comm, NUM_ELEMS)
    domain.geometry.x[:, 0] *= PHI_R_VAL
    
    # Continuous Galerkin function space
    V = fem.functionspace(domain, ("CG", DEGREE))
    
    # Define boundary at the pore wall (r = PHI_R_VAL)
    def is_pore_wall(x):
        return np.isclose(x[0], PHI_R_VAL, rtol=1e-10)
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, is_pore_wall)
    boundary_tags = mesh.meshtags(domain, fdim, boundary_facets,
                                  np.full_like(boundary_facets, 1))
    
    # Integration measures
    dx = ufl.dx
    ds = ufl.Measure("ds", domain=domain, subdomain_data=boundary_tags)
    
    # Trial and test functions, and functions for current and previous solutions
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    u_n, u_curr = fem.Function(V), fem.Function(V)
    
    # Constants for dimensionless parameters
    dt    = fem.Constant(domain, PETSc.ScalarType(DT_VAL))
    alpha = fem.Constant(domain, PETSc.ScalarType(ALPHA_VAL))
    phi_R = fem.Constant(domain, PETSc.ScalarType(PHI_R_VAL))
    
    # Spatial coordinate and radial variable
    x = ufl.SpatialCoordinate(domain)
    phi = x[0]  # radial coordinate
    
    # Initial condition: uniform magnetization M(r,0) = 1
    u_n.interpolate(lambda x: np.ones_like(x[0]))
    
    # ==========================================================================
    # 4. VARIATIONAL FORMULATION (BACKWARD EULER)
    # ==========================================================================
    # Weak form of the dimensionless Bloch-Torrey equation with spherical symmetry.
    # The Jacobian r² (phi**2) is included in all volume integrals.
    F_lhs = (u * v * phi**2 * dx +
             dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * phi**2 * dx +
             dt * u * v * phi**2 * dx +
             dt * alpha * phi_R**2 * u * v * ds(1))
    
    F_rhs = u_n * v * phi**2 * dx
    
    # Linear problem with specified solver options
    solver_options = get_solver_options(SOLVER_TYPE)
    problem = LinearProblem(F_lhs, F_rhs,
                            petsc_options_prefix="solver",
                            petsc_options=solver_options)
    
    # Analytical normalization factor for spherical volume
    norm_factor = 3.0 / (PHI_R_VAL**3)
    
    # ==========================================================================
    # 5. TIME INTEGRATION LOOP
    # ==========================================================================
    duration_tau = TAU_FINAL - TAU_START
    steps = int(np.round(duration_tau / DT_VAL))
    
    # Arrays to store time and raw (un-normalized) signal
    time_array = []
    signal_raw = []
    
    # Compute initial magnetization signal by volume integration
    vol_int_0 = fem.assemble_scalar(fem.form(u_n * phi**2 * dx))
    vol_int_0 = comm.allreduce(vol_int_0, op=MPI.SUM)
    signal_0 = norm_factor * vol_int_0
    
    time_array.append(t_start)
    signal_raw.append(signal_0)
    
    # Optional verbose output
    if verbose and comm.rank == 0:
        start_time = time.perf_counter()
        print(f"Starting simulation: {steps} time steps")
        print(f"Dimensionless radius φ_R = {PHI_R_VAL:.3f}")
        print(f"Dimensionless rho α = {ALPHA_VAL:.3f}")
    
    # Time stepping loop
    for step in range(steps):
        # Solve linear system for current time step
        u_curr = problem.solve()
        
        # Update dimensionless and physical time
        curr_tau = TAU_START + (step + 1) * DT_VAL
        curr_phys_time = curr_tau * T2B_phys
        
        # Compute volume integral of magnetization
        vol_int = fem.assemble_scalar(fem.form(u_curr * phi**2 * dx))
        vol_int = comm.allreduce(vol_int, op=MPI.SUM)
        signal = norm_factor * vol_int
        
        # Store results on rank 0
        if comm.rank == 0:
            time_array.append(curr_phys_time)
            signal_raw.append(signal)
        
        # Update previous solution for next time step
        u_n.x.array[:] = u_curr.x.array[:]
    
    # Optional verbose output: timing
    if verbose and comm.rank == 0:
        end_time = time.perf_counter()
        print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    # ==========================================================================
    # 6. NORMALIZATION AND OUTPUT
    # ==========================================================================
    if comm.rank == 0:
        # Convert to numpy arrays
        time_array = np.array(time_array)
        signal_raw = np.array(signal_raw)
        
        # Normalize signal by its initial value
        signal_norm = signal_raw / signal_raw[0]
        
        # Optional file export
        if export_files:
            np.savetxt("nmr_decay.txt",
                       np.column_stack((time_array, signal_norm)),
                       header="Time_s Signal",
                       comments='')
        
        return time_array, signal_norm
    
    # For non-zero MPI ranks, return None
    return None, None