from dolfin import *
import numpy as np

__version__ = "1.3-legacy"

# Intento de importar utilidades del usuario
try:
    from Functions_NMR import mag_sat, mesh_statistics, T2star_conventional
except Exception:
    mag_sat = None
    mesh_statistics = None
    T2star_conventional = None

# Silenciar salida
set_log_level(LogLevel.ERROR)
parameters["linear_algebra_backend"] = "PETSc"
parameters["form_compiler"]["representation"] = "uflacs"

def NMR_FEM(
    radius=1.0, mesh_res=200, mesh_stats=False,          # malla
    T2B=1.0, diffusion=1.0, rho=1.0,                     # parámetros físicos
    B_0=0.05, Temp=303.15, fluid='water',                # IC vía mag_sat
    t_0=0.0, t_f=1.0, dt=1e-2,                           # tiempo
    volume_=False, return_data='all',                    # 'all' | 'mag_amounts' | 'mag_assemble'
    linear_solver='mumps',                               # 'mumps' | 'cg'
    progress=True                                        # usa tqdm si está disponible
):
    """
    FEM 1D radial esférico (Bloch–Torrey con T2B y relajación superficial):
        ∂t m = D (m_rr + 2/r m_r) - m/T2B   en r∈(0,R)
        D m_r(R) + ρ m(R) = 0,   m_r(0)=0
        M(t) = 4π ∫_0^R m(r,t) r^2 dr
    Forma variacional ponderada por r^2 (evita 1/r en r≈0).
    """

    # --- Mallado 1D en [0, R]
    mesh = IntervalMesh(MPI.comm_self, mesh_res, 0.0, radius)

    # Estadísticas de malla (opcional)
    if mesh_stats and callable(mesh_statistics):
        try:
            mesh_statistics(mesh, T2B, rho)
        except Exception as e:
            warning("mesh_statistics() lanzó una excepción y se omitió: %s" % str(e))

    # Marcado de borde exterior r=R con id=1
    facet_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
    class Outer(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], radius, DOLFIN_EPS)
    Outer().mark(facet_markers, 1)

    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh, subdomain_data=facet_markers)

    # Espacio y funciones
    V = FunctionSpace(mesh, "CG", 2)
    m = TrialFunction(V)
    w = TestFunction(V)

    # Coordenada radial y pesos
    x = SpatialCoordinate(mesh)
    r = x[0]
    r2 = r*r
    R = Constant(float(radius))

    # Condición inicial con mag_sat si existe
    try:
        m0_val = float(mag_sat(B_0, Temp, fluid)) if callable(mag_sat) else 1.0
    except Exception:
        m0_val = 1.0
    m_prev = Function(V)
    m_prev.interpolate(Constant(m0_val))

    # Coeficientes físicos y temporales
    D = Constant(float(diffusion))
    invT2B = Constant(0.0 if (T2B is None or T2B <= 0.0) else 1.0/float(T2B))
    dtc = Constant(float(dt))
    rho_c = Constant(float(rho))

    # Forma variacional ponderada por r^2
    a = ( m*w*r2*dx
          + dtc*D*dot(grad(m), grad(w))*r2*dx
          + dtc*invT2B*m*w*r2*dx
          + dtc*rho_c*R*R*m*w*ds(1) )
    L = m_prev*w*r2*dx

    # Ensamble de matriz una vez
    A = assemble(a)

    # Solver PETSc
    use_mumps = (linear_solver.lower() == "mumps")
    if use_mumps:
        solver = LUSolver(A, "mumps")
    else:
        solver = KrylovSolver("cg", "hypre_amg")
        solver.set_operator(A)

    # Discretización temporal
    nt = int(round((t_f - t_0)/dt)) + 1
    time_array = np.linspace(t_0, t_f, nt)

    # Arreglo de señal
    mag_amounts = np.zeros(nt)
    denom = assemble(r2*dx)

    # Primera muestra
    m_now = Function(V)
    if volume_:
        mag_amounts[0] = 4.0*np.pi * assemble(m_prev*r2*dx)
    else:
        mag_amounts[0] = assemble(m_prev*r2*dx) / denom

    # --- tqdm opcional
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

    # Bucle temporal
    for i in iterator:
        b = assemble(L)  # RHS con m_prev

        if use_mumps:
            solver.solve(m_now.vector(), b)
        else:
            solver.solve(m_now.vector(), b)

        # Señal
        if volume_:
            mag_amounts[i] = 4.0*np.pi * assemble(m_now*r2*dx)
        else:
            mag_amounts[i] = assemble(m_now*r2*dx) / denom

        # Avanzar
        m_prev.assign(m_now)

        # Info adicional en la barra (opcional)
        if use_tqdm and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(t=time_array[i])

    # Ensamble final de M(t) físico
    mag_assemble = 4.0*np.pi * assemble(m_now*r2*dx)

    if return_data == 'all':
        return time_array, m_now, mag_amounts, mag_assemble
    elif return_data == 'time-mag':
        return time_array, mag_amounts
    elif return_data == 'mag_assemble':
        return mag_assemble
    elif return_data == 'mag_amounts':
        return mag_amounts


NMR_FEM.__version__ = __version__