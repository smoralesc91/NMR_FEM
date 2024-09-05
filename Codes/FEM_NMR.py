from dolfin import *
import mshr
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from Functions_NMR import T2star_conventional
from Functions_NMR import mag_sat
from Functions_NMR import mesh_statistics

__version__ = "1.0"

set_log_level(30)

parameters['allow_extrapolation'] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4
parameters['form_compiler']['representation'] = 'uflacs'
parameters['linear_algebra_backend'] = "PETSc"

def NMR_FEM(radius=1., mesh_res=10, mesh_stats=False,       # mesh options
            T2B=1., diffusion=1, rho=1,             # model parameters 
            B_0=0.05, Temp=303.15, fluid='water',   # initial condition ms
            t_0=0, t_f=1, dt=0.1, print_time=False, # time parameters
            volume_=False, return_data='all',       # return data: 'all', 'mag_amounts', 'mag_assemble'
            linear_solver='umfpack'): 
    
    mesh = IntervalMesh(MPI.comm_self, mesh_res, 0, radius)
    marker_sub = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
        
    class Boundary(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1.e-14
            return on_boundary and abs(x[0]-radius) < tol
        
    boundary_ = Boundary()
    boundary_.mark(marker_sub, 1)
    dx = Measure('dx', domain=mesh)
    ds = Measure('ds', domain=mesh, subdomain_data=marker_sub)

    if mesh_stats:
        mesh_statistics(mesh, T2B, rho)
         
    # Define Finte Element, Function Space, Test function (weight function) and Trial function
    polynomial_degree = 2
    element  = FiniteElement(family='CG', cell=mesh.ufl_cell(), 
                             degree=polynomial_degree, quad_scheme='default')

    V = FunctionSpace(mesh, element)
    m = TrialFunction(V)
    w = TestFunction(V)
    
    # Initial condition over the domain
    m_sat = mag_sat(B_0, Temp, fluid) # Magnetic saturation (Curie's law)
    m_0 = Constant(m_sat)

    m_init = Function(V)
    m_init.interpolate(m_0)
    
    # Define Diffusion coefficient
    D = diffusion
    
    # Bilinear a(u, w) and linear form L(w)
    bulk_term = dt / T2B * m * w * dx if T2B > 0 else 0.

    r = Expression('sqrt(x[0]*x[0])', degree=2)
    r2 = r*r
    a = m*w*dx + dt*D*Dx(m,0)*Dx(w,0)*dx + dt*rho*m*w*ds(1) - dt*D*Constant(2)/r*Dx(m,0)*w*dx + bulk_term 
    
    L = m_init*w*dx

    # Apply initial condition
    m = Function(V)
    m.assign(m_init)

    # Time array
    nt = int((t_f-t_0)/dt) + 1 
    time_array = np.linspace(t_0, t_f, nt)

    # Save magnetization vector 
    mag_amounts = np.zeros(nt)

    # Precompute denominator if volume_ is False
    denominator = assemble(r2*dx(mesh)) if not volume_ else 1.

    # Integer initial magnetization over the magnetization vector 
    if volume_:
        mag_amounts[0] = assemble(Constant(4)*np.pi*m*r2*dx)
    else:
        mag_amounts[0] = assemble(m*r2*dx) / denominator

    # Loop through time steps
    if print_time:
        for i, t in tqdm(enumerate(time_array[1:]), desc='Progress', total=nt-1):
            solve(a == L, m, solver_parameters={'linear_solver': linear_solver}, 
                             form_compiler_parameters={'optimize': True,
                                                      'cpp_optimize': True,})
    
            if volume_:
                mag_amounts[i+1] = assemble(Constant(4)*np.pi*m*r2*dx)
            else:
                mag_amounts[i+1] = assemble(m*r2*dx) / denominator
            
            m_init.assign(m)
    else:
        for i in range(1, nt):
            solve(a == L, m, solver_parameters={'linear_solver': linear_solver}, 
                             form_compiler_parameters={'optimize': True,
                                                      'cpp_optimize': True,})
    
            if volume_:
                mag_amounts[i] = assemble(Constant(4)*np.pi*m*r2*dx)
            else:
                mag_amounts[i] = assemble(m*r2*dx) / denominator
            
            m_init.assign(m)

    # Integer to obtain total magnetization M(t)
    mag_assemble = assemble(Constant(4)*np.pi*m*r2*dx) 
    
    if return_data == 'all':
        return time_array, m, mag_amounts, mag_assemble
    elif return_data == 'mag_assemble':
        return mag_assemble
    elif return_data == 'mag_amounts':
        return mag_amounts

NMR_FEM.__version__ = __version__