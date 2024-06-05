from dolfin import *
import mshr
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from Functions_NMR import T2star_conventional

set_log_level(30)

parameters['allow_extrapolation'] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 3
parameters['form_compiler']['representation'] = 'uflacs'
parameters['linear_algebra_backend'] = "PETSc"

def NMR_FEM(radius=1., aspect_ratio = 1, dimension=1, mesh_res=10, 
            mesh_stats=False, mesh_load=None,       #mesh options, mesh_load must be a list (mesh, markers, subdomains)
            T2B=1., diffusion=1, rho=1,             #model parameters 
            B_0=0.05, Temp=303.15, fluid='water',   #initial condition ms
            t_0=0, t_f=1, dt=0.1, print_time=False, #time parameters
            volume_=False, return_data='all',       #return data: 'all', 'mag_amounts', 'mag_assemble'
            linear_solver='umfpack'): 
    
    if mesh_load:
        mesh = mesh_load[0]
        dx = Measure('dx', domain=mesh, subdomain_data=mesh_load[1])
        ds = Measure('ds', domain=mesh, subdomain_data=mesh_load[2])
    elif dimension == 1:
        mesh = UnitIntervalMesh(mesh_res)
        mesh.scale(radius)
        marker_sub = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
        
        class Boundary(SubDomain):
            def inside(self, x, on_boundary):
                tol = 1.e-14
                return on_boundary and abs(x[0]-radius) < tol
        
        boundary_ = Boundary()
        boundary_.mark(marker_sub, 1)
        dx = Measure('dx', domain=mesh)
        ds = Measure('ds', domain=mesh, subdomain_data=marker_sub)
    elif dimension == 2:
        mesh = UnitSquareMesh(mesh_res, mesh_res)
        mesh.scale(radius)
    else:  # dimension == 3
        domain = mshr.Ellipsoid(Point(0, 0, 0), 1, aspect_ratio, aspect_ratio)
        mesh = mshr.generate_mesh(domain, mesh_res)
        mesh.scale(radius)
    
    if not mesh_load and dimension > 1:
        dx = Measure('dx', domain=mesh)
        ds = Measure('ds', domain=mesh, subdomain_data=mesh)

    if mesh_stats:
        def mesh_statistics(mesh):
            if mesh.topology().dim() == 1:
                r2 = Expression('x[0]*x[0]', degree=3)
                volume_mesh = assemble(Constant(4) * np.pi * r2 * dx(mesh))
                surface_mesh = assemble(Constant(4) * np.pi * r2 * ds(mesh))
            else:
                volume_mesh = assemble(Constant(1) * dx(mesh))
                surface_mesh = assemble(Constant(1) * ds(mesh))
        
            surface_volume_ratio = surface_mesh / volume_mesh
        
            mesh_data = [
                ["hmin", "{:.4e}".format(mesh.hmin())],
                ["hmax", "{:.4e}".format(mesh.hmax())],
                ["num. cells", mesh.num_cells()],
                ["num. edges", mesh.num_edges()],
                ["num. entities 0d", mesh.num_entities(0)],
                ["num. entities 1d", mesh.num_entities(1)],
                ["num. entities 2d", mesh.num_entities(2)],
                ["num. entities 3d", mesh.num_entities(3)],
                ["num. faces", mesh.num_faces()],
                ["num. facets", mesh.num_facets()],
                ["num. vertices", mesh.num_vertices()],
                ["Volume", "{:.4e} [m^3]".format(volume_mesh)],
                ["Surface area", "{:.4e} [m^2]".format(surface_mesh)],
                ["Surface to Volume ratio", "{:.4e} [m^-1]".format(surface_volume_ratio)],
                ["T2star conventional", "{:.4e} [s]".format(T2star_conventional(T2B, rho, surface_volume_ratio))]
            ]
        
            print(tabulate(mesh_data, headers=["Mesh statistics", ""], tablefmt="github"))

        mesh_statistics(mesh)
        
    # Calculating magnetic saturation (Curie's law)
    def mag_sat(B_0, Temp, fluid):
        avogadro_number = 6.0220e23
        h_planck = 6.626e-34
        k_boltzmann = 1.380e-23
        gamma = 267.5e6
        
        assert fluid in ['water', 'oil', 'gas']

        if fluid == 'water':
            number_hydrogen = 2.
            fluid_mol_weight = 18.0153e-3
            fluid_density = 9.97e2
        elif fluid == 'oil':
            number_hydrogen = 12.
            fluid_mol_weight = 72.151e-3
            fluid_density = 6.26e2
        elif fluid == 'gas':
            number_hydrogen = 4.
            fluid_mol_weight = 16.04e-3
            fluid_density = 6.56e-1

        proton_density = (number_hydrogen * avogadro_number * fluid_density) / fluid_mol_weight
        m_s = (proton_density * B_0 * (gamma**2) * (h_planck**2)) / (4. * k_boltzmann * Temp)
        return m_s

    # Define Finte Element, Function Space, Test function (weight function) and Trial function
    polynomial_degree = 2
    element  = FiniteElement(family='CG', cell=mesh.ufl_cell(), 
                             degree=polynomial_degree, quad_scheme='default')

    V = FunctionSpace(mesh, element)
    m = TrialFunction(V)
    w = TestFunction(V)
    
    # Initial condition over the domain
    m_sat = mag_sat(B_0, Temp, fluid)
    m_0 = Constant(m_sat)

    m_init = Function(V)
    m_init.interpolate(m_0)
    

    # Identify mesh dimension
    dim = mesh.topology().dim()

    # Define Diffusion tensor
    if dim == 1:
        D = diffusion
    if dim == 2:
        D = as_matrix(((diffusion, 0),
                       (0, diffusion)))
    elif dim == 3:
        D = as_matrix(((diffusion, 0, 0),
                       (0, diffusion, 0),
                       (0, 0, diffusion)))

    # Bilinear a(u, w) and linear form L(w)
    if T2B <= 0:
        bulk_term = 0.
    else:
        bulk_term = dt * (Constant(1) / Constant(T2B)) * m * w * dx

    if dim == 1:
        r = Expression('sqrt(x[0]*x[0])', degree=2)
        r2 = r*r
        a = m*w*dx + dt*D*dot(grad(m),grad(w))*dx + dt*Constant(rho)*m*w*ds(1) - dt*D*Constant(2)/r*Dx(m,0)*w*dx + bulk_term 
        #a = m*w*dx + dt*D*Dx(m,0)*Dx(w,0)*dx + dt*rho*m*w*ds(1) - dt*D*Constant(2)/r*Dx(m,0)*w*dx + bulk_term 
    else:
        a = m*w*dx + dt*inner(dot(D, grad(m)), grad(w))*dx + dt*Constant(rho)*m*w*ds + bulk_term
    
    L = m_init*w*dx

    # Apply initial condition
    m = Function(V)
    m.assign(m_init)

    # Time array
    nt = int((t_f-t_0)/dt) + 1 
    times = np.linspace(t_0, t_f, nt)

    # Save magnetization vector 
    mag_amounts = np.zeros(nt)

    # Integer initial magnetization over the magentization vector 
    if volume_:
        if dim == 1:
            mag_amounts[0] = assemble(Constant(4)*np.pi*m*r2*dx)
        else:
            mag_amounts[0] = assemble(m*dx)
    else:
        if dim == 1:
            mag_amounts[0] = assemble(Constant(4)*np.pi*m*r2*dx) / assemble(Constant(4)*np.pi*r2*dx(mesh))
        else:
            mag_amounts[0] = assemble(m*dx) / assemble(Constant(1.0)*dx(mesh))
    
    # Solve the problem
    if print_time:
        for i, t in tqdm(enumerate(times[1:]), desc='Progress', total=nt-1):
            solve(a == L, m, solver_parameters={'linear_solver': linear_solver}, 
                             form_compiler_parameters={"optimize": True})

            if volume_:
                # Integer over the domain
                if dim == 1:
                    mag_amounts[i+1] = assemble(Constant(4)*np.pi*m*r2*dx)
                else:
                    mag_amounts[i+1] = assemble(m*dx)
            else:
                if dim ==1:
                    mag_amounts[i+1] = assemble(Constant(4)*np.pi*m*r2*dx) / assemble(Constant(4)*np.pi*r2*dx(mesh))
                else:
                    mag_amounts[i+1] = assemble(m*dx) / assemble(Constant(1.0)*dx(mesh))

            m_init.assign(m)
    else:
        for i, t in enumerate(times[1:], 1):
            solve(a == L, m, solver_parameters={'linear_solver': linear_solver}, 
                             form_compiler_parameters={"optimize": True})

            if volume_:
                # Integer over the domain
                if dim == 1:
                    mag_amounts[i] = assemble(Constant(4)*np.pi*m*r2*dx)
                else:
                    mag_amounts[i] = assemble(m*dx)
            else:
                if dim ==1:
                    mag_amounts[i] = assemble(Constant(4)*np.pi*m*r2*dx) / assemble(Constant(4)*np.pi*r2*dx(mesh))
                else:
                    mag_amounts[i] = assemble(m*dx) / assemble(Constant(1.0)*dx(mesh))

            m_init.assign(m)

    # Integer to obtain total magnetization M(t)
    if dim == 1:
        mag_assemble = assemble(Constant(4)*np.pi*m*r2*dx) 
    else:
        mag_assemble = assemble(m*dx)

    if return_data == 'all':
        return times, m, mag_amounts, mag_assemble
    elif return_data == 'mag_assemble':
        return mag_assemble
    elif return_data == 'mag_amounts':
        return mag_amounts