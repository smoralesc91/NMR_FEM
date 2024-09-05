from tqdm import tqdm
import mpmath
import numpy as np
import scipy.optimize as opt
from scipy.integrate import quad
from Functions_NMR import mag_sat

def NMR_SemiA_sphere(radius=1., T2B=1., diffusion=1., rho=1.,
                     B_0=0.05, Temp=303.15, fluid='water',
                     n_terms=10, root_lim_left=np.pi, root_lim_right=np.pi, root_lim_tol=1.e-5, 
                     t_0=0, t_f=1, dt=0.1, print_time=False, 
                     volume_=False, return_data='all'):

    def f(x):
        return float(1-(x*mpmath.cot(x)) - ((radius*rho)/diffusion))
        
    eigbetam = []
    root = []
    Tn_values = []

    bulk_term = 1/T2B if T2B > 0 else 0.

    for i in range(1, n_terms + 1):
        a = (i - 1) * root_lim_left + root_lim_tol
        b = i * root_lim_right - root_lim_tol
        r = opt.brentq(f, float(a), float(b), xtol=2.e-12, maxiter=100)
        rt = np.power(r, 2)
        eigbetam.append(r)
        root.append(rt)
        Tn = radius**2 / (diffusion * r**2 + bulk_term * radius**2)
        Tn_values.append(Tn)
    
    eigbetam = np.unique(eigbetam)
    lengthv = len(eigbetam)

    volume_sphere = 4/3 * np.pi * radius**3
    
    def M(t, volume_):
        s = 0.0
        for i in range(0, lengthv):
            r = eigbetam[i]
            a = np.sin(r) - r * np.cos(r)
            b = r**3 * (2 * r - np.sin(2 * r)) 
            m0 = mag_sat(B_0, Temp, fluid) * volume_sphere if volume_ else mag_sat(B_0, Temp, fluid)
            s += m0 * (12 * a**2 / b * np.exp(-t / Tn_values[0]))
        return s

    # Time array
    nt = int((t_f - t_0) / dt) + 1
    time_array = np.linspace(t_0, t_f, nt)
    
    # Magnetization vector
    mag_amounts = np.zeros(nt)

    if print_time:
        for i in tqdm(range(nt), desc='Progress', total=nt-1):
            mag_amounts[i] = M(time_array[i], volume_)
    else:
        for i in range(nt):
            mag_amounts[i] = M(time_array[i], volume_)

    # NMR integral
    mag_assemble = quad(M, 0, volume_sphere, args=(volume_))[0]

    if return_data == 'all':
        return time_array, mag_amounts, eigbetam, root, Tn_values, mag_assemble
    elif return_data == 'mag_assemble':
        return mag_assemble
    elif return_data == 'mag_amounts':
        return mag_amounts