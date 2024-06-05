from tqdm import tqdm
import mpmath
import numpy as np
import scipy.optimize as opt
from scipy.integrate import quad

def NMR_SemiA_sphere(radius=1., T2B=1., diffusion=1., rho=1.,
                     B_0=0.05, Temp=303.15, fluid='water',
                     n=10, rleft=np.pi, rright=np.pi, tol=1.e-5, 
                     t_0=0, t_f=1, dt=0.1, print_time=False, 
                     volume_=False, return_data='all'):

    def f(x):
        return float(1-(x*mpmath.cot(x)) - ((radius*rho)/diffusion))

    # Calculating magnetic saturation
    def mag_sat(B_0, Temp, fluid):
        assert fluid in ['water', 'oil', 'gas']
        avogadro_number = 6.0220e23
        h_planck = 6.626e-34
        k_boltzmann = 1.380e-23
        gamma = 267.5e6

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
        
    eigbetam = []
    root = []
    Tn_values = []
    
    for i in range(1, n + 1):
        a = (i - 1) * rleft + tol
        b = i * rright - tol
        r = opt.brentq(f, float(a), float(b))
        rt= np.power(r, 2)
        eigbetam.append(r)
        root.append(rt)
        Tn = np.power(radius, 2) / (diffusion * np.power(r, 2))
        Tn_values.append(Tn)
    
    eigbetam = np.unique(eigbetam)
    lengthv = len(eigbetam)

    volume_sphere = 4/3 * np.pi * radius**3
    
    def M(t, volume_):
        s = 0.0
        for i in range(0, lengthv):
            r = eigbetam[i]
            a = np.sin(r) - r * np.cos(r)
            b = np.power(r,3) * (2 * r - np.sin(2 * r)) 
            if T2B <= 0:
                bulk_term = 0
            else:
                bulk_term = 1/T2B
            if volume_:
                m0 = mag_sat(B_0, Temp, fluid) * volume_sphere
            else:
                m0 = mag_sat(B_0, Temp, fluid)
            s += m0 * (12 *np.power(a,2)/ b * np.exp(-t*((1/Tn_values[0]) + bulk_term)))
        return s
        
    t = 0.0
    NT = int((t_f - t_0) / dt)
    times = np.zeros(NT + 1)
    MM = np.zeros(NT + 1)

    if print_time:
        for i in tqdm(range(0, NT + 1), desc='Progress', total=NT):
            t = i * dt
            times[i] = t
            MM[i] = M(t, volume_)
    else:
        for i in range(0, NT+1):
            t = i * dt
            times[i] = t
            MM[i] = M(t, volume_)

    # NMR integral
    mag_assemble = quad(M, 0, volume_sphere, args=(volume_))[0]

    if return_data == 'all':
        return times, MM, eigbetam, root, Tn_values, mag_assemble
    elif return_data == 'mag_assemble':
        return mag_assemble
    elif return_data == 'mag_amounts':
        return MM