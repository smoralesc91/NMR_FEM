import numpy as np
from scipy.integrate import quad

#TO-DO: añadir opciones para estimar T2 bulk
def NMR_Conventional(radius=1, aspect_ratio = 1, #SV_ratio 
                     T2B=1., diffusion=2.3e-9, rho=30e-6, #T2_star
                     B_0=0.05, Temp=303.15, fluid='water', #mag_sat 
                     t_0=0, t_f=1, dt=1.e-3, #time must be an array
                     volume_=False , return_data = 'all'): 
    
    nt = int((t_f-t_0)/dt) + 1
    t = np.linspace(t_0, t_f, nt)
    
    assert fluid in ['water', 'oil', 'gas']
    
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
    
    # Calculating surface area and volume ratio
    def SV_ratio(radius, aspect_ratio):

        if aspect_ratio == 1:
            S = 4. * np.pi * np.power(radius, 2)
            V = 4. * np.pi * np.power(radius, 3) / 3.
        else:
            a = radius
            b = radius*aspect_ratio
            c = radius*aspect_ratio
            S = 4. * np.pi * (((a * b) ** 1.6 + (a * c) ** 1.6 + (b * c) ** 1.6) / 3.) ** (1. / 1.6)
            V = 4. * np.pi * a * b * c / 3.
            
        SVratio = S / V
        return S, V, SVratio

    S_value_geom, V_value_geom, SV_ratio_value_geom = SV_ratio(radius, aspect_ratio)
        
    # Calculating Amplitude
    def A(ms, V, volume_):
        if volume_:
            Amplitude = ms * V
        else:
            Amplitude = ms
        return Amplitude

    # Calculating conventional T2star
    def T2star_conventional(T2B, rho, SVratio):
        if T2B <= 0:
            T_2star = 1. / (rho * SVratio)
        else:
            T_2star = 1. / ((1. / T2B) + rho * SVratio)
        return T_2star
    
    # Calculating NMR signal
    def nmr_signal(t, radius, aspect_ratio, T2B, rho, B_0, Temp, fluid, volume_):
        m_sat = mag_sat(B_0, Temp, fluid)
        S_value, V_value, SV_ratio_value = SV_ratio(radius, aspect_ratio)
        Amplitude_value = A(m_sat, V_value, volume_)
        T2s = T2star_conventional(T2B, rho, SV_ratio_value)
        return Amplitude_value * np.exp(-t / T2s)

    # NMR signal
    mag_amounts = nmr_signal(t, radius, aspect_ratio, T2B, rho, B_0, Temp, fluid, volume_)
    
    # NMR integral
    mag_assemble = quad(nmr_signal, 0, V_value_geom, args=(radius, aspect_ratio, T2B, rho, B_0, Temp, fluid, volume_))[0]

    if return_data == 'all':
        return t, mag_amounts, mag_assemble
    elif return_data == 'mag_assemble':
        return mag_assemble
    elif return_data == 'mag_amounts':
        return mag_amounts