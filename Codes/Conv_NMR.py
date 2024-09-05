import numpy as np
from scipy.integrate import quad
from Functions_NMR import mag_sat, SV_ratio_analytical, T2star_conventional

#TO-DO: añadir opciones para estimar T2 bulk
def NMR_Conventional(radius=1, aspect_ratio = 1, #SV_ratio 
                     T2B=1., diffusion=2.3e-9, rho=30e-6, #T2_star
                     B_0=0.05, Temp=303.15, fluid='water', #mag_sat 
                     t_0=0, t_f=1, dt=1.e-3, #time must be an array
                     volume_=False , return_data = 'all'): 
    
    nt = int((t_f-t_0)/dt) + 1
    time_array = np.linspace(t_0, t_f, nt)
    
    S_value_geom, V_value_geom, SV_ratio_value_geom = SV_ratio_analytical(radius, aspect_ratio)
        
    # Calculating Amplitude
    def A(ms, V, volume_):
        return ms * V if volume_ else ms
    
    # Calculating NMR signal
    def nmr_signal(time_array, radius, aspect_ratio, T2B, rho, B_0, Temp, fluid, volume_):
        m_sat = mag_sat(B_0, Temp, fluid)
        S_value, V_value, SV_ratio_value = SV_ratio_analytical(radius, aspect_ratio)
        Amplitude_value = A(m_sat, V_value, volume_)
        T2s = T2star_conventional(T2B, rho, SV_ratio_value)
        return Amplitude_value * np.exp(-time_array / T2s)

    # NMR signal
    mag_amounts = nmr_signal(time_array, radius, aspect_ratio, T2B, rho, B_0, Temp, fluid, volume_)
    
    # NMR integral
    mag_assemble = quad(nmr_signal, 0, V_value_geom, args=(radius, aspect_ratio, T2B, rho, B_0, Temp, fluid, volume_))[0]

    if return_data == 'all':
        return time_array, mag_amounts, mag_assemble
    elif return_data == 'mag_assemble':
        return mag_assemble
    elif return_data == 'mag_amounts':
        return mag_amounts