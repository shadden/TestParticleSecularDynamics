import numpy as np
import celmech as cm
from celmech.disturbing_function import df_coefficient_C, evaluate_df_coefficient_dict

def calc_g0_and_s0(semi_major_axis,synthetic_secular_theory,GM=1.0):
    """
    Evaluate the lowest-order prediction for the secular free apsidal and nodal precession frequencies for a test particle.

    Parameters
    ----------
    semi_major_axis : float
        Semi-major axis of test particle
    ss_theory : SyntheticSecularTheory
        Representation of planetary system in which test particle resides.
    GM : float, optional
        Newton's constant times stellar mass, by default 1.0

    Returns
    -------
    _type_
        _description_
    """
    n = np.sqrt(GM / semi_major_axis**3)
    z6 = [0 for _ in range(6)]
    C_e_in = df_coefficient_C(*z6,0,0,1,0)
    C_e_out = df_coefficient_C(*z6,0,0,0,1)
    C_I_in = df_coefficient_C(*z6,1,0,0,0)
    C_I_out = df_coefficient_C(*z6,0,1,0,0)
    g0,s0 = 0,0
    for m_i,a_i in zip(synthetic_secular_theory.masses,synthetic_secular_theory.semi_major_axes):
        if a_i>semi_major_axis:
            alpha = semi_major_axis/a_i
            C_e = evaluate_df_coefficient_dict(C_e_in,alpha)
            C_I = evaluate_df_coefficient_dict(C_I_in,alpha)
            g0 += m_i * alpha * C_e
            s0 += m_i * alpha * C_I
        else:
            alpha = a_i / semi_major_axis
            C_e = evaluate_df_coefficient_dict(C_e_out,alpha)
            C_I = evaluate_df_coefficient_dict(C_I_out,alpha)
            g0 += m_i *  C_e
            s0 += m_i *  C_I
    g0 *= 2*n
    s0 *= 0.5*n
    return g0,s0

class SyntheticSecularTheory():
    def __init__(self,masses,semi_major_axes,omega_vector,x_dicts,y_dicts):
        self.masses = masses
        self.semi_major_axes = semi_major_axes
        self.N_freq = len(omega_vector)
        self.x_dicts = x_dicts
        self.y_dicts = y_dicts



class TestParticleSecularHamiltonian():
    def __init__(self):
        pass