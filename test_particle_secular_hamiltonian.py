import numpy as np
import celmech as cm
from celmech.disturbing_function import df_coefficient_C, evaluate_df_coefficient_dict
from celmech.poisson_series import PSTerm, PoissonSeries

def list_multinomial_exponents(pwr,ndim):
    r"""
    List exponents appearing in the expansion of the multinomial 
    
    .. math:: 
    (x_1 + x_2 + ... + x_\mathrm{ndim})^pwr

    Parameters
    ----------
    pwr : int
        Exponent of multinomial
    ndim : int
        Number of variables in multinomial

    Returns
    -------
    list
        A list of lists of length `ndim` representing the exponents of each
        varaible :math:`x_i`
    """
    if ndim==1:
        return [[pwr],]
    else:
        x =[]
        for pow1 in range(0,pwr+1):
            subpows = list_multinomial_exponents(pwr-pow1,ndim-1)
            x+=[[pow1]+y for y in subpows]
        return x
    
from math import factorial
def multinomial_coefficient(p, ks):
    """Calculate multinomial coefficient for given p and ks"""
    num = factorial(p)
    denom = 1
    for k in ks:
        denom *= factorial(k)
    return num // denom

def calc_g0_and_s0(semi_major_axis,synthetic_secular_theory,GM=1.0):
    """
    Evaluate the lowest-order prediction for the secular free apsidal and nodal precession frequencies for a 
    test particle.

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
_p  = lambda k,nu: max(0,k) + nu
_p1 = lambda k,nu: max(0,-k) + nu 
_RT2 = np.sqrt(2)

def test_particle_secular_terms(a,k,nu,synthetic_scular_system):
    C_coeff = df_coefficient_C(k,nu)
    for i in range(synthetic_scular_system.N_planets):
        a_i = synthetic_scular_system.semi_major_axes[i]
        if a<a_i:
            alpha = a/a_i
            C_i = evaluate_df_coefficient_dict(C_coeff,alpha)
            pow_2_factor = _RT2**(2*nu[2]-2*nu[0] + np.abs(k[2])-np.abs(k[0]))
            X_i_pow = _p(k[3],nu[3])
            Xbar_i_pow = _p1(k[3],nu[3])
            Y_i_pow = _p(k[5],nu[1])
            Ybar_i_pow = _p1(k[5],nu[1])
            
            X_i_series_terms=[]
            for pows in list_multinomial_exponents(X_i_pow,synthetic_scular_system.M_i[i]):
                coeff = multinomial_coefficient(X_i_pow,pows)

    
class SyntheticSecularTheory():
    def __init__(self,masses,semi_major_axes,omega_vector,x_dicts,y_dicts):
        self.masses = masses
        self.semi_major_axes = semi_major_axes
        self.N_planets = len(self.masses)
        self.N_freq = len(omega_vector)
        self.x_dicts = x_dicts
        self.y_dicts = y_dicts
        self.M_i = np.array([len(x_d) for x_d in x_dicts],dtype=int)
        self.N_i = np.array([len(y_d) for y_d in y_dicts],dtype=int)
        self.m_ls = [list(x_d.keys()) for x_d in x_dicts]
        self.n_ls = [list(y_d.keys()) for y_d in y_dicts]
        self.omega_vector = omega_vector
    @property
    def Nfreq(self):
        return len(self.omega_vector)
    
    def Xi_to_pow_poisson_series(self,i,pow):
        M_i = self.M_i[i]
        x_dict = self.x_dicts[i]
        m_ls = self.m_ls[i]
        
        pvec = np.zeros(self.Nfreq,dtype = int)
        terms = []
        for ks in list_multinomial_exponents(pow,M_i):
            qvec = np.zeros(len(self.omega_vector),dtype = int)
            coeff = multinomial_coefficient(pow,ks)
            for m_l,k in zip(m_ls,ks):
                if k>0:
                    coeff *= (x_dict[m_l])**k
                    qvec += k * np.array(m_l)
            terms.append(PSTerm(coeff,[0],[0],pvec,qvec))
        return PoissonSeries.from_PSTerms(terms)

    def Yi_to_pow_poisson_series(self,i,pow):
        N_i = self.N_i[i]
        y_dict = self.y_dicts[i]
        n_ls = self.n_ls[i]
        
        pvec = np.zeros(self.Nfreq,dtype = int)
        terms = []
        for ks in list_multinomial_exponents(pow,N_i):
            qvec = np.zeros(len(self.omega_vector),dtype = int)
            coeff = multinomial_coefficient(pow,ks)
            for m_l,k in zip(n_ls,ks):
                if k>0:
                    coeff *= (y_dict[m_l])**k
                    qvec += k * np.array(m_l)
            terms.append(PSTerm(coeff,[0],[0],pvec,qvec))
        return PoissonSeries.from_PSTerms(terms)

    def Xbari_to_pow_poisson_series(self,i,pow):
        return self.Xi_to_pow_poisson_series(i,pow).conj
    
    def Ybari_to_pow_poisson_series(self,i,pow):
        return self.Yi_to_pow_poisson_series(i,pow).conj
        
from collections import defaultdict

def mathcal_X_dictionary(semi_major_axis,synthetic_secular_theory,GM=1):
    n = np.sqrt(GM / semi_major_axis**3)
    z4 = [0 for _ in range(4)]
    C_e_mixed = df_coefficient_C(*[0,0,1,-1,0,0],*z4)
    Xcal_dict = defaultdict(float)
    for i in range(synthetic_secular_theory.N_planets):
        a_i = synthetic_secular_theory.semi_major_axes[i]
        m_i = synthetic_secular_theory.masses[i]
        if a_i>semi_major_axis:
            C  = (semi_major_axis/a_i) * evaluate_df_coefficient_dict(C_e_mixed,semi_major_axis/a_i)
        else:
            C  = evaluate_df_coefficient_dict(C_e_mixed,a_i/semi_major_axis)
        x_dict = synthetic_secular_theory.x_dicts[i]
        for mvec,amplitude in x_dict.items():
            Xcal_dict[mvec] += m_i*n*C*amplitude
    return Xcal_dict

def mathcal_Y_dictionary(semi_major_axis,synthetic_secular_theory,GM=1):
    n = np.sqrt(GM / semi_major_axis**3)
    z4 = [0 for _ in range(4)]
    C_I_mixed = df_coefficient_C(*[0,0,0,0,1,-1],*z4)
    Ycal_dict = defaultdict(complex)
    for i in range(synthetic_secular_theory.N_planets):
        a_i = synthetic_secular_theory.semi_major_axes[i]
        m_i = synthetic_secular_theory.masses[i]
        if a_i>semi_major_axis:
            C  = 0.5 * (semi_major_axis/a_i) * evaluate_df_coefficient_dict(C_I_mixed,semi_major_axis/a_i)
        else:
            C  = 0.5 * evaluate_df_coefficient_dict(C_I_mixed,a_i/semi_major_axis)
        y_dict = synthetic_secular_theory.y_dicts[i]
        for mvec,amplitude in y_dict.items():
            Ycal_dict[mvec] += m_i*n*C*amplitude
    return Ycal_dict

class TestParticleSecularHamiltonian():
    def __init__(self,semi_major_axis,synthetic_secular_theory,GM=1.0):
        """
        Class representing the Hamiltonian of a test particle subject to secular forcing from a system of planets.

        Parameters
        ----------
        semi_major_axis : float
            The test particle's semi-major axis
        synthetic_secular_theory : SyntheticSecularTheory
            A representation of the planetary system's secular dynamics.
        GM : float, optional
            Newton's constant times the stellar mass, by default 1.0
        """
        self.semi_major_axis = semi_major_axis
        self.synthetic_secular_theory = synthetic_secular_theory
        self.g0,self.s0 = calc_g0_and_s0(semi_major_axis,synthetic_secular_theory,GM=GM)
        mathcal_X_dict = mathcal_X_dictionary(semi_major_axis,synthetic_secular_theory)
        mathcal_Y_dict = mathcal_Y_dictionary(semi_major_axis,synthetic_secular_theory)

        # Get forced eccentricity
        self.F_e = dict()
        for m,amp in mathcal_X_dict.items():
            omega_m = np.dot(m,self.omega_vector)
            denom = self.g0 - omega_m
            self.F_e[m] = -1*amp / denom

        # Get forced inclination
        self.F_inc = dict()
        for m,amp in mathcal_Y_dict.items():
            omega_m = np.dot(m,self.omega_vector)
            denom = self.s0 - omega_m
            self.F_inc[m] = -1*amp / denom

    @property
    def omega_vector(self):
        return np.array(self.synthetic_secular_theory.omega_vector)
    
    @property
    def M(self):
        return len(self.F_e)
    
    @property
    def N(self):
        return len(self.F_inc)
    
    def x_to_pow_poisson_series(self,p):
        exponents = list_multinomial_exponents(p,self.M+1)
        mvecs = list(self.F_e.keys())
        terms = []
        for ks in exponents:
            coeff = multinomial_coefficient(p,ks)
            k0 = ks[0]
            qvec = np.zeros(self.omega_vector.size,dtype = int)
            pvec = np.zeros(self.omega_vector.size,dtype = int)
            for k,m_l in zip(ks[1:],mvecs):
                if k>0:
                    coeff *= (self.F_e[m_l])**k
                    qvec += k*np.array(m_l)
            terms.append(PSTerm(coeff,[k0],[0],pvec,qvec))
        return PoissonSeries.from_PSTerms(terms)
    
    def y_to_pow_poisson_series(self,p):
        exponents = list_multinomial_exponents(p,self.N+1)
        nvecs = list(self.F_inc.keys())
        terms = []
        for ks in exponents:
            coeff = multinomial_coefficient(p,ks)
            k0 = ks[0]
            qvec = np.zeros(self.omega_vector.size,dtype = int)
            pvec = np.zeros(self.omega_vector.size,dtype = int)
            for k,n_l in zip(ks[1:],nvecs):
                if k>0:
                    coeff *= (self.F_inc[n_l])**k
                    qvec += k*np.array(n_l)
            terms.append(PSTerm(coeff,[k0],[0],pvec,qvec))
        return PoissonSeries.from_PSTerms(terms)
    
    def xbar_to_pow_poisson_series(self,p):
        return self.x_to_pow_poisson_series(p).conj
    
    def ybar_to_pow_poisson_series(self,p):
        return self.y_to_pow_poisson_series(p).conj

    def linear_theory_solution(self,x0,y0,times):
        xForced = np.sum([amp * np.exp(1j * np.dot(m,self.omega_vector) * times) for m, amp in self.F_e.items()],axis=0)            
        yForced = np.sum([amp * np.exp(1j * np.dot(m,self.omega_vector) * times) for m, amp in self.F_inc.items()],axis=0)
        x0_free = x0 - xForced[0]
        y0_free = y0 - yForced[0]
        x_soln = x0_free * np.exp(1j * self.g0 * times) + xForced
        y_soln = y0_free * np.exp(1j * self.s0 * times) + yForced
        return x_soln,y_soln
