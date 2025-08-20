import numpy as np
from celmech.poisson_series import PSTerm, PoissonSeries
_RT2 = np.sqrt(2)
_RT2_INV = 1/_RT2
class PoissonSeriesHamiltonian:
    r"""
    Represents a Hamiltonian and its corresponding equations of motion expressed as a Poisson series.

    The Hamiltonian and equations of motion are assumed to be functions of the real variables
    .. math::
        y_1, \dots, y_N,\ \theta_1, \dots, \theta_M,\ x_1, \dots, x_N,\ p_1, \dots, p_M

    These are mapped to complex Poisson series variables via
    .. math::
        z_j = \frac{x_j - i y_j}{\sqrt{2}}

    Parameters
    ----------
    hamiltonian_poisson_series : PoissonSeries
        The Hamiltonian expressed as a Poisson series in terms of the complex variables.
    """

    def __init__(self, hamiltonian_poisson_series):
        self.hamiltonian = hamiltonian_poisson_series
        self._cvar_flow = hamiltonian_series_to_flow_series_list(hamiltonian_poisson_series)
        self._cvar_real_flow_jacobian = real_vars_jacobian_series(self._cvar_flow)

    @property
    def N(self):
        """Number of canonical (z, z̄) variables (i.e., degrees of freedom related to x, y)."""
        return self.hamiltonian.N

    @property
    def M(self):
        """Number of canonical (Q, P) variables (i.e., angle-action type coordinates)."""
        return self.hamiltonian.M

    def _real_vars_to_cvars(self, real_vars):
        """
        Convert the real-valued state vector to complex canonical variables.

        Parameters
        ----------
        real_vars : array_like
            A 1D array of real variables in the order:
            [y₁, ..., y_N, Q₁, ..., Q_M, x₁, ..., x_N, P₁, ..., P_M]

        Returns
        -------
        z : ndarray
            Complex canonical variables defined as z_j = (x_j - i y_j) / sqrt(2)
        P : ndarray
            Canonical momenta corresponding to Q
        Q : ndarray
            Canonical coordinates corresponding to P
        """
        y = real_vars[:self.N]
        Q = real_vars[self.N:self.N + self.M]
        x = real_vars[self.N + self.M:2 * self.N + self.M]
        P = real_vars[2 * self.N + self.M:2 * self.N + 2 * self.M]
        z =   (x - 1j * y) / _RT2
        return z, P, Q

    def __call__(self, real_vars):
        """
        Evaluate the Hamiltonian at the given point in real variable space.

        Parameters
        ----------
        real_vars : array_like
            Real-valued phase space coordinates.

        Returns
        -------
        float
            Value of the Hamiltonian.
        """
        z, P, Q = self._real_vars_to_cvars(real_vars)
        return self.hamiltonian(z, P, Q)

    def flow(self, real_vars):
        """
        Compute the time derivatives of the real-valued state vector (Hamiltonian flow).

        Parameters
        ----------
        real_vars : array_like
            Real-valued phase space coordinates.

        Returns
        -------
        ndarray
            Time derivatives [dy/dt, dQ/dt, dx/dt, dP/dt] of the real variables.
        """
        z, P, Q = self._real_vars_to_cvars(real_vars)
        zdot = np.array([f(z, P, Q) for f in self._cvar_flow[:self.N]])
        Pdot = np.real([f(z, P, Q) for f in self._cvar_flow[self.N:self.N + self.M]])
        Qdot = np.real([f(z, P, Q) for f in self._cvar_flow[self.N + self.M:]])
        return np.concatenate((-np.imag(zdot) * _RT2, Qdot, np.real(zdot) * _RT2, Pdot))

    def jacobian(self, real_vars):
        """
        Compute the Jacobian matrix of the Hamiltonian flow at the given state.

        Parameters
        ----------
        real_vars : array_like
            Real-valued phase space coordinates.

        Returns
        -------
        ndarray
            The Jacobian matrix of the flow, shape (2N+2M, 2N+2M)
        """
        z, P, Q = self._real_vars_to_cvars(real_vars)
        jj = np.array([[series(z, P, Q) for series in row] for row in self._cvar_real_flow_jacobian])
        dzdot_dvar = jj[:self.N, :]
        dxdot_dvar = _RT2 * np.real(dzdot_dvar)
        dydot_dvar = -_RT2 * np.imag(dzdot_dvar)
        dPdot_dvar = np.real(jj[self.N:self.N + self.M, :])
        dQdot_dvar = np.real(jj[self.N + self.M:, :])
        return np.vstack([dydot_dvar, dQdot_dvar, dxdot_dvar, dPdot_dvar])


def hamiltonian_series_to_flow_series_list(ham_series):
    I_N = np.eye(ham_series.N,dtype=int)
    zero_N = np.zeros(ham_series.N,dtype=int)
    I_M = np.eye(ham_series.M,dtype=int)
    zero_M = np.zeros(ham_series.M,dtype=int)
    flow_series_list = []
    # dx/dt
    for i in range(ham_series.N):
        var_series = PSTerm(1,I_N[i],zero_N,zero_M,zero_M).as_series()
        dxi_dt = ham_series.Lie_deriv(var_series)
        flow_series_list.append(dxi_dt)
    # dP/dt 
    for i in range(ham_series.M):
        var_series = PSTerm(1,zero_N,zero_N,I_M[i],zero_M).as_series()
        dPi_dt = ham_series.Lie_deriv(var_series)
        flow_series_list.append(dPi_dt)
    # dQ/dt 
    for i in range(ham_series.M):
        var_series = PSTerm(1,zero_N,zero_N,zero_M,I_M[i]).as_series()
        dexp_iQ_dt = ham_series.Lie_deriv(var_series)
        factor = PSTerm(-1j,zero_N,zero_N,zero_M,-I_M[i]).as_series()
        dQi_dt = factor * dexp_iQ_dt
        flow_series_list.append(dQi_dt)
    return flow_series_list
def dseries_dQi(series,i):
    return PoissonSeries.from_PSTerms([1j * term.q[i] * term for term in series.terms if term.q[i]], N=series.N, M=series.M)
def dseries_dPi(series,i):
    one_i = np.eye(series.M)[i]
    return PoissonSeries.from_PSTerms([PSTerm(term.C * term.p[i], term.k,term.kbar,term.p - one_i,term.q) for term in series.terms if term.p[i]], N=series.N, M=series.M)
def dseries_dzi(series,i):
    one_i = np.eye(series.N)[i]
    return PoissonSeries.from_PSTerms([PSTerm(term.C * term.k[i], term.k - one_i,term.kbar,term.p,term.q) for term in series.terms if term.k[i]], N=series.N, M=series.M)
def dseries_dzbari(series,i):
    one_i = np.eye(series.N)[i]
    return PoissonSeries.from_PSTerms([PSTerm(term.C * term.kbar[i], term.k,term.kbar - one_i,term.p,term.q) for term in series.terms if term.kbar[i]], N=series.N, M=series.M)

def real_vars_jacobian_series(complex_flow_series):
    flow0 = complex_flow_series[0]
    N,M = flow0.N,flow0.M
    N_dim = 2 * (N+M)
    jac = [[None for j in range(N_dim)] for i in range(N+2*M)]
    for i,flow_i in enumerate(complex_flow_series):
        for j in range(N):
            df_dz = dseries_dzi(flow_i,j)
            df_dzbar = dseries_dzbari(flow_i,j)
            df_dy = 1j * _RT2_INV * (df_dzbar + -1*df_dz)
            df_dx = _RT2_INV * (df_dzbar + df_dz)
            jac[i][j] = df_dy
            jac[i][N+M+j] = df_dx
        for j in range(M):            
            df_dP = dseries_dPi(flow_i,j)
            df_dQ = dseries_dQi(flow_i,j)
            jac[i][N+j] = df_dQ
            jac[i][2*N+M+j] = df_dP
    return jac

