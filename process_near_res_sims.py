import celmech as cm
import rebound as rb
import numpy as np
from test_particle_secular_hamiltonian import SyntheticSecularTheory
from test_particle_secular_hamiltonian import TestParticleSecularHamiltonian
from celmech.miscellaneous import frequency_modified_fourier_transform as fmft
from matplotlib import pyplot as plt
def closest_per_row(gs, targets):
    """
    For each row in `gs`, return the value closest to the corresponding value in `targets`.

    Parameters:
    gs (np.ndarray): A 2D array of shape (n, m), each row is a 1D array to search.
    targets (np.ndarray): A 1D array of shape (n,), each value is the target for the corresponding row.

    Returns:
    np.ndarray: A 1D array of shape (n,), each entry is the closest value in the corresponding row of `gs`.
    """
    # Compute absolute differences between each element and its row-specific target
    diffs = np.abs(gs - targets[:, np.newaxis])  # shape (n, m)
    
    # Indices of the closest values in each row
    idx = np.argmin(diffs, axis=1)
    
    # Extract the closest values using advanced indexing
    result = gs[np.arange(gs.shape[0]), idx]
    
    return result
def get_fmft_results(sim,results,Ne_freq = 6,Ni_freq = 4):
    T = results['time']
    results['x'] = np.sqrt(2) * np.sqrt(1-np.sqrt(1-results['e']**2)) * np.exp(1j * results['pomega'])
    results['y'] = (1-results['e']**2)**(0.25) * np.sin(0.5*results['inc']) * np.exp(1j * results['Omega'])
    x_fmft_results = []
    y_fmft_results = []
    for i in np.where([p.m==0 for p in sim.particles[1:]])[0]:
        x_i = results['x'][i]
        x_fmft = fmft(T,x_i,Ne_freq)
        x_fmft_results.append(x_fmft)
        y_i = results['x'][i]
        y_fmft = fmft(T,y_i,Ni_freq)
        y_fmft_results.append(y_fmft)
    return x_fmft_results, y_fmft_results
if __name__=="__main__":
    import glob
    from celmech.nbody_simulation_utilities import get_simarchive_integration_results as get_results
    archive_files = glob.glob("sim_m_*_long.sa")
    results_dict = dict()
    sim_dict = dict()
    for archive_file in archive_files:
        sim = rb.Simulation(archive_file)
        mass = sim.particles[1].m
        results_dict[mass] = get_results(archive_file,coordinates='heliocentric')
        sim_dict[mass] = sim
    masses = np.array(list(results_dict.keys()))
    masses = np.sort(masses)

    fmft_results = dict()
    for mass in masses:
        results = results_dict[mass]
        sim = sim_dict[mass]
        planet = sim.particles[1]
        omega_sec = planet.n * 1e-5
        a_tp = np.mean(results['a'][1])
        ss_theory = SyntheticSecularTheory(
            [p.m for p in sim.particles[1:2]],
            [p.a for p in sim.particles[1:2]],
            np.array([omega_sec]),
            [{(1,):sim.particles[1].e},],
            [{(1,):0,},]
        )
        tph = TestParticleSecularHamiltonian(a_tp,ss_theory)
        tph._update_linear_theory_with_second_order_in_mass_terms({0:[(3,1)]})
        g0 = tph.g0
        x_fmfts, y_fmfts = get_fmft_results(sim,results)
        N_tp=len(x_fmfts)
        e_forced = np.zeros(N_tp)
        e_free = np.zeros(N_tp)
        g_free = np.zeros(N_tp)
        for i,x_fmft in enumerate(x_fmfts):
            freqs = np.array(list(x_fmft.keys()))
            j = np.argmin(np.abs(freqs - omega_sec))
            e_forced[i] = np.abs(x_fmft[freqs[j]])
            j_free = np.argmin(np.abs(freqs - g0))
            g_free[i] = freqs[j_free]
            e_free[i] = np.abs(x_fmft[freqs[j_free]])
        fmft_results[mass]  = {"g0":g0,"g_free":g_free,"e_free":e_free,"e_forced":e_forced}

    import pickle
    with open("fmft_results.bin","wb") as fi:
        pickle.dump(fmft_results,fi)
