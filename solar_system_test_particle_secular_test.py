import celmech as cm
import rebound as rb
import numpy as np
from test_particle_secular_hamiltonian import SyntheticSecularTheory, TestParticleSecularHamiltonian
from celmech.miscellaneous import frequency_modified_fourier_transform as fmft
from matplotlib import pyplot as plt
from argparse import ArgumentParser
import os
ARCSEC_PER_YR = 1/(180*60*60*2)

def select_by_nearest_key(d,val):
    """
    Select dictionary key and value by identifying the key value numerically closest to 'val'.

    Parameters
    ----------
    d : dictionary
        The dictionary from which to select a key-value pair.
    val : float
        The value for which to find the closest key.

    Returns
    -------
    closest_key : 
        The key in the dictionary that is closest to 'val'.
    closest_value : 
        The value corresponding to the closest key in the dictionary.
    """
    keys_arr = np.array(list(d.keys()))
    nearest_key = keys_arr[np.argmin(np.abs(keys_arr - val))]
    return nearest_key,d[nearest_key]

def setup_solar_system_integration(test_particle_semimajor_axes):
    """
    Set up a solar system integration with the Sun, Jupiter, Saturn, Uranus, Neptune, and test particles.
    """
    sim = rb.Simulation()
    sim.add("Sun")
    for planet in ("Jupiter","Saturn","Uranus","Neptune"):
        sim.add(planet)
    sim.N_active = sim.N

    cm.nbody_simulation_utilities.align_simulation(sim)
    for a in test_particle_semimajor_axes:
        sim.add(m=0, a=a, e=0.01, inc=0.0, pomega='uniform',primary=sim.particles[0])
    sim.integrator = "whfast"
    sim.ri_whfast.safe_mode = 0
    sim.dt = np.min([p.P for p in sim.particles[1:]]) / 20.0
    return sim

def run_and_save_simulation(sim, tmax, Nout, filename):
    """
    Run simulation and save to file.

    Parameters
    ----------
    sim : rebound.Simulation
        simulation to run
    tmax : float
        time to integrate to
    Nout : int
        number of outputs to save
    filename : string
        name of file to save simulation to.
    """
    total_steps = tmax / sim.dt
    step = int(np.floor(total_steps / Nout))    
    sim.save_to_file(filename,step = step,delete_file=True)
    sim.integrate(tmax, exact_finish_time=0)

def get_planets_fmft_results(results):
    """
    Get FMFT results for giant planets' complex inclinations and eccentricities.

    Parameters
    ----------
    results : dictionary
        dictionary of integration results as returned by celmech's get_simarchive_integration_results function.

    Returns
    -------
    ecc_fmft_results : dictionary
        A dictionary containing the eccentricity FMFT results for each planet.
    inc_fmft_results : dictionary
        A dictionary containing the inclination FMFT results for each planet.
    """
    planets = ("Jupiter","Saturn","Uranus","Neptune")

    results['X'] = np.sqrt(2*(1-np.sqrt(1-results['e']**2))) * np.exp(1j * results['pomega'])
    results['Y'] = (1-results['e']**2)**(0.25) * np.sin(0.5 * results['inc'] )* np.exp(1j * results['Omega'])

    ecc_fmft_results = dict()
    inc_fmft_results = dict()
    for i,pl in enumerate(planets):
        ecc_fmft_results[pl] = fmft(results['time'],results['X'][i],14)
        inc_fmft_results[pl] = fmft(results['time'],results['Y'][i],8)

    return ecc_fmft_results, inc_fmft_results

def construct_synthetic_secular_theory(masses,semimajor_axes,ecc_fmft_results, inc_fmft_results):
    """
    Construct a secular theory from the given FMFT results.

    Parameters
    ----------
    masses : list
        A list of the masses of the planets.
    semimajor_axes : list
        A list of the semimajor axes of the planets.
    ecc_fmft_results : dict
        A dictionary containing the eccentricity FMFT results for each planet.
    inc_fmft_results : dict
        A dictionary containing the inclination FMFT results for each planet.

    Returns
    -------
    secular_theory : SyntheticSecularTheory
        An instance of the SyntheticSecularTheory class constructed from the provided FMFT results.
    """
    planets = ("Jupiter","Saturn","Uranus","Neptune")
    g_vec = np.zeros(4)
    s_vec = np.zeros(3)

    g_vec[:3] = np.array(list(ecc_fmft_results['Jupiter'].keys()))[:3]
    g_vec[3] = list(ecc_fmft_results['Neptune'].keys())[0]
    s_vec[0] = list(inc_fmft_results['Jupiter'].keys())[0]
    s_vec[1] = list(inc_fmft_results['Jupiter'].keys())[2]
    s_vec[2] = list(inc_fmft_results['Jupiter'].keys())[1]
    omega_vec = np.concatenate((g_vec,s_vec))
    eye_N = np.eye(omega_vec.size,dtype = int)

    # eccentricty term dictionaries
    x_dicts = []
    for pl in planets:
        print(pl)    
        print("-"*len(pl))
        print("kvec \t\t\t omega \t err. \t amplitude")
        x_dict = {}
        for i,omega_i in enumerate(omega_vec[:4]):
            omega_N,amp = select_by_nearest_key(ecc_fmft_results[pl],omega_vec[i])
            omega_error = np.abs(omega_N/omega_i-1)
            if omega_error<0.001:
                print (eye_N[i],"\t{:+07.3f}\t{:.1g} \t{:.1g}".format(omega_i/ARCSEC_PER_YR,omega_error,np.abs(amp)))
                x_dict[tuple(eye_N[i])] = amp
        #NL terms
        for a in range(7):
            for b in range(a,7):
                for c in range(7):
                    if c==a:
                        continue
                    if c==b:
                        continue
                    k = np.zeros(7,dtype = int)
                    k[a] +=1
                    k[b] +=1
                    k[c] -=1
                    omega=k@omega_vec
                    omega_N,amp = select_by_nearest_key(ecc_fmft_results[pl],omega)
                    omega_error = np.abs(omega_N/omega-1)
                    if omega_error<0.001:
                        print (k,"\t{:+07.3f}\t{:.1g},\t{:.1g}".format(omega/ARCSEC_PER_YR,omega_error,np.abs(amp)))
                        x_dict[tuple(k)] = amp
        x_dicts.append(x_dict)
    y_dicts = []

    for pl in planets:
        print(pl)    
        print("-"*len(pl))
        print("kvec \t\t\t omega \t err. \t amplitude")
        y_dict = {}
        for i,omega_i in enumerate(omega_vec[4:]):
            omega_N,amp = select_by_nearest_key(inc_fmft_results[pl],omega_vec[4+i])
            omega_error = np.abs(omega_N/omega_i-1)
            if omega_error<0.001:
                print (eye_N[4+i],"\t{:+07.3f}\t{:.1g} \t{:.1g}".format(omega_i/ARCSEC_PER_YR,omega_error,np.abs(amp)))
                y_dict[tuple(eye_N[4+i])] = amp
        #NL terms
        for a in range(7):
            for b in range(a,7):
                for c in range(7):
                    if c==a:
                        continue
                    if c==b:
                        continue
                    k = np.zeros(7,dtype = int)
                    k[a] +=1
                    k[b] +=1
                    k[c] -=1
                    omega=k@omega_vec
                    omega_N,amp = select_by_nearest_key(inc_fmft_results[pl],omega)
                    omega_error = np.abs(omega_N/omega-1)
                    if omega_error<0.001:
                        print (k,"\t{:+07.3f}\t{:.1g} \t{:.1g}".format(omega/ARCSEC_PER_YR,omega_error,np.abs(amp)))
                        y_dict[tuple(k)] = amp
        y_dicts.append(y_dict)

    return SyntheticSecularTheory(
        masses,
        semimajor_axes,
        omega_vec,
        x_dicts,
        y_dicts
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("--tmax", type=float, default=5e7, help="Maximum integration time.")
    parser.add_argument("--Nout", type=int, default=4096, help="Number of output points.")
    parser.add_argument("--min_test_particle_semimajor_axes", type=float, default= 0.9, help="Minimum test particle semimajor axes.")
    parser.add_argument("--max_test_particle_semimajor_axes", type=float, default= 1.5, help="Maximum test particle semimajor axes.")
    parser.add_argument("--N_test_particle", type=int, default=10, help="Number of test particles.")
    parser.add_argument("--output_filename", type=str, default="solar_system_test_particle_secular_test.bin", help="Output filename for the simulation data.")
    parser.add_argument("--rerun_simulation", action="store_true", help="Flag to rerun the simulation even if the output file exists.")
    args = parser.parse_args()

    if args.rerun_simulation or not os.path.exists(args.output_filename):
        print(f"Output file {args.output_filename} not found. Running simulation...")
        test_particle_semimajor_axes = np.linspace(
            args.min_test_particle_semimajor_axes, 
            args.max_test_particle_semimajor_axes, 
            args.N_test_particle
        )
        sim = setup_solar_system_integration(test_particle_semimajor_axes)
        run_and_save_simulation(sim, args.tmax, args.Nout, args.output_filename)
        results = cm.nbody_simulation_utilities.get_simarchive_integration_results(args.output_filename,coordinates="heliocentric")

    else:
        sim = rb.Simulation(args.output_filename)
        results = cm.nbody_simulation_utilities.get_simarchive_integration_results(args.output_filename,coordinates="heliocentric")
        print(f"Loaded simulation results from {args.output_filename}.")

    masses =  np.array([sim.particles[i].m for i in range(1, sim.N_active)])
    semimajor_axes = np.array([np.mean(results['a'][i]) for i in range(sim.N_active-1)])
    synthetic_theory = construct_synthetic_secular_theory(
        masses,
        semimajor_axes,
        *get_planets_fmft_results(results)
    )
    print("Masses of the planets:", masses)
    print("Semimajor axes of the planets:", semimajor_axes)
    print("Number of planets:",synthetic_theory.N_planets)
    print("Omega vector:", synthetic_theory.omega_vector)
    
    particle_fmft_results_e = [
        fmft(results['time'],results['X'][i],6) for i in range(sim.N_active-1,sim.N-1)
    ]
    particle_fmft_results_inc = [
        fmft(results['time'],2*results['Y'][i],6) for i in range(sim.N_active-1,sim.N-1)
    ]
    for key,val in particle_fmft_results_inc[0].items():
        print(key,np.abs(val))
    a0s = np.array([np.mean(results['a'][i]) for i in range(sim.N_active-1,sim.N-1)])
    test_particle_hamiltonians = [
        TestParticleSecularHamiltonian(a0,synthetic_theory) for a0 in a0s
    ]
    g0s_pred = np.array([tph.g0 for tph in test_particle_hamiltonians])
    g0s_obs = np.array([select_by_nearest_key(particle_fmft_results_e[i],g0_pred)[0] for i,g0_pred in enumerate(g0s_pred)])
    s0s_pred = np.array([tph.s0 for tph in test_particle_hamiltonians])
    s0s_obs = np.array([select_by_nearest_key(particle_fmft_results_inc[i],s0_pred)[0] for i,s0_pred in enumerate(s0s_pred)])

    fig,axes = plt.subplots(4,2,sharex=True,figsize=(8,12))
    for ax in axes.flatten():
        plt.sca(ax)
        plt.tick_params(axis='both', which='major', labelsize=10,size=8,direction='in',top=True,right=True)

    axes[0,0].plot(a0s,1e5*g0s_pred,'kx-',ms=10,lw=3)
    axes[0,0].plot(a0s,1e5*g0s_obs,'o',ms=10)

    axes[0,1].plot(a0s,1e5*s0s_pred,'kx-',ms=10,lw=3)
    axes[0,1].plot(a0s,1e5*s0s_obs,'o',ms=10)

    for i,ax in enumerate(axes[1:,0]):
        omega_target = synthetic_theory.omega_vector[i]
        
        e_nui_pred = [
            np.abs(tph.F_e[tuple(np.eye(synthetic_theory.N_freq,dtype=int)[i])])
            for tph in test_particle_hamiltonians
        ]
        e_nui_obs = np.abs(
            [select_by_nearest_key(res,omega_target)[1] for res in particle_fmft_results_e]
        )
        nui_obs = np.abs(
            [select_by_nearest_key(res,omega_target)[0] for res in particle_fmft_results_e]
        )
        msk = np.abs(nui_obs / omega_target-1)<0.02
        ax.plot(a0s,e_nui_pred,'kx-')
        ax.scatter(a0s[msk],e_nui_obs[msk])

    for i,ax in enumerate(axes[1:,1]):
        # 1. Move the text label to the right side
        ax.yaxis.set_label_position("right")
        # 2. Move the tick marks and numeric labels to the right side
        ax.yaxis.tick_right()

        omega_target = synthetic_theory.omega_vector[4 + i]
        print(omega_target)
        inc_nui_pred = [
            np.abs(tph.F_inc[tuple(np.eye(synthetic_theory.N_freq,dtype=int)[4+i])])
            for tph in test_particle_hamiltonians
        ]
        inc_nui_obs = np.abs(
            [select_by_nearest_key(res,omega_target)[1] for res in particle_fmft_results_inc]
        )
        nui_obs = np.array([select_by_nearest_key(res,omega_target)[0] for res in particle_fmft_results_inc])

        msk = np.abs(nui_obs / omega_target-1)<0.02
        ax.plot(a0s,inc_nui_pred,'kx-')
        ax.scatter(a0s[msk],inc_nui_obs[msk])

    axes[-1,0].set_xlabel("Test Particle Semimajor Axis [AU]",fontsize=12)
    axes[-1,1].set_xlabel("Test Particle Semimajor Axis [AU]",fontsize=12)
    axes[0,0].set_ylabel("g0 [arcsec/yr]",fontsize=12)
    axes[0,1].set_ylabel("s0 [arcsec/yr]",fontsize=12)
    for i in range(1,4):
        axes[i,0].set_ylabel("e_nu_{}".format(4+i),fontsize=12)
        axes[i,1].set_ylabel("i_nu_{}".format(4+i),fontsize=12)
    plt.show()  
if __name__=="__main__":
    main()