import rebound as rb
import reboundx as rbx
import celmech as cm
import numpy as np
import sys
sa = rb.Simulationarchive("sim_mass_growth.bin")
m0 = sa[0].particles[1].m
Tfin = sa.tmax
m_fin = sa[-1].particles[1].m
tau_mass = Tfin / np.log(m_fin / m0)

mass_i = int(sys.argv[1])
m_targets = np.linspace(1e-5,5e-5,15)
m_target = m_targets[mass_i]
time_target = tau_mass * np.log(m_target/m0)

# set up new simulation
sim_old = sa.getSimulation(time_target)
sim = rb.Simulation()
for p in sim_old.particles:
    sim.add(p.copy())
sim.N_active = 2 
sim.move_to_com()
sim.integrator='whfast'
sim.dt = sim.particles[1].P / 30.
extras = rbx.Extras(sim)
mod = extras.load_operator("modify_orbits_direct")
omega_sec = sim.particles[1].n * 1e-5
sim.particles[1].params['tau_omega'] = 2*np.pi / (omega_sec)
extras.add_operator(mod)

Tfin_approx = (1e-5 / m_target) * 2e6 * 2 * np.pi
total_steps = np.ceil(Tfin_approx / sim.dt)
Tfin = total_steps * sim.dt + sim.dt
Nout = 2048

new_file = "sim_m_{:.2g}_long.sa".format(sim.particles[1].m)
try: 
    results = cm.nbody_simulation_utilities.get_simarchive_integration_results(new_file,coordinates='heliocentric')
except:
    sim.save_to_file(new_file,step=int(np.floor(total_steps/Nout)),delete_file=True)
    sim.integrate(Tfin,exact_finish_time=0)
    results = cm.nbody_simulation_utilities.get_simarchive_integration_results(new_file,coordinates='heliocentric')
np.savez_compressed(new_file.replace("_long.sa","results.bin"),**results)