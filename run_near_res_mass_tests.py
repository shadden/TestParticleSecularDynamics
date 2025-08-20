import rebound as rb
import reboundx as rbx
import numpy as np

from celmech.nbody_simulation_utilities import add_canonical_heliocentric_elements_particle
from test_particle_secular_hamiltonian import SyntheticSecularTheory, TestParticleSecularHamiltonian
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--Delta", required=True,type=int, help="Integer index of Delta value to use")
parser.add_argument("--j", required=True, type = int, help="Together with 'k', specify resonance.")
parser.add_argument("--k", required=True, type = int, help="Together with 'k', specify resonance.")

args = parser.parse_args()

Delta_vals = np.linspace(0.005,0.025,9)
Delta = Delta_vals[args.Delta]
jres = args.j
kres = args.k
period_ratio = (1 + Delta) * jres/(jres-kres)

m_pl0 = 1e-7
e_pl = 0.01
pomega_pl = 0.
g_pl = 2*np.pi * 1e-5
T_pl = 2*np.pi / g_pl
T_growth = 10 * T_pl
x_pl = np.sqrt(2*(1-np.sqrt(1-e_pl**2))) * np.exp(1j * pomega_pl)
secular_theory = SyntheticSecularTheory(
    [m_pl0,],
    [1,],
    [g_pl,],
    [{(1,):x_pl},],
    [{(1,):0,},]
)

sim = rb.Simulation()
sim.add(m=1)
add_canonical_heliocentric_elements_particle(m_pl0,{'a':1,'e':0.01,'l':0,'pomega':0},sim)
sim.N_active = sim.N

a_tp = (period_ratio)**(2/3)
tp_h = TestParticleSecularHamiltonian(a_tp,secular_theory)

for e_free in np.linspace(0.001,0.02,5):
    z_tot  = e_free * np.exp(1j * 0.5 * np.pi) + np.sum(list(tp_h.F_e.values()))
    add_canonical_heliocentric_elements_particle(0,{'a':a_tp,'e':np.abs(z_tot),'l':np.pi,'pomega':np.angle(z_tot)},sim)
    
extras = rbx.Extras(sim)
mod = extras.load_operator("modify_orbits_direct")
sim.particles[1].params['tau_omega'] = T_pl
extras.add_operator(mod)

growth = extras.load_operator("modify_mass")
sim.particles[1].params['tau_mass'] = 10 * T_pl 
extras.add_operator(growth)

sim.integrator='whfast'
sim.dt = 2*np.pi / 20.

Tint =  np.log(0.5e-5/1e-7) * T_growth
mass_growth_file = f"./archives/mass_growth_j_{jres:d}_k_{kres:d}_Delta_{Delta:.4f}.sa"
sim.save_to_file(mass_growth_file,interval=Tint / 256 ,delete_file=True)
sim.integrate(Tint)


sa = rb.Simulationarchive(mass_growth_file)
for i,mass in enumerate(np.linspace(1e-6,0.45e-5,10)):
    print(f"running mass {i}")
    T_target = np.log(mass/1e-7) * T_growth
    sim_new = sa.getSimulation(T_target)
    sim_new.t = 0
    extras = rbx.Extras(sim_new)
    mod = extras.load_operator("modify_orbits_direct")
    sim_new.particles[1].params['tau_omega'] = T_pl
    extras.add_operator(mod)

    Tint = 2e7
    Nout = 513
    steps = int(np.floor(Tint / Nout / sim_new.dt))
    sim_new.save_to_file(f'./archives/test_particles_mass_{i}_j_{jres:d}_k_{kres:d}_Delta{Delta:.4f}.sa',step = steps,delete_file=True)
    sim_new.integrate(Tint,exact_finish_time=0)

