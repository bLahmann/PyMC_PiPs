import functools
import numpy as np
from sampling import make_sampler
from boundaries import evaluate_legendre_modes, verify_boundaries
from scipy.constants import physical_constants
from scipy.stats import norm

from bosch_hale import DDn_reactivity as DDn
import matplotlib.pyplot as plotter

amu_to_mev = physical_constants['atomic mass constant energy equivalent in MeV'][0]
c = physical_constants['speed of light in vacuum'][0] * 100.0                           # cm/s
e = physical_constants['elementary charge'][0] * 3e9                                    # statC
h_bar = physical_constants['reduced Planck constant'][0] * 100 * 100 * 1000             # cm^2 g / s


def run_simulation(species_masses, species_charges, temperature_profiles, number_density_profiles,
                   source_radial_distribution, num_source_particles, source_nuclear_reaction_masses,
                   boundaries,
                   source_plasma_index=0
                   ):

    # Add an inner boundary that's always 0
    boundaries.insert(0, [(0, 0, 0.0)])

    # Verify the boundaries
    verify_boundaries(boundaries)

    # Create the position vectors
    r_norms, pos_vectors = \
        _sample_positions(source_radial_distribution, num_source_particles, boundaries, source_plasma_index)

    # Create the direction vectors
    dir_vectors = _sample_directions(num_source_particles)

    # Sample the energies
    energies = _sample_energies(
        r_norms, species_masses, temperature_profiles, source_nuclear_reaction_masses, num_source_particles)


    ax = plotter.figure().add_subplot(projection="3d")
    ax.quiver(pos_vectors[:, 0], pos_vectors[:, 1], pos_vectors[:, 2], dir_vectors[:, 0], dir_vectors[:, 1], dir_vectors[:, 2])
    plotter.show()


def _sample_positions(source_radial_distribution, num_source_particles, boundaries, source_plasma_index):

    # Sample the source theta and phi
    inner_radius = functools.partial(evaluate_legendre_modes, boundaries[source_plasma_index])
    outer_radius = functools.partial(evaluate_legendre_modes, boundaries[source_plasma_index + 1])
    plasma_length = lambda theta, phi: outer_radius(theta, phi) - inner_radius(theta, phi)

    theta_pos_dist = lambda theta: plasma_length(theta, np.array([0.0])) * np.sin(theta)
    theta_pos = make_sampler(theta_pos_dist, 0.0, np.pi)(num_source_particles)
    phi_pos_dist = lambda phi: plasma_length(np.array([np.pi/2.]), phi)
    phi_pos = make_sampler(phi_pos_dist, 0.0, 2.*np.pi)(num_source_particles)

    # Sample the source radial distribution
    r_norms = make_sampler(source_radial_distribution, 0, 1)(num_source_particles)
    r_pos = r_norms * plasma_length(theta_pos, phi_pos) + inner_radius(theta_pos, phi_pos)

    # Create position vectors
    x_pos = r_pos * np.sin(theta_pos) * np.cos(phi_pos)
    y_pos = r_pos * np.sin(theta_pos) * np.sin(phi_pos)
    z_pos = r_pos * np.cos(theta_pos)
    pos_vectors = np.array([x_pos, y_pos, z_pos]).T

    return r_norms, pos_vectors


def _sample_directions(num_source_particles):

    theta_dir_dist = lambda theta: np.sin(theta)
    theta_dir = make_sampler(theta_dir_dist, 0.0, np.pi)(num_source_particles)
    phi_dir_dist = lambda phi: np.ones(phi.shape)
    phi_dir = make_sampler(phi_dir_dist, 0.0, 2.*np.pi)(num_source_particles)

    x_dir = np.sin(theta_dir) * np.cos(phi_dir)
    y_dir = np.sin(theta_dir) * np.sin(phi_dir)
    z_dir = np.cos(theta_dir)
    dir_vectors = np.array([x_dir, y_dir, z_dir]).T

    return dir_vectors


def _sample_energies(
        r_norms, species_masses, temperature_profiles, source_nuclear_reaction_masses, num_source_particles):

    # Unpack the source masses to construct the energy distribution
    m1 = source_nuclear_reaction_masses[0]
    m2 = source_nuclear_reaction_masses[1]
    m3 = source_nuclear_reaction_masses[2]
    m4 = source_nuclear_reaction_masses[3]

    # Calculate some constants
    mr_c2 = 1000 * amu_to_mev * m1 * m2 / (m1 + m2)         # Reduced mass (keV)
    q = 1000 * amu_to_mev * (m1 + m2 - m3 - m4)             # Fusion Q (keV)

    # Figure out where those masses are in the profile list
    index1 = species_masses.index(m1)
    index2 = species_masses.index(m1)

    # Determine the temperature at each particle location
    T = 0.5*(temperature_profiles[index1](r_norms) + temperature_profiles[index2](r_norms))

    # Calculate temperature dependent parameters
    k = (np.pi ** 2 * e ** 4 * mr_c2 / (2 * h_bar ** 2 * c ** 2)) ** (1. / 3.) * T ** (2. / 3.) + (5. / 6.) * T  # keV
    v2 = 3.0 * T / (m1 + m2)

    # Get the means and the sigmas
    mu_brisk = 1e-3 * (0.5 * m3 * v2 + m4 * (q + k) / (m3 + m4))        # MeV
    sig_brisk = 1e-3 * np.sqrt(2.0 * m3 * T * mu_brisk / (m3 + m4))     # MeV

    # Sample the energies
    energies = norm.rvs(loc=mu_brisk, scale=sig_brisk, size=num_source_particles)

    return energies


if __name__ == "__main__":

    from utils import get_prav_profiles, construct_radial_distribution

    me = physical_constants['electron mass in u'][0]
    mn = physical_constants['neutron mass in u'][0]
    mp = physical_constants['proton mass in u'][0]
    mD = physical_constants['deuteron mass in u'][0]
    mT = physical_constants['triton mass in u'][0]

    species_masses = (me, mD)
    species_charges = (-1, 1)
    source_masses = (mD, mD, mT, mp)

    boundaries = [
        [(0, 0, 50.0), (0, 2, 40.0), (1, 2, 10.0)]
    ]

    density = lambda x: 5.0 * np.ones(x.shape)
    temperature = get_prav_profiles(5.0, 0.2, 1.0)

    source_radial_distribution = construct_radial_distribution([temperature, temperature], [density, density], DDn)

    run_simulation(species_masses, species_charges, (temperature, temperature), (density, density),
                   source_radial_distribution, 1000000, source_masses, boundaries)


    """
    outer_legendre_modes = [
        (0, 0, 50.0),
        (0, 2, 30.0)
    ]
    
    inner_legendre_modes = [
        (0, 0, 10.0)
    ]
    
    outer_radius = functools.partial(evaluate_legendre_modes, outer_legendre_modes)
    inner_radius = functools.partial(evaluate_legendre_modes, inner_legendre_modes)
    length = lambda theta, phi: outer_radius(theta, phi) - inner_radius(theta, phi)
    theta = np.linspace(0, np.pi, 1000)
    
    fig = plotter.figure()
    ax = fig.add_subplot()
    print(inner_radius(theta, 0.0))
    plotter.plot(theta, length(theta, 0.0))
    plotter.show()
    """

    """
    
    
    sampler = make_sampler(radial_distribution, 0, 1)
    x_s = sampler(1000000)
    
    fig = plotter.figure()
    ax = fig.add_subplot()
    plotter.hist(x_s, bins=np.linspace(0, 1, 1000))
    plotter.show()
    """

