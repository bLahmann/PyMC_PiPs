import functools
import numpy as np
from sampling import make_sampler
from boundaries import evaluate_legendre_modes, verify_boundaries

from bosch_hale import DDn_reactivity as DDn

import matplotlib.pyplot as plotter

def run_simulation(species_masses, species_charges, temperature_profiles, number_density_profiles,
                   source_radial_distribution, num_source_particles, boundaries,
                   source_plasma_index=0
                   ):

    # Add an inner boundary that's always 0
    boundaries.insert(0, [(0, 0, 0.0)])

    # Verify the boundaries
    for i in range(1, len(boundaries)):
        inner_radius = functools.partial(evaluate_legendre_modes, boundaries[i - 1])
        outer_radius = functools.partial(evaluate_legendre_modes, boundaries[i])
        plasma_length = lambda theta, phi: outer_radius(theta, phi) - inner_radius(theta, phi)
        print("Testing plasma " + str(i))
        verify_boundaries(inner_radius, outer_radius, plasma_length)

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

    # Create the direction vectors
    theta_dir_dist = lambda theta: np.sin(theta)
    theta_dir = make_sampler(theta_dir_dist, 0.0, np.pi)(num_source_particles)
    phi_dir_dist = lambda phi: np.ones(phi.shape)
    phi_dir = make_sampler(phi_dir_dist, 0.0, 2.*np.pi)(num_source_particles)

    x_dir = np.sin(theta_dir) * np.cos(phi_dir)
    y_dir = np.sin(theta_dir) * np.sin(phi_dir)
    z_dir = np.cos(theta_dir)

    dir_vectors = np.array([x_dir, y_dir, z_dir]).T

    ax = plotter.figure().add_subplot(projection="3d")
    ax.quiver(pos_vectors[:, 0], pos_vectors[:, 1], pos_vectors[:, 2], dir_vectors[:, 0], dir_vectors[:, 1], dir_vectors[:, 2])
    plotter.show()

    pass

    # Sample the source directions





if __name__ == "__main__":
    from utils import get_prav_profiles, construct_radial_distribution


    boundaries = [
        [(0, 0, 50.0), (0, 2, 40.0), (1, 2, 10.0)]
    ]

    density = lambda x: 5.0 * np.ones(x.shape)
    temperature = get_prav_profiles(5.0, 0.2, 1.0)

    radial_distribution = construct_radial_distribution([temperature, temperature], [density, density], DDn)

    run_simulation(None, None, None, None, radial_distribution, 1000, boundaries)


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

