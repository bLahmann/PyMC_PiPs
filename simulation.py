import functools
import multiprocessing
import time

import numpy
import numpy as np
import multiprocessing
from sampling import make_sampler, sample_directions
from boundaries import evaluate_legendre_modes, verify_boundaries, get_scale_lengths
from scipy.constants import physical_constants
from scipy.stats import norm
from nuclear_reaction import react_particles

from bosch_hale import DDn_reactivity as DDn
from endf import DTn_cross_section as DTn_xs
import matplotlib.pyplot as plotter
from stopping_power import li_petrasso

amu_to_g = physical_constants['atomic mass constant'][0] * 1000
amu_to_kg = physical_constants['atomic mass constant'][0]
amu_to_mev = physical_constants['atomic mass constant energy equivalent in MeV'][0]
c = physical_constants['speed of light in vacuum'][0] * 100.0  # cm/s
e_si = physical_constants['elementary charge'][0]         # C
e_cgs = physical_constants['elementary charge'][0] * 3e9  # statC
h_bar = physical_constants['reduced Planck constant'][0] * 100 * 100 * 1000  # cm^2 g / s


def run_simulation_multithread(species_masses,
                               species_charges,
                               temperature_profiles,
                               number_density_profiles,
                               boundaries,
                               source_radial_distribution,
                               num_source_particles_per_cpu,
                               source_masses,
                               source_charges,
                               stopping_power,
                               source_plasma_index=0,
                               source_particle_direction=None,
                               magnetic_field_strength=0.0,
                               secondary_reaction_masses=None,
                               secondary_reaction_cross_section=None,
                               secondary_reaction_direction=None,
                               secondary_energy_bins=None,
                               max_energy_fraction_loss_per_step=0.01,
                               step_size_scale_length_fraction=0.01,
                               min_energy_cutoff=0.1,
                               num_cpus=multiprocessing.cpu_count() - 1
                               ):
    manager = multiprocessing.Manager()

    processes = []
    source_escape_energies = manager.list()
    source_escape_weights = manager.list()
    secondary_birth_energies = manager.list()
    secondary_birth_weights = manager.list()
    seeds = np.random.randint(low=0, high=2 ** 32 - 1, size=num_cpus)

    # Start the processes
    for i in range(num_cpus):
        processes.append(multiprocessing.Process(target=_single_thread, args=(
            species_masses, species_charges, temperature_profiles, number_density_profiles, boundaries,
            source_radial_distribution, num_source_particles_per_cpu, source_masses, source_charges, stopping_power,
            source_plasma_index, source_particle_direction, magnetic_field_strength, secondary_reaction_masses,
            secondary_reaction_cross_section, secondary_reaction_direction, secondary_energy_bins,
            max_energy_fraction_loss_per_step, step_size_scale_length_fraction, min_energy_cutoff, seeds[i], num_cpus,
            source_escape_energies, source_escape_weights, secondary_birth_energies, secondary_birth_weights)
                                                 ))

        processes[i].start()

    # Join
    for i in range(num_cpus):
        processes[i].join()

    # todo If we binned the data, it needs to be summed
    if secondary_energy_bins is not None:

        index = range(0, len(secondary_energy_bins)-1)
        secondary_birth_energies = np.array(secondary_birth_energies)[index]
        temp_weights = np.zeros(secondary_birth_energies.size)
        for i in range(num_cpus):
            index = range(i * (len(secondary_energy_bins) - 1), (i+1) * (len(secondary_energy_bins) - 1))
            temp_weights += np.array(secondary_birth_weights)[index]

        secondary_birth_energies = secondary_birth_energies.tolist()
        secondary_birth_weights = temp_weights.tolist()

    return source_escape_energies, source_escape_weights, secondary_birth_energies, secondary_birth_weights


def _single_thread(species_masses,
                   species_charges,
                   temperature_profiles,
                   number_density_profiles,
                   boundaries,
                   source_radial_distribution,
                   num_source_particles_per_cpu,
                   source_masses,
                   source_charges,
                   stopping_power,
                   source_plasma_index=0,
                   source_particle_direction=None,
                   magnetic_field_strength=0.0,
                   secondary_reaction_masses=None,
                   secondary_reaction_cross_section=None,
                   secondary_reaction_direction=None,
                   secondary_energy_bins=None,
                   max_energy_fraction_loss_per_step=0.01,
                   step_size_scale_length_fraction=0.01,
                   min_energy_cutoff=0.1,
                   seed=0,
                   num_cpus=1,
                   mp_source_escape_energies=None,
                   mp_source_escape_weights=None,
                   mp_secondary_birth_energies=None,
                   mp_secondary_birth_weights=None
                   ):
    source_escape_energies, source_escape_weights, secondary_birth_energies, secondary_birth_weights = \
        run_simulation(species_masses, species_charges, temperature_profiles, number_density_profiles, boundaries,
                       source_radial_distribution, num_source_particles_per_cpu, source_masses, source_charges,
                       stopping_power, source_plasma_index, source_particle_direction, magnetic_field_strength,
                       secondary_reaction_masses, secondary_reaction_cross_section, secondary_reaction_direction,
                       secondary_energy_bins, max_energy_fraction_loss_per_step, step_size_scale_length_fraction,
                       min_energy_cutoff, seed, num_cpus)

    mp_source_escape_energies += source_escape_energies
    mp_source_escape_weights += source_escape_weights
    mp_secondary_birth_energies += secondary_birth_energies
    mp_secondary_birth_weights += secondary_birth_weights


# Todo multithread
def run_simulation(species_masses,
                   species_charges,
                   temperature_profiles,
                   number_density_profiles,
                   boundaries,
                   source_radial_distribution,
                   num_source_particles_per_cpu,
                   source_masses,
                   source_charges,
                   stopping_power,
                   source_plasma_index=0,
                   source_particle_direction=None,
                   magnetic_field_strength=0.0,
                   secondary_reaction_masses=None,
                   secondary_reaction_cross_section=None,
                   secondary_reaction_direction=None,
                   secondary_energy_bins=None,
                   max_energy_fraction_loss_per_step=0.01,
                   step_size_scale_length_fraction=0.01,
                   min_energy_cutoff=0.1,
                   seed=0,
                   num_cpus=1
                   ):
    """ Primary function that runs the simulation

    SPECIES INPUTS: All species inputs are list of arrays/tuples. Each item in the list refers to a plasma layer and
                    each element in the array/tuple refers to a species in that layer. Layers must be ordered from the
                    innermost plasma to the outermost plasma. All species inputs must be ordered in the same way for
                    proper performance. Note that electrons are not automatically included and must be explicitly
                    defined.

        species_masses - All the species' masses in amu

        species_charges - All the species' charges in elementary charge number

        temperature_profiles - Function handles for the species' temperature profiles defined between [0, 1].

        number_density_profiles - Function handles for the species' number density profiles defined between [0, 1].

    boundaries - List of lists of tuples that represent the boundaries of each plasma layer. Each item in the outer list
                 refers to a plasma layer. Each element in the inner lists refers to one legendre mode that defines the
                 boundary. Each tuple should be formatted as (m, ell, mode_magnitude). Boundaries must be constructed
                 in such a way that they don't go negative and don't cross other plasma layer boundaries. This is
                 verified in the code.

    SOURCE INPUTS:

        source_radial_distribution - Function handle the defines the pdf of the source particles radial distribution
                                     defined between [0, 1]. Note that the r^2 dependence is not explicitly added and
                                     must be included (if desired)

        num_source_particles - Number of source particles to simulate

        source_masses - Tuple of 4 masses in amu that define the two-body nuclear reaction that creates the source
                        particle. The order is (mA, mB, mC, mD) corresponding to the reaction A+B->C+D. The source
                        particle is assumed to be the C particle.

        source_charges

        stopping_power,

        source_plasma_index=0,

        source_particle_direction=None,

        secondary_reaction_masses=None,

        secondary_reaction_cross_section=None,

        max_energy_fraction_loss_per_step=0.005,

        step_size_scale_length_fraction=0.01,

        min_energy_cutoff=0.1

    """

    # Add an inner boundary that's always 0
    boundaries.insert(0, [(0, 0, 0.0)])

    # Verify the boundaries
    verify_boundaries(boundaries)

    # Get scale lengths
    scale_lengths = get_scale_lengths(boundaries)

    # Set the random seed
    np.random.seed(seed)

    # Create the position vectors
    r_norms, pos_vectors, plasma_indexes = \
        _sample_positions(source_radial_distribution, num_source_particles_per_cpu, boundaries, source_plasma_index)

    # Create the direction vectors
    if source_particle_direction is None:
        dir_vectors = sample_directions(num_source_particles_per_cpu)
    else:
        dir_vectors = sample_directions(num_source_particles_per_cpu, fixed_theta=source_particle_direction[0],
                                        fixed_phi=source_particle_direction[1])

    # Sample the energies
    energies = _sample_energies(r_norms, species_masses, temperature_profiles,
                                source_masses, num_source_particles_per_cpu, source_plasma_index)

    # Init the source weights
    source_weights = (1. / (num_cpus * num_source_particles_per_cpu)) * np.ones(energies.shape)

    # If the user provided bins, we'll histogram the data to save time
    if secondary_energy_bins is not None:
        secondary_birth_energies = 0.5 * (secondary_energy_bins[0:-1] + secondary_energy_bins[1:])
        secondary_birth_weights = np.zeros(secondary_birth_energies.size)

    # Otherwise, we'll just return a list of everything
    else:
        secondary_birth_energies = []
        secondary_birth_weights = []

    # Init the return variables (need to be lists for multiprocessing to return them)
    source_escape_energies = []
    source_escape_weights = []
    num_escaped = 0
    num_died = 0
    while not energies.size == 0:

        # todo some kind of multi-threaded progress bar

        # Loop through each plasma
        for i in range(len(species_masses)):

            # Make a mask for particles in this plasma
            mask = plasma_indexes == i

            # Evaluate all the profiles for dEdx
            field_densities = []
            field_temperatures = []
            for density_profile, temperature_profile in zip(number_density_profiles[i], temperature_profiles[i]):
                field_densities.append(density_profile(r_norms[mask]))
                field_temperatures.append(temperature_profile(r_norms[mask]))

            # Compute dEdx (assumes source particle is particle C)
            dE_dx = -stopping_power(species_masses[i], species_charges[i], field_densities, field_temperatures,
                                    source_masses[2], source_charges[2], energies[mask])[0]

            # Get the velocities (assumes source particle is particle C)
            speeds = np.sqrt(2. * energies[mask] / source_masses[2] / amu_to_mev)           # Fraction of c
            velocities = dir_vectors[mask] * speeds[:, np.newaxis]                          # Fraction of c

            # Determine a step size
            dx_scale_length = scale_lengths[i] * step_size_scale_length_fraction
            dx_energy_loss = max_energy_fraction_loss_per_step * energies[mask] / dE_dx
            dx = np.minimum(dx_scale_length, dx_energy_loss)

            # Calculate energy loss
            dE = dE_dx * dx

            # Handle the secondary reaction
            if secondary_reaction_masses is not None and secondary_reaction_cross_section is not None:

                # Verify that the background species is in this plasma
                if secondary_reaction_masses[0] in species_masses[i]:
                    # Rename the masses we'll use for clarity
                    m_bg = secondary_reaction_masses[0]  # Background particle is assumed to be index 0
                    m_s = secondary_reaction_masses[1]  # Source particle is assumed to be index 1

                    # Identify which species is the background species
                    background_index = species_masses[i].index(m_bg)

                    # Get the background number density
                    n_bg = number_density_profiles[i][background_index](r_norms[mask])

                    # Get the background velocities
                    kT = 1e-3 * temperature_profiles[i][background_index](r_norms[mask])  # T in MeV
                    sigma = np.sqrt(kT / m_bg / amu_to_mev)

                    v_x_bg = norm.rvs(loc=0.0, scale=sigma, size=len(kT))
                    v_y_bg = norm.rvs(loc=0.0, scale=sigma, size=len(kT))
                    v_z_bg = norm.rvs(loc=0.0, scale=sigma, size=len(kT))

                    velocity_background = np.array([v_x_bg.T, v_y_bg.T, v_z_bg.T]).T
                    speed_background = np.linalg.norm(velocity_background, axis=1)
                    energy_background = 0.5 * m_bg * amu_to_mev * speed_background ** 2

                    # Get the center of mass energy
                    energy_cm = energies * (m_bg / (m_bg + m_s)) + \
                                energy_background * (m_s / (m_bg + m_s))

                    # Calculate the probability of a reaction this step
                    sigma = 1e-24 * secondary_reaction_cross_section(energy_cm)  # Per cm^2
                    secondary_reaction_prob = (1.0 - np.exp(-n_bg * sigma * dx)) * source_weights[mask]
                    source_weights[mask] -= secondary_reaction_prob

                    secondary_dir_vectors, secondary_energies, secondary_biases = \
                        react_particles(secondary_reaction_masses, velocity_background,
                                        velocities, secondary_reaction_direction)

                    if secondary_energy_bins is not None:
                        hist = numpy.histogram(secondary_energies, weights=secondary_biases*secondary_reaction_prob,
                                               bins=secondary_energy_bins)[0]
                        secondary_birth_weights += hist
                    else:
                        secondary_birth_energies += secondary_energies.tolist()
                        secondary_birth_weights += (secondary_reaction_prob * secondary_biases).tolist()


            # Take a step
            pos_vectors[mask] += dir_vectors[mask] * dx[:, np.newaxis]
            energies[mask] -= dE

            q = e_si * source_charges[2]            # C
            m = source_masses[2] * amu_to_kg        # kg
            B = magnetic_field_strength             # T
            velocities[mask, 0] += dir_vectors[mask, 1] * B * q * dx / (m * c)
            velocities[mask, 1] -= dir_vectors[mask, 0] * B * q * dx / (m * c)

            dir_vectors[mask] = velocities[mask] / np.linalg.norm(velocities[mask], axis=1)[:, np.newaxis]

        # Determine what plasma we're in and update r_norms
        plasma_indexes.fill(-1)
        for i in range(1, len(boundaries)):
            inner_radius = functools.partial(evaluate_legendre_modes, boundaries[i - 1])
            outer_radius = functools.partial(evaluate_legendre_modes, boundaries[i])

            r = np.linalg.norm(pos_vectors, axis=1)
            theta = np.arccos(pos_vectors[:, 2] / r)
            phi = np.arctan2(pos_vectors[:, 1], pos_vectors[:, 0])
            phi[phi < 0] += 2 * np.pi

            inner_mask = r >= inner_radius(theta, phi)
            outer_mask = r < outer_radius(theta, phi)
            mask = inner_mask * outer_mask

            plasma_indexes[mask] = i - 1
            r_norms[mask] = (r[mask] - inner_radius(theta[mask], phi[mask])) / \
                            (outer_radius(theta[mask], phi[mask]) - inner_radius(theta[mask], phi[mask]))

        # Tally then delete any particle that is no longer in the system
        mask = plasma_indexes == -1
        source_escape_energies += energies[mask].tolist()
        source_escape_weights += source_weights[mask].tolist()
        num_escaped += np.sum(mask)

        r_norms = np.delete(r_norms, mask)
        pos_vectors = np.delete(pos_vectors, mask, axis=0)
        dir_vectors = np.delete(dir_vectors, mask, axis=0)
        energies = np.delete(energies, mask)
        plasma_indexes = np.delete(plasma_indexes, mask)
        source_weights = np.delete(source_weights, mask)

        # Kill any particle below the energy cutoff
        mask = energies < min_energy_cutoff
        num_died += np.sum(mask)

        r_norms = np.delete(r_norms, mask)
        pos_vectors = np.delete(pos_vectors, mask, axis=0)
        dir_vectors = np.delete(dir_vectors, mask, axis=0)
        energies = np.delete(energies, mask)
        plasma_indexes = np.delete(plasma_indexes, mask)
        source_weights = np.delete(source_weights, mask)

    # If we binned the data, we need to turn it into a list before returning
    if secondary_energy_bins is not None:
        secondary_birth_energies = secondary_birth_energies.tolist()
        secondary_birth_weights = secondary_birth_weights.tolist()

    return source_escape_energies, source_escape_weights, secondary_birth_energies, secondary_birth_weights


def _sample_positions(source_radial_distribution, num_source_particles, boundaries, source_plasma_index):
    # Sample the source theta and phi
    inner_radius = functools.partial(evaluate_legendre_modes, boundaries[source_plasma_index])
    outer_radius = functools.partial(evaluate_legendre_modes, boundaries[source_plasma_index + 1])
    plasma_length = lambda theta, phi: outer_radius(theta, phi) - inner_radius(theta, phi)

    theta_pos_dist = lambda theta: plasma_length(theta, np.array([0.0])) * np.sin(theta)
    theta_pos = make_sampler(theta_pos_dist, 0.0, np.pi)(num_source_particles)
    phi_pos_dist = lambda phi: plasma_length(np.array([np.pi / 2.]), phi)
    phi_pos = make_sampler(phi_pos_dist, 0.0, 2. * np.pi)(num_source_particles)

    # Sample the source radial distribution
    r_norms = make_sampler(source_radial_distribution, 0, 1)(num_source_particles)
    r_pos = r_norms * plasma_length(theta_pos, phi_pos) + inner_radius(theta_pos, phi_pos)

    # Create position vectors
    x_pos = r_pos * np.sin(theta_pos) * np.cos(phi_pos)
    y_pos = r_pos * np.sin(theta_pos) * np.sin(phi_pos)
    z_pos = r_pos * np.cos(theta_pos)
    pos_vectors = np.array([x_pos, y_pos, z_pos]).T

    # Create the location indexes
    plasma_indexes = source_plasma_index * np.ones(r_norms.shape)

    return r_norms, pos_vectors, plasma_indexes


def _sample_energies(r_norms, species_masses, temperature_profiles, source_nuclear_reaction_masses,
                     num_source_particles, source_plasma_index):
    # Unpack the source masses to construct the energy distribution
    m1 = source_nuclear_reaction_masses[0]
    m2 = source_nuclear_reaction_masses[1]
    m3 = source_nuclear_reaction_masses[2]
    m4 = source_nuclear_reaction_masses[3]

    # Calculate some constants
    mr_c2 = 1000 * amu_to_mev * m1 * m2 / (m1 + m2)  # Reduced mass (keV)
    q = 1000 * amu_to_mev * (m1 + m2 - m3 - m4)  # Fusion Q (keV)

    # Figure out where those masses are in the profile list
    index1 = species_masses[source_plasma_index].index(m1)
    index2 = species_masses[source_plasma_index].index(m2)

    # Determine the temperature at each particle location
    T = 0.5 * (temperature_profiles[source_plasma_index][index1](r_norms) +
               temperature_profiles[source_plasma_index][index2](r_norms))

    # Calculate temperature dependent parameters
    k = (np.pi ** 2 * e_cgs ** 4 * mr_c2 / (2 * h_bar ** 2 * c ** 2)) ** (1. / 3.) * T ** (2. / 3.) + (5. / 6.) * T  # keV
    v2 = 3.0 * T / (m1 + m2)

    # Get the means and the sigmas
    mu_brisk = 1e-3 * (0.5 * m3 * v2 + m4 * (q + k) / (m3 + m4))  # MeV
    sig_brisk = 1e-3 * np.sqrt(2.0 * m3 * T * mu_brisk / (m3 + m4))  # MeV

    # Sample the energies
    energies = norm.rvs(loc=mu_brisk, scale=sig_brisk, size=num_source_particles)

    return energies


def _construct_stopping_power():
    pass


if __name__ == "__main__":
    from utils import get_prav_profiles, construct_radial_distribution

    m_e = physical_constants['electron mass in u'][0]
    m_n = physical_constants['neutron mass in u'][0]
    m_p = physical_constants['proton mass in u'][0]
    m_D = physical_constants['deuteron mass in u'][0]
    m_T = physical_constants['triton mass in u'][0]
    m_a = physical_constants['alpha particle mass in u'][0]
    m_3He = 3.0160293 - 2 * physical_constants['electron mass in u'][0]

    # Yield Ratio Plot
    rho = 10.0      # g/cc
    T = 4.0         # keV
    n_D = rho / m_D / amu_to_g
    r = np.logspace(0, 4, 101)
    r_x_norm = r * np.sqrt(4.0 * np.pi)
    rho_r = 0.1 * rho * r

    # Uniform profiles
    density_profile = lambda x: n_D * np.ones(x.shape)
    temperature_profile = lambda x: T * np.ones(x.shape)
    source_radial_distribution = construct_radial_distribution(
        [temperature_profile, temperature_profile], [density_profile, density_profile], DDn)

    # Loop through the rhoRs
    yield_ratio = []
    for r in r_x_norm:
        source_escape_energies, source_escape_weights, secondary_birth_energies, secondary_birth_weights = \
            run_simulation_multithread(
                species_masses=[
                    (m_e, m_D)
                ],
                species_charges=[
                    (-1, 1)
                ],
                temperature_profiles=[
                    (temperature_profile, temperature_profile)
                ],
                number_density_profiles=[
                    (density_profile, density_profile)
                ],
                boundaries=[
                    [(0, 0, 1e-4 * 150.0)]
                ],
                source_radial_distribution=source_radial_distribution,
                num_source_particles_per_cpu=10000,
                source_masses=(m_D, m_D, m_T, m_p),
                source_charges=(1, 1, 1, 1),
                source_particle_direction=None,  # np.deg2rad((90, 78)),
                magnetic_field_strength=6000.0,
                stopping_power=li_petrasso,
                secondary_reaction_masses=(m_D, m_T, m_n, m_a),
                secondary_reaction_cross_section=DTn_xs,
                secondary_reaction_direction=np.deg2rad((90, 78)),
                secondary_energy_bins=np.linspace(11, 18, 101)
            )

        yield_ratio.append(np.sum(np.array(secondary_birth_weights)))

        fig = plotter.figure()
        ax = fig.add_subplot()
        plotter.plot(secondary_birth_energies, secondary_birth_weights)
        plotter.show()

    fig = plotter.figure()
    ax = fig.add_subplot()
    ax.set_yscale("log")
    ax.set_xscale("log")

    plotter.plot(rho_r, yield_ratio)
    plotter.show()


