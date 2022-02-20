import functools

import numpy
import numpy as np
from sampling import make_sampler, sample_directions
from boundaries import evaluate_legendre_modes, verify_boundaries, get_scale_lengths
from scipy.constants import physical_constants
from scipy.stats import norm
from nuclear_reaction import react_particles

from bosch_hale import DDn_reactivity as DDn
from endf import DTn_cross_section as DTn_xs
import matplotlib.pyplot as plotter
from stopping_power import li_petrasso

amu_to_mev = physical_constants['atomic mass constant energy equivalent in MeV'][0]
c = physical_constants['speed of light in vacuum'][0] * 100.0                           # cm/s
e = physical_constants['elementary charge'][0] * 3e9                                    # statC
h_bar = physical_constants['reduced Planck constant'][0] * 100 * 100 * 1000             # cm^2 g / s


# Todo multithread
def run_simulation(species_masses,
                   species_charges,
                   temperature_profiles,
                   number_density_profiles,
                   boundaries,
                   source_radial_distribution,
                   num_source_particles,
                   source_masses,
                   source_charges,
                   stopping_power,
                   source_plasma_index=0,
                   source_particle_direction=None,
                   secondary_reaction_masses=None,
                   secondary_reaction_cross_section=None,
                   secondary_reaction_direction=None,
                   max_energy_fraction_loss_per_step=0.005,
                   step_size_scale_length_fraction=0.01,
                   min_energy_cutoff=0.1
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

    # Create the position vectors
    r_norms, pos_vectors, plasma_indexes = \
        _sample_positions(source_radial_distribution, num_source_particles, boundaries, source_plasma_index)

    # Create the direction vectors
    if source_particle_direction is None:
        dir_vectors = sample_directions(num_source_particles)
    else:
        dir_vectors = sample_directions(num_source_particles, fixed_theta=source_particle_direction[0],
                                        fixed_phi=source_particle_direction[1])

    # Sample the energies
    energies = _sample_energies(r_norms, species_masses, temperature_profiles,
                                source_masses, num_source_particles, source_plasma_index)

    # Print status update title (todo this won't work when we multi-thread)
    print("\n--- SOURCE PARTICLE BUDGET ---")

    secondary_reaction_probability = np.zeros(energies.shape)
    source_escape_energies = np.array([])
    secondary_birth_energies = np.array([])
    num_escaped = 0
    num_died = 0
    while not energies.size == 0:

        # Print status update
        debug_string = ""
        for i in range(len(species_masses)):
            debug_string += "Plasma Layer %d: %d " % (i+1, sum(plasma_indexes == i))
        debug_string += "Escaped: %d Dead %d " % (num_escaped, num_died)
        print("\r", end="")
        print(debug_string, end="")

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

            # Determine a step size
            dx_scale_length = scale_lengths[i] * step_size_scale_length_fraction
            dx_energy_loss = max_energy_fraction_loss_per_step * energies[mask] / dE_dx
            dx = numpy.minimum(dx_scale_length, dx_energy_loss)

            # Calculate energy loss
            dE = dE_dx * dx

            # Todo calculate secondary particle energy, probability
            # todo get the cross section at the conditions corresponding to the species making the reaction
            if secondary_reaction_masses is not None and secondary_reaction_cross_section is not None:

                # Verify that the background species is in this plasma
                if secondary_reaction_masses[0] in species_masses[i]:

                    # Identify which species is the background species
                    background_index = species_masses[i].index(secondary_reaction_masses[0])

                    # Get the background velocities
                    kT = 1e-3 * temperature_profiles[i][background_index](r_norms[mask])            # T in MeV
                    sigma = np.sqrt(kT / secondary_reaction_masses[0] / amu_to_mev)

                    v_x_bg = norm.rvs(loc=0.0, scale=sigma, size=len(kT))
                    v_y_bg = norm.rvs(loc=0.0, scale=sigma, size=len(kT))
                    v_z_bg = norm.rvs(loc=0.0, scale=sigma, size=len(kT))

                    velocity_background = np.array([v_x_bg.T, v_y_bg.T, v_z_bg.T]).T

                    # Get the source velocities (assumes source particle is particle C)
                    speed_source = np.sqrt(2.*energies[mask]/source_masses[2]/amu_to_mev)
                    velocity_source = dir_vectors[mask] * speed_source[:, np.newaxis]

                    secondary_dir_vectors, secondary_energies = \
                        react_particles(secondary_reaction_masses, velocity_background,
                                        velocity_source, secondary_reaction_direction)

                    secondary_birth_energies = np.append(secondary_birth_energies, secondary_energies)

            # Take a step
            pos_vectors[mask] += dir_vectors[mask] * dx[:, np.newaxis]
            energies[mask] -= dE

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

            plasma_indexes[mask] = i-1
            r_norms[mask] = (r[mask] - inner_radius(theta[mask], phi[mask])) / \
                            (outer_radius(theta[mask], phi[mask]) - inner_radius(theta[mask], phi[mask]))

        # Tally then delete any particle that is no longer in the system
        mask = plasma_indexes == -1
        source_escape_energies = np.append(source_escape_energies, energies[mask])
        num_escaped += np.sum(mask)

        r_norms = np.delete(r_norms, mask)
        pos_vectors = np.delete(pos_vectors, mask, axis=0)
        dir_vectors = np.delete(dir_vectors, mask, axis=0)
        energies = np.delete(energies, mask)
        plasma_indexes = np.delete(plasma_indexes, mask)

        # Kill any particle below the energy cutoff
        mask = energies < min_energy_cutoff
        num_died += np.sum(mask)

        r_norms = np.delete(r_norms, mask)
        pos_vectors = np.delete(pos_vectors, mask, axis=0)
        dir_vectors = np.delete(dir_vectors, mask, axis=0)
        energies = np.delete(energies, mask)
        plasma_indexes = np.delete(plasma_indexes, mask)

    fig = plotter.figure()
    ax = fig.add_subplot()

    plotter.hist(secondary_birth_energies, np.linspace(np.min(secondary_birth_energies), np.max(secondary_birth_energies), 1000))
    plotter.show()

    return source_escape_energies, secondary_birth_energies


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
    mr_c2 = 1000 * amu_to_mev * m1 * m2 / (m1 + m2)         # Reduced mass (keV)
    q = 1000 * amu_to_mev * (m1 + m2 - m3 - m4)             # Fusion Q (keV)

    # Figure out where those masses are in the profile list
    index1 = species_masses[source_plasma_index].index(m1)
    index2 = species_masses[source_plasma_index].index(m2)

    # Determine the temperature at each particle location
    T = 0.5 * (temperature_profiles[source_plasma_index][index1](r_norms) +
               temperature_profiles[source_plasma_index][index2](r_norms))

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
    ma = physical_constants['alpha particle mass in u'][0]
    m3He = 3.0160293 - 2*physical_constants['electron mass in u'][0]

    density = lambda x: 1e26 * np.ones(x.shape)
    temperature = get_prav_profiles(5.0, 0.2, 1.0)

    source_radial_distribution = construct_radial_distribution([temperature, temperature], [density, density], DDn)

    run_simulation(
        species_masses=[
            (me, mD),
            (me, mT)
        ],
        species_charges=[
            (-1, 1),
            (-1, 1)
        ],
        temperature_profiles=[
            (temperature, temperature),
            (temperature, temperature),
        ],
        number_density_profiles=[
            (density, density),
            (density, density),
        ],
        boundaries=[
            [(0, 0, 1e-4 * 50.0)],
            [(0, 0, 1e-4 * 150.0)]
        ],
        source_radial_distribution=source_radial_distribution,
        num_source_particles=10000,
        source_masses=(mD, mD, mT, mp),
        source_charges=(1, 1, 1, 1),
        source_particle_direction=np.deg2rad((90, 78)),
        stopping_power=li_petrasso,
        secondary_reaction_masses=(mD, mT, mn, ma),
        secondary_reaction_cross_section=DTn_xs,
    )



