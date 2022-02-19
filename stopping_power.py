import numpy
from numpy import exp, sqrt, log

from scipy.constants import physical_constants
from scipy.constants import elementary_charge as e
from scipy.constants import speed_of_light as c
from scipy.constants import pi
from scipy.constants import hbar
from scipy.constants import eV as joules_per_ev, erg as joules_per_erg, kilo, mega

from fdint import fdk as fermi_integral
from scipy.special import erf
from scipy.special import kn
from scipy.special import gamma

import matplotlib.pyplot as plotter


hbar /= joules_per_erg      # J-s to erg-s
e *= 3e9                    # C to statC
c *= 1e2                    # m/s to cm/s

amu_to_grams = 1e3 * physical_constants["atomic mass unit-kilogram relationship"][0]
mev_to_erg = mega * joules_per_ev / joules_per_erg
kev_to_erg = kilo * joules_per_ev / joules_per_erg

m_d = physical_constants["deuteron mass in u"][0]
m_t = physical_constants["triton mass in u"][0]
m_e = physical_constants["electron mass in u"][0]
m_a = physical_constants["alpha particle mass in u"][0]


def bethe_bloch():
    pass


def li_petrasso(field_masses_amu, field_charges, field_densities_per_cm3, field_temperatures_kev,
                test_mass_amu, test_charge, test_energy_mev):

    # Rename test inputs
    m_t_grams = amu_to_grams * test_mass_amu              # Test mass (grams)
    ze_t = e * test_charge                          # Test charge (statcoulombs)
    E_t = mev_to_erg * test_energy_mev              # Test energy (erg)
    v_t = sqrt(2.0 * E_t / m_t_grams)                 # Test velocity (cm/s)

    # Calculate the total debye length
    length_sum = 0.0
    for z_f, n_f, kT_f_kev in zip(field_charges, field_densities_per_cm3, field_temperatures_kev):
        ze_f = z_f * e                                      # Field charge (statcoulombs)
        kT_f_erg = kT_f_kev * kev_to_erg                                  # Field temperature (ergs)
        length_sum += 4 * pi * n_f * ze_f ** 2 / kT_f_erg
    total_debye_length = 1.0 / sqrt(length_sum)             # Total Debye length

    # Calculate dEdx for each field
    de_dx = 0.0
    de_dx_species = []
    for m_f_amu, z_f, n_f, kT_f_kev in zip(field_masses_amu, field_charges, field_densities_per_cm3, field_temperatures_kev):

        is_electron_field = z_f < 0             # Electric field boolean
        is_ion_field = z_f > 0                  # Ion field boolean

        m_f_grams = m_f_amu * amu_to_grams                     # Field mass (grams)
        ze_f = z_f * e                          # Field charge (statcoulombs)
        kT_f_erg = kT_f_kev * kev_to_erg                      # Field temperature (ergs)

        # Quantum correction to Te
        if is_electron_field:
            kT_fermi = hbar ** 2 / (2 * m_f_grams) * pow(3 * pi ** 2 * n_f, 2./3.)            # Fermi temperature (ergs)
            theta = kT_f_erg / kT_fermi
            mu_e = -1.5 * log(theta) + log(4 / (3 * sqrt(pi))) + \
                (0.25054 * pow(theta, -1.858) + 0.072 * pow(theta, -1.858/2)) / \
                (1 + 0.25054 * pow(theta, -0.868))                                      # Equation (25)
            kT_f_erg *= fermi_integral(3/2, mu_e) / fermi_integral(1/2, mu_e)               # Equation (23)
            kT_f_erg *= gamma(3/2) / gamma(5/2)

        w_pf = sqrt(4 * pi * n_f * ze_f ** 2 / m_f_grams)                 # Plasma frequency
        lambda_df = sqrt(kT_f_erg / (4 * pi * n_f * ze_f ** 2))         # Field Debye length
        x_tf_col = v_t ** 2 * m_f_grams / (2 * kT_f_erg)                      # Test to field velocity ratio (collective term)
        binary_term = (w_pf * lambda_df) / v_t                      # Evaluation point for binary collisions

        u = sqrt(2 * kT_f_erg / (pi * m_f_grams)) * exp(- m_f_grams * v_t ** 2 / (2 * kT_f_erg)) + \
            v_t * (1 + kT_f_erg / (m_f_grams * v_t ** 2)) * erf(sqrt(m_f_grams * v_t ** 2 / (2 * kT_f_erg)))        # Relative velocity

        m_r = m_f_grams * m_t_grams / (m_f_grams + m_t_grams)                               # Reduced mass
        p_perp = ze_f * ze_t / (m_r * u ** 2)                       # Classical impact parameter
        p_min = sqrt(p_perp ** 2 + (hbar / (2 * m_r * u)) ** 2)     # Minimum impact parameter

        # The coulomb logarithm uses a different debye length for ions
        lambda_d = 0                                                # Debye length for Coulomb log calculation
        if is_electron_field:
            lambda_d = lambda_df                    # For electrons, we use the (electron) field debye length
        if is_ion_field:
            lambda_d = total_debye_length           # For ions, we use the TOTAL debye length
        coulomb_log = 0.5 * log(1 + (lambda_d / p_min) ** 2)

        dmu_dx = sqrt(x_tf_col) * exp(-x_tf_col)                                    # Equation (7)
        mu = sqrt(pi) * erf(sqrt(x_tf_col)) / 2.0 - dmu_dx                          # Equation (6)

        G = mu - (m_f_grams / m_t_grams) * (dmu_dx - (mu + dmu_dx) / coulomb_log)               # Equation (4)

        de_dx_species.append(-(ze_t / v_t) ** 2 * w_pf ** 2 * 
                        (G * coulomb_log + binary_term * kn(0, binary_term) * kn(1, binary_term)))
        de_dx_species[-1] /= mev_to_erg     # MeV/cm
        de_dx += de_dx_species[-1]

    return de_dx, de_dx_species         # MeV / cm



def bps():
    pass


def zimmerman():
    pass


if __name__ == "__main__":

    rho = 10.0
    m_dt = amu_to_grams * (m_d + m_t) / 2.
    ne = rho / m_dt

    field_masses = [m_d, m_t, m_e]
    field_charges = [1.0, 1.0, -1.0]
    field_densities = [ne/2., ne/2., ne]


    test_mass = m_a
    test_charge = 2.0

    energies = numpy.linspace(0.1, 20, 10000)

    field_temperatures = [0.1]*3
    dEdx_total, dEdx = li_petrasso(field_masses, field_charges, field_densities, field_temperatures, test_mass, test_charge, energies)

    fig = plotter.figure()
    ax = fig.add_subplot()
    ax.set_yscale("log")
    plotter.plot(energies, -dEdx_total / 10000)
    for y in dEdx:
        plotter.plot(energies, -y / 10000)
    plotter.ylim([1*10**-1, 1.5*10**0])
    plotter.ion()
    plotter.show()

    field_temperatures = [1.0]*3
    dEdx_total, dEdx = li_petrasso(field_masses, field_charges, field_densities, field_temperatures, test_mass, test_charge, energies)

    fig = plotter.figure()
    ax = fig.add_subplot()
    ax.set_yscale("log")
    plotter.plot(energies, -dEdx_total / 10000)
    for y in dEdx:
        plotter.plot(energies, -y / 10000)
    plotter.ylim([5*10**-2, 3*10**-1])
    plotter.ion()
    plotter.show()

    field_temperatures = [20.0]*3
    dEdx_total, dEdx = li_petrasso(field_masses, field_charges, field_densities, field_temperatures, test_mass, test_charge, energies)

    fig = plotter.figure()
    ax = fig.add_subplot()
    ax.set_xscale("log")
    ax.set_yscale("log")
    plotter.plot(energies, -dEdx_total / 10000)
    for y in dEdx:
        plotter.plot(energies, -y / 10000)
    plotter.ylim([2*10**-3, 2.2*10**-2])
    plotter.ioff()
    plotter.show()
