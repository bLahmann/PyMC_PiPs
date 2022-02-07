import numpy
from numpy import exp, sqrt, log

from scipy.constants import physical_constants
from scipy.constants import elementary_charge as e
from scipy.constants import speed_of_light as c
from scipy.constants import pi
from scipy.constants import hbar


from scipy.special import erf

amu_to_MeV = physical_constants["atomic mass constant energy equivalent in MeV"][0]


def bethe_bloch():
    pass


def li_petrasso(field_masses, field_charges, field_densities, field_temperatures, test_mass, test_charge, test_energy):

    # Rename test inputs
    m_t = test_mass
    z_t = test_charge
    E_t = test_energy

    m_t_c2 = m_t * amu_to_MeV                   # Test mass in MeV per c^2
    ze_t = z_t * e                              # Test charge in C
    v_t = c * sqrt(2.0 * E_t / m_t_c2)    # Test velocity

    # Calculate the total ion debye length
    length_sum = 0.0
    for z_f, n_f, kT_f in zip(field_charges, field_densities, field_temperatures):
        ze_f = z_f * e
        length_sum += 4 * pi * ze_f ** 2 / kT_f
    total_ion_debye_length = 1.0 / sqrt(length_sum)

    # Calculate dEdx for each field
    dE_dx = 0.0
    for m_f, z_f, n_f, kT_f in zip(field_masses, field_charges, field_densities, field_temperatures):

        electron_field = z_f < 0
        ion_field = z_f > 0

        # Quantum correction to Te
        if electron_field:
            kT_fermi = hbar ** 2 / (2 * m_f) * pow(3 * pi ** 2 * n_f, 2./3.)
            theta = kT_f / kT_fermi
            mu_e = -1.5 * log(theta) + log(4 / (3 * sqrt(pi))) + \
                (0.25054 * pow(theta, -1.858) + 0.072 * pow(theta, -1.858/2))


        ze_f = z_f * e                                              # Field charge in C
        w_pf = sqrt(4 * pi * n_f * ze_f ** 2 / m_f)                 # Plasma frequency
        x_tf = v_t ** 2 * m_f / (2 * kT_f)                          # Test to field velocity ratio
        lambda_df = sqrt(kT_f / (4 * pi * n_f * ze_f ** 2))         # Field Debye length

        u = sqrt(2 * kT_f / (pi * m_f)) * exp(- m_f * v_t ** 2 / (2 * kT_f)) + \
            v_t * (1 + kT_f / (m_f * v_t ** 2)) * erf(sqrt(m_f * v_t ** 2 / (2 * kT_f)))        # Relative velocity

        m_r = m_f * m_t / (m_f + m_t)                               # Reduced mass
        p_perp = ze_f * ze_t / (m_r * u ** 2)                       # Classical impact parameter
        p_min = sqrt(p_perp ** 2 + (hbar / (2 * m_r * u)) ** 2)     # Minimum impact parameter

        # The coulomb logarithm uses a different debye length for ions
        lambda_d = 0                                                # Debye length for Coulomb log calculation
        if electron_field:
            lambda_d = lambda_df                    # For electrons, we use the (electron) field debye length
        if ion_field:
            lambda_d = total_ion_debye_length       # For ions, we use the TOTAL ion debye length
        coulomb_log = 0.5 * log(1 + (lambda_d / p_min) ** 2)




        dmu_dx = sqrt(x_tf) * exp(-x_tf)
        mu = sqrt(pi) * erf(sqrt(x_tf)) / 2.0 - dmu_dx


        pass

    dE_dx *= (ze_t / v_t)


def bps():
    pass


def zimmerman():
    pass

