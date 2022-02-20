import numpy as np
from sampling import sample_directions
from scipy.constants import physical_constants

amu_to_mev = physical_constants['atomic mass constant energy equivalent in MeV'][0]


def react_particles(reaction_masses, v_a, v_b, product_direction=None):

    (m_a, m_b, m_c, m_d) = np.array(reaction_masses) * amu_to_mev           # Masses in MeV/c**2

    mr = m_a*m_b / (m_a + m_b)                                              # Reduced mass
    u = v_a - v_b                                         # Relative velocity (fraction of c)
    k = 0.5 * mr * np.sum(u ** 2, axis=1)               # Kinetic Energy
    q = m_a + m_b - m_c - m_d                               # Reaction Q value

    v_cm = (m_a * v_a + m_b * v_b) / (m_a + m_b)              # Center of mass velocity

    e_c_cm = (m_d / (m_c + m_d)) * (q + k)                 # Product center of mass energy
    speed_c_cm = np.sqrt(2 * e_c_cm / m_c)               # Product center of mass speed

    if product_direction is None:
        dir_vectors = sample_directions(speed_c_cm.size)
    else:
        dir_vectors = sample_directions(speed_c_cm.size, fixed_theta=product_direction[0],
                                        fixed_phi=product_direction[1])

    v_c_cm = speed_c_cm * dir_vectors
    v_c = v_c_cm - v_cm

    return v_c





