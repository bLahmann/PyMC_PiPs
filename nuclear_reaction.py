import numpy as np
from sampling import sample_directions
from scipy.constants import physical_constants

amu_to_mev = physical_constants['atomic mass constant energy equivalent in MeV'][0]


def react_particles(reaction_masses, v_a, v_b, product_direction_lab=None):

    (m_a, m_b, m_c, m_d) = np.array(reaction_masses) * amu_to_mev           # Masses in MeV/c**2

    mr = m_a*m_b / (m_a + m_b)                              # Reduced mass (MeV/c**2)
    u = v_a - v_b                                           # Relative velocity (fraction of c)
    k = 0.5 * mr * np.sum(u ** 2, axis=1)                   # Kinetic Energy (MeV)
    q = m_a + m_b - m_c - m_d                               # Reaction Q value (MeV)

    v_cm = (m_a * v_a + m_b * v_b) / (m_a + m_b)            # Center of mass velocity (fraction of c)

    e_c_cm = (m_d / (m_c + m_d)) * (q + k)                  # Product center of mass energy (MeV)
    speed_c_cm = np.sqrt(2 * e_c_cm / m_c)                  # Product center of mass speed (fraction of c)

    if product_direction_lab is None:
        dir_vectors_cm = sample_directions(speed_c_cm.size)
        v_c_cm = dir_vectors_cm * speed_c_cm[:, np.newaxis]     # Product center of mass velocity (fraction of c)
        v_c = v_c_cm - v_cm                                     # Product lab velocity (fraction of c)
        e_c = 0.5 * m_c * np.sum(v_c ** 2, axis=1)              # Product lab energy (MeV)
    else:
        dir_vectors_lab = sample_directions(speed_c_cm.size, fixed_theta=product_direction_lab[0],
                                            fixed_phi=product_direction_lab[1])
        v_cm_norm = v_cm / np.linalg.norm(v_cm)
        angle = np.arccos(np.dot(dir_vectors_lab, v_cm_norm))



    return e_c





