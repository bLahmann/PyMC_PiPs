import numpy as np
from sampling import sample_directions
from scipy.constants import physical_constants

amu_to_mev = physical_constants['atomic mass constant energy equivalent in MeV'][0]


def react_particles(reaction_masses, v_a, v_b, product_direction_lab=None):

    (m_a, m_b, m_c, m_d) = np.array(reaction_masses) * amu_to_mev           # Masses in MeV/c**2

    mr = m_a*m_b / (m_a + m_b)                              # Reduced mass (MeV/c**2)
    u = v_a - v_b                                           # Relative velocity (fraction of c)
    k = 0.5 * mr * np.linalg.norm(u, axis=1)**2             # Kinetic Energy (MeV)
    q = m_a + m_b - m_c - m_d                               # Reaction Q value (MeV)

    v_cm = (m_a * v_a + m_b * v_b) / (m_a + m_b)            # Center of mass velocity (fraction of c)

    energy_c_cm = (m_d / (m_c + m_d)) * (q + k)             # Product center of mass energy (MeV)
    speed_c_cm = np.sqrt(2 * energy_c_cm / m_c)             # Product center of mass speed (fraction of c)

    if product_direction_lab is None:
        dir_vectors_cm = sample_directions(speed_c_cm.size)         # Direction vectors in CM Frame
        v_c_cm = dir_vectors_cm * speed_c_cm[:, np.newaxis]         # Product center of mass velocity (fraction of c)
        v_c_lab = v_c_cm - v_cm                                     # Product lab velocity (fraction of c)
        dir_vectors_lab = v_c_lab / np.linalg.norm(v_c_lab, axis=1)[:, np.newaxis]  # Direction vectors in Lab Frame
        energy_c_lab = 0.5 * m_c * np.linalg.norm(v_c_lab, axis=1)**2               # Product lab energy (MeV)
    else:
        dir_vectors_lab = sample_directions(speed_c_cm.size,
                                            fixed_theta=product_direction_lab[0],
                                            fixed_phi=product_direction_lab[1])     # Direction vectors in Lab Frame
        v_cm_norm = v_cm / np.linalg.norm(v_cm, axis=1)[:, np.newaxis]              # Center of mass direction
        angle = np.arccos(dir_vectors_lab[:, 0] * v_cm_norm[:, 0] +
                          dir_vectors_lab[:, 1] * v_cm_norm[:, 1] +
                          dir_vectors_lab[:, 2] * v_cm_norm[:, 2]
                          )                                                         # Angle between the two

        # Law of cosines
        speed_c_lab = np.linalg.norm(v_cm, axis=1) * np.cos(angle) + \
                      np.sqrt(speed_c_cm**2 + np.linalg.norm(v_cm, axis=1)**2 * (np.cos(angle)**2 - 1))
        energy_c_lab = 0.5 * m_c * speed_c_lab**2

    return dir_vectors_lab, energy_c_lab





