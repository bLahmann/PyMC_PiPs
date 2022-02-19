

def react_particles(reaction_masses, vA, vB):

    (mA, mB, mC, mD) = reaction_masses

    mr = mA*mB / (mA + mB)              # Reduced mass
    u = vA - vB                         # Relative velocity
    K = 0.5


