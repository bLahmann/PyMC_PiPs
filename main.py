import matplotlib.pyplot as plotter
from scipy.integrate import trapezoid as integrate
import numpy
from endf_data import nDn, nTn, n3Hen

fig = plotter.figure()
ax = fig.add_subplot()

counter = 1
for sig in [nDn, nTn, n3Hen]:

    """
    energies = numpy.logspace(-11, 2, 100)
    total = sig.total_cross_section(energies)
    for i in range(len(total)):
        print("%.4e %.4e" % (energies[i], total[i]))
    """

    mu = numpy.linspace(-1, 1, 1000)
    E = 1000 * numpy.ones(mu.shape)
    y = sig.diff_cross_section(E, mu)
    plotter.plot(mu, y, label=counter)
    counter += 1

# ax.set_xscale("log")
ax.set_yscale("log")
plotter.legend(loc="upper right")
plotter.show()

