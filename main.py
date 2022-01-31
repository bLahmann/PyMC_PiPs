import matplotlib.pyplot as plotter
from scipy.integrate import trapezoid as integrate
import numpy
from endf_data import nDn, nTn, n3Hen

fig = plotter.figure()
ax = fig.add_subplot()

counter = 1
for sig in [nDn]:

    """
    energies = numpy.logspace(-11, 2, 100)
    total = sig.total_cross_section(energies)
    for i in range(len(total)):
        print("%.4e %.4e" % (energies[i], total[i]))
    """

    E = [1, 10, 100]
    mu = [0, 0, 0]
    y = sig.diff_cross_section(E, mu)
    plotter.plot(mu, y, label=counter)
    counter += 1
    print(2*numpy.pi*integrate(y, mu), sig.total_cross_section(E))

# ax.set_xscale("log")
ax.set_yscale("log")
plotter.legend(loc="upper right")
plotter.show()

