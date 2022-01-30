import matplotlib.pyplot as plotter
import numpy
from endf_data import nDn, nTn, n3Hen

fig = plotter.figure()
ax = fig.add_subplot()

counter = 1
for sig in [nDn, nTn, n3Hen]:

    x = numpy.linspace(sig.min_cross_section_energy, sig.max_cross_section_energy, 100000)
    x = numpy.logspace(-11, 2, 10000)
    y = sig.total_cross_section(x)
    plotter.plot(x, y, label=counter)
    counter += 1

ax.set_xscale("log")
ax.set_yscale("log")
plotter.legend(loc="upper right")
plotter.show()
