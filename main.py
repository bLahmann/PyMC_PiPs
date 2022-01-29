import matplotlib.pyplot as plotter
import numpy
from endf_data import DDn, DDp, DTn, D3Hep


fig = plotter.figure()
ax = fig.add_subplot()

for data in [DDn, DDp, DTn, D3Hep]:
    x = numpy.linspace(data.min_cross_section_energy, data.max_cross_section_energy, 100000)
    y = data.total_cross_section(x)
    plotter.plot(x, y)

ax.set_xscale("log")
ax.set_yscale("log")
plotter.show()
