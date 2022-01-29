import matplotlib.pyplot as plotter
import numpy
from endf_data import DDn, DDp, DTn, D3Hep
from bosch_hale import DDn_cross_section as DDn_bh, DDp_cross_section as DDp_bh, DTn_cross_section as DTn_bh, D3Hep_cross_section as D3Hep_bh

fig = plotter.figure()
ax = fig.add_subplot()

for data in [DDn, DDp, DTn, D3Hep]:
    x = numpy.linspace(data.min_cross_section_energy, data.max_cross_section_energy, 100000)
    y = data.total_cross_section(x)
    # x = numpy.logspace(-10, -1, 10000)
    # y = data.value(x)
    plotter.plot(x, y)

ax.set_xscale("log")
ax.set_yscale("log")
plotter.show()
