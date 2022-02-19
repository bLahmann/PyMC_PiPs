import numpy as np
from fdint import fdk
from scipy.integrate import quad
from scipy.special import gamma
import numpy
import matplotlib.pyplot as plotter


def fermi_integral(j, x):
    f = lambda t, j, x: t ** j / (numpy.exp(t - x) + 1)
    return quad(f, 0.0, numpy.infty, (x, j)) / gamma(j+1)


print(fermi_integral(5.0, 0)[0])
print(fdk(0, 5.0))


fig = plotter.figure()
x = np.linspace(-100, 100, 1000)
plotter.plot(x, fdk(3/2, x) * gamma(3/2) / (fdk(1/2, x) * gamma(5/2)))
plotter.show()

