from fdint import fdk
from scipy.integrate import quad
import numpy

def fermi_integral(x, j):
    f = lambda t, x, j: t ** j / (numpy.exp(t - x) + 1)
    return quad(f, 0.0, numpy.infty, (x, j))


print(fermi_integral(5.0, 0))
print(fdk(0, 5.0))

