import numpy as np
from scipy.interpolate import interp1d
import numpy


def log_interp1d(x, y, kind='linear', fill_value='extrapolate'):
    log_x = numpy.log10(x)
    log_y = numpy.log10(y)
    f = interp1d(log_x, log_y, kind=kind, fill_value=fill_value)
    return lambda z: np.power(10.0, f(np.log10(z)))
