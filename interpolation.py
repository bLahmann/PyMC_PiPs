import functools

import numpy as np
import scipy.interpolate as si
from scipy.interpolate import interp1d


def logx_interp1d(x, y, kind='linear', fill_value='extrapolate'):
    log_x = np.log10(x)
    f = interp1d(log_x, y, kind=kind, fill_value=fill_value)
    return lambda z: f(np.log10(z))


def logy_interp1d(x, y, kind='linear', fill_value='extrapolate'):
    log_y = np.log10(y)
    f = interp1d(x, log_y, kind=kind, fill_value=fill_value)
    return lambda z: np.power(10.0, f(z))


def loglog_interp1d(x, y, kind='linear', fill_value='extrapolate'):
    log_x = np.log10(x)
    log_y = np.log10(y)
    f = interp1d(log_x, log_y, kind=kind, fill_value=fill_value)
    return lambda z: np.power(10.0, f(np.log10(z)))


def gamow_interp1d(x, y):
    return functools.partial(gamow_interp1d_eval, x, y)


def gamow_interp1d_eval(x, y, x_q):

    x_array = np.array(x)
    y_array = np.array(y)

    result = np.empty(x_q.shape)
    result[:] = np.NaN

    f = interp1d(x_array, y_array, kind="previous")
    y1 = f(x_q)
    x1 = x_array[np.searchsorted(y_array, y1)]

    f = interp1d(x_array, y_array, kind="next")
    y2 = f(x_q)
    x2 = x_array[np.searchsorted(y_array, y2)]

    b = np.sqrt(x1*x2) * (np.log(y2) - np.log(y1)) / (np.sqrt(x2) - np.sqrt(x1))
    a = y1 * np.exp(b / np.sqrt(x1))

    mask = np.isin(x_q, x)
    result[mask] = y1[mask]

    mask = np.invert(mask)
    result[mask] = a[mask] * np.exp(-b[mask]/np.sqrt(x_q[mask]))

    return result


def interp2d_pairs(*args,**kwargs):
    """ Same interface as interp2d but the returned interpolant will evaluate its inputs as pairs of values.
    """
    # Internal function, that evaluates pairs of values, output has the same shape as input
    def interpolant(x,y,f):
        x,y = np.asarray(x), np.asarray(y)
        return (si.dfitpack.bispeu(f.tck[0], f.tck[1], f.tck[2], f.tck[3], f.tck[4], x.ravel(), y.ravel())[0]).reshape(x.shape)
    # Wrapping the scipy interp2 function to call out interpolant instead
    return lambda x,y: interpolant(x,y,si.interp2d(*args,**kwargs))


if __name__ == "__main__":
    y = gamow_interp1d([1., 2., 3.], [3., 4., 5.], np.linspace(1, 2, 20))
    print(y)
