import numpy
import functools


def cross_section(bg, a, b, energy_mev):
    energy = 1.0e3 * energy_mev
    num = a[0] + energy * (a[1] + energy * (a[2] + energy * (a[3] + energy * a[4])))
    den = 1.0 + energy * (b[0] + energy * (b[1] + energy * (b[2] + energy * b[3])))
    s = num / den
    return 1e-3 * s / (energy * numpy.exp(bg / numpy.sqrt(energy)))


def reactivity(bg, mr, c, temperature):
    num = temperature * (c[1] + temperature * (c[3] + temperature * c[5]))
    den = 1 + temperature * (c[2] + temperature * (c[4] + temperature * c[6]))
    theta = temperature / (1 - num / den)
    xi = (bg ** 2.0 / (4.0 * theta)) ** (1.0 / 3.0)
    return c[0] * theta * numpy.sqrt(xi / (mr * temperature ** 3.0)) * numpy.exp(-3.0 * xi)


_DTn_bg = 34.3827
_D3Hep_bg = 68.7508
_DDp_bg = 31.397
_DDn_bg = 31.3970

_DTn_mr = 1124656
_D3Hep_mr = 1124572
_DDp_mr = 937814
_DDn_mr = 937814

_DTn_a = [6.927e4, 7.454e8, 2.050e6, 5.2002e4, 0.0]
_D3Hep_a = [5.7501e6, 2.5226e3, 4.5566e1, 0.0, 0.0]
_DDp_a = [5.5576e4, 2.1054e2, -3.2638e-2, 1.4987e-6, 1.8181e-10]
_DDn_a = [5.3701e4, 3.3027e2, -1.2706e-1, 2.9327e-5, -2.5151e-9]

_DTn_b = [6.38e1, -9.95e-1, 6.981e-5, 1.728e-4]
_D3Hep_b = [-3.1995e-3, -8.5530e-6, 5.9014e-8, 0.0]
_DDp_b = [0.0, 0.0, 0.0, 0.0]
_DDn_b = [0.0, 0.0, 0.0, 0.0]

_DTn_c = [1.17302e-9, 1.51361e-2, 7.51886e-2, 4.60643e-3, 1.35e-2, -1.06750e-4, 1.366e-5]
_D3Hep_c = [5.51036e-10, 6.41918e-3, -2.02896e-3, -1.91080e-5, 1.35776e-4, 0.0, 0.0]
_DDp_c = [5.65718e-12, 3.41267e-3, 1.99167e-3, 0.0, 1.05060e-5, 0.0, 0.0]
_DDn_c = [5.43360e-12, 5.85779e-3, 7.68222e-3, 0.0, -2.964e-6, 0.0, 0.0]


DTn_cross_section = functools.partial(cross_section, _DTn_bg, _DTn_a, _DTn_b)
D3Hep_cross_section = functools.partial(cross_section, _D3Hep_bg, _D3Hep_a, _D3Hep_b)
DDp_cross_section = functools.partial(cross_section, _DDp_bg, _DDp_a, _DDp_b)
DDn_cross_section = functools.partial(cross_section, _DDn_bg, _DDn_a, _DDn_b)

DTn_reactivity = functools.partial(reactivity, _DTn_bg, _DTn_mr, _DTn_c)
D3Hep_reactivity = functools.partial(reactivity, _D3Hep_bg, _D3Hep_mr, _D3Hep_c)
DDp_reactivity = functools.partial(reactivity, _DDp_bg, _DDp_mr, _DDp_c)
DDn_reactivity = functools.partial(reactivity, _DDn_bg, _DDn_mr, _DDn_c)


# Sanity Check
if __name__ == "__main__":
    print("Cross-Section Check (Table VIII)")
    for sigma in [DTn_cross_section, D3Hep_cross_section, DDp_cross_section, DDn_cross_section]:
        x = 1e-3 * numpy.array([3, 4, 5, 6, 7, 8, 9, 10, 12, 15,
                         20, 30, 40, 50, 60, 70, 80, 100,
                         120, 140, 150, 200, 250, 300, 400])
        print(1e3 * sigma(x))

    print("Reactivity Check (Table VIII)")
    for sigma_v in [DTn_reactivity, D3Hep_reactivity, DDp_reactivity, DDn_reactivity]:
        x = numpy.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
             1.0, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0,
             4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0,
             20.0, 30.0, 40.0, 50.0])
        print(sigma_v(x))
