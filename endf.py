import sys
import numpy
import functools
from scipy.interpolate import interp1d
from scipy.special import eval_legendre
from interpolation import logx_interp1d, logy_interp1d, loglog_interp1d, gamow_interp1d, interp2d_pairs


def _convert_string_to_number(string):
    """Converts a string entry from the ENDF file into either a float or int if possible"""

    # Handle the sign
    sign = +1
    if string[0] == "-":
        sign = -1
    string = string[1:]

    # If it's a positive exponent
    if "+" in string:
        temp = string.split("+")
        try:
            base = float(temp[0])
            power = +int(temp[1])
            return sign * base * 10 ** power
        except ValueError:
            return string

    # If it's a negative exponent
    if "-" in string:
        temp = string.split("-")
        try:
            base = float(temp[0])
            power = -int(temp[1])
            return sign * base * 10 ** power
        except ValueError:
            return string

    # Check if it's an int before giving up
    try:
        return int(string)
    except ValueError:
        return string


def _convert_line_entries(line):
    """Segments an ENDF line into it's 10 entries"""

    return (
        _convert_string_to_number(line[0:11]),
        _convert_string_to_number(line[11:22]),
        _convert_string_to_number(line[22:33]),
        _convert_string_to_number(line[33:44]),
        _convert_string_to_number(line[44:55]),
        _convert_string_to_number(line[55:66]),
        int(line[66:70]),                           # MAT Number
        int(line[70:72]),                           # MF Number
        int(line[72:75]),                           # MT Number
        int(line[75:80])                            # Line Number
    )


def _convert_file(file):
    """Converts an ENDF file into a dict """

    f = open(file)
    lines = f.readlines()
    f.close()

    data = []
    for line in lines:
        data.append(_convert_line_entries(line))

    data_dictionary = {}
    for entry in data:
        mf = entry[7]
        mt = entry[8]

        if mf not in data_dictionary:
            data_dictionary[mf] = {}
        if mt not in data_dictionary[mf]:
            data_dictionary[mf][mt] = []

        data_dictionary[mf][mt].append(entry)

    return data_dictionary


def _get_cross_section_tables(file, mt):
    """Organizes cross-sections into interpolation tables"""

    data = _convert_file(file)
    header_data = data[1][451]
    cross_section_data = data[3][mt]

    # Headers
    zaid = cross_section_data[0][0]
    awr = cross_section_data[0][1]      # Target mass / neutron mass
    awi = header_data[2][0]             # Projectile mass / neutron mass
    cm_factor = awr / (awr + awi)       # Convert from Lab -> CM Frame

    qm = cross_section_data[1][0]
    qi = cross_section_data[1][1]
    nr = cross_section_data[1][4]
    np = cross_section_data[1][5]

    # Get the number of points for each interpolation region
    interp_indexes = [1]
    interp_method_ids = []
    for i in range(nr):
        interp_indexes.append(cross_section_data[2][2*i])
        interp_method_ids.append(cross_section_data[2][2*i+1])

    # Get the cross-section table
    energies = []
    cross_sections = []
    for i in range(3, len(cross_section_data)):
        energies.append(cross_section_data[i][0])
        energies.append(cross_section_data[i][2])
        energies.append(cross_section_data[i][4])
        cross_sections.append(cross_section_data[i][1])
        cross_sections.append(cross_section_data[i][3])
        cross_sections.append(cross_section_data[i][5])

    # Remove any string entries
    while isinstance(energies[-1], str):
        energies.pop(-1)
        cross_sections.pop(-1)

    # Convert to numpy arrays
    energies = numpy.array(energies)
    cross_sections = numpy.array(cross_sections)

    # Convert from the Lab to the CM Frame
    energies *= cm_factor

    # Divide the table into interpolation sections
    interp_energies = []
    interp_cross_sections = []
    for i in range(len(interp_method_ids)):
        interp_energies.append(energies[(interp_indexes[i]-1):interp_indexes[i+1]])
        interp_cross_sections.append(cross_sections[(interp_indexes[i]-1):interp_indexes[i+1]])

    return interp_energies, interp_cross_sections, interp_method_ids


def _interpolate_cross_section_tables(interp_energies, interp_cross_sections, interp_method_ids, query_energies_mev):
    """Function that interpolates the cross-section data based on the specified methods"""

    interp_methods = {
        1: functools.partial(interp1d, kind="nearest"),
        2: functools.partial(interp1d, kind="linear"),
        3: functools.partial(logx_interp1d, kind="linear"),
        4: functools.partial(logy_interp1d, kind="linear"),
        5: functools.partial(loglog_interp1d, kind="linear"),
        6: gamow_interp1d
    }

    # Init
    E = 1.0e6 * numpy.array(query_energies_mev)
    result = numpy.empty(E.shape)
    result[:] = numpy.NaN

    for i in range(len(interp_energies)):

        min_mask = E >= interp_energies[i][0]
        max_mask = E <= interp_energies[i][-1]
        mask = min_mask * max_mask

        f = interp_methods[interp_method_ids[i]](interp_energies[i], interp_cross_sections[i])
        result[mask] = f(E[mask])

    return result


def _get_angular_dist_tables(file, mt):
    """Organizes angular distribution data into interpolation tables"""

    data = _convert_file(file)
    header_data = data[1][451]
    angular_dist_data = data[4][mt]

    # Header
    zaid = angular_dist_data[0][0]
    awr = angular_dist_data[0][1]       # Target mass / neutron mass
    ltt = angular_dist_data[0][3]
    awi = header_data[2][0]             # Projectile mass / neutron mass
    cm_factor = awr / (awr + awi)       # Convert from Lab -> CM Frame

    li = angular_dist_data[1][2]
    lct = angular_dist_data[1][3]

    nr = angular_dist_data[2][4]
    np = angular_dist_data[2][5]

    # If applicable, get the coefficient data
    coefficient_energies = []
    coefficients = []
    coefficient_arrays = []
    if ltt == 1 or ltt == 3:

        line = 4
        while len(coefficient_energies) < np:
            coefficient_energies.append(angular_dist_data[line][1])
            num_coefficients = angular_dist_data[line][4]

            coefficients.append([])
            for j in range(0, num_coefficients):
                if j % 6 == 0:
                    line += 1
                coefficients[-1].append(angular_dist_data[line][j % 6])

            line += 1

        # Transpose the array data for easier interpolation
        coefficient_arrays = []
        for i in range(len(coefficients[-1])):
            coefficient_arrays.append(numpy.zeros(len(coefficients)))
            for j in range(len(coefficients)):
                try:
                    coefficient_arrays[i][j] = coefficients[j][i]
                except IndexError:
                    pass

        # Convert to numpy arrays
        coefficient_energies = cm_factor * numpy.array(coefficient_energies)

    # If applicable, get the interpolation tables
    interp_energies = []
    interp_mus = []
    interp_values = []
    interp_value_array = None
    if ltt == 2 or ltt == 3:

        if ltt == 2:
            line = 2

        num_sections = angular_dist_data[line][5]
        line += 2

        endf_interpolation_mus = []
        for _ in range(num_sections):

            interp_energies.append(angular_dist_data[line][1])
            num_points = angular_dist_data[line][5]
            line += 2

            endf_interpolation_mus.append([])
            interp_values.append([])
            while len(interp_values[-1]) < num_points:
                endf_interpolation_mus[-1].append(angular_dist_data[line][0])
                endf_interpolation_mus[-1].append(angular_dist_data[line][2])
                endf_interpolation_mus[-1].append(angular_dist_data[line][4])
                interp_values[-1].append(angular_dist_data[line][1])
                interp_values[-1].append(angular_dist_data[line][3])
                interp_values[-1].append(angular_dist_data[line][5])
                line += 1

            # Get rid of any extra strings
            while isinstance(endf_interpolation_mus[-1][-1], str):
                endf_interpolation_mus[-1].pop(-1)
                interp_values[-1].pop(-1)

        # We need to make all the sections the same size for interpolation
        # The final mu vector should be the most finely divided
        interp_mus = endf_interpolation_mus[-1]

        # Create a 2D array for interpolation
        interp_value_array = numpy.zeros((len(interp_energies), len(interp_mus)))
        for i in range(len(interp_values)):
            f = interp1d(endf_interpolation_mus[i], interp_values[i])
            interp_value_array[i] = f(interp_mus)

        # Convert to numpy arrays
        interp_energies = cm_factor * numpy.array(interp_energies)
        interp_mus = numpy.array(interp_mus)

    return coefficient_energies, coefficient_arrays, interp_energies, interp_mus, interp_value_array


def _interpolate_angular_dist_tables(coefficient_energies, coefficients,
                                     interp_energies, interp_mus, interp_values,
                                     energy_mev, mus):
    """Function that interpolates the angular distribution tables"""

    energy_array = 1.0e6 * numpy.array(energy_mev)
    mu_array = numpy.array(mus)

    result = numpy.empty(mu_array.shape)
    result[:] = numpy.NaN

    if coefficient_energies.any():
        min_coefficient_energy = coefficient_energies[0]
        max_coefficient_energy = coefficient_energies[-1]
    else:
        min_coefficient_energy = sys.float_info.max
        max_coefficient_energy = sys.float_info.min

    if interp_energies.any():
        min_interpolation_energy = interp_energies[0]
        max_interpolation_energy = interp_energies[-1]
    else:
        min_interpolation_energy = sys.float_info.max
        max_interpolation_energy = sys.float_info.min


    # Do coefficients if required
    min_mask = energy_array >= min_coefficient_energy
    max_mask = energy_array <= max_coefficient_energy
    mask = min_mask * max_mask
    if mask.any():

        a = []
        for i in range(len(coefficients)):
            f = interp1d(coefficient_energies, coefficients[i], fill_value="extrapolate")
            a.append(f(energy_array[mask]))

        result[mask] = 0.5
        for l in range(1, len(coefficients) + 1):
            norm = (2*l + 1) / 2.0
            result[mask] += norm * a[l-1] * eval_legendre(l, mu_array[mask])

    # Do interpolation if required
    min_mask = energy_array >= min_interpolation_energy
    max_mask = energy_array <= max_interpolation_energy
    mask = min_mask * max_mask
    if mask.any():
        f = interp2d_pairs(interp_mus, interp_energies, interp_values)
        result[mask] = f(mus[mask], energy_array[mask])

    return result


def get_cross_section(file, mt):
    """Returns the cross-section as a function of energy in MeV"""
    interp_energies, interp_cross_sections, method_ids = _get_cross_section_tables(file, mt)
    return functools.partial(_interpolate_cross_section_tables, interp_energies, interp_cross_sections, method_ids)


def get_angular_distribution(file, mt):
    coefficient_energies, coefficients, interp_energies, interp_mus, interp_values = \
        _get_angular_dist_tables("data/endf/D/n.txt", 2)
    return functools.partial(_interpolate_angular_dist_tables, coefficient_energies, coefficients,
                             interp_energies, interp_mus, interp_values)


def get_differential_cross_section(file, mt):
    f = get_cross_section(file, mt)
    g = get_angular_distribution(file, mt)
    return lambda x, y: f(x) * g(x, y) / (2*numpy.pi)


# Fusion cross-sections
DDn_cross_section = get_cross_section("data/endf/D/D.txt", 50)
DDp_cross_section = get_cross_section("data/endf/D/D.txt", 600)
DTn_cross_section = get_cross_section("data/endf/T/D.txt", 50)
D3Hep_cross_section = get_cross_section("data/endf/3He/D.txt", 600)

# Neutron scatter cross-sections
nDn_cross_section = get_cross_section("data/endf/D/n.txt", 2)
nTn_cross_section = get_cross_section("data/endf/T/n.txt", 2)
n3Hen_cross_section = get_cross_section("data/endf/3He/n.txt", 2)

# Neutron scatter differential cross-sections
nDn_diff_cross_section = get_differential_cross_section("data/endf/D/n.txt", 2)
nTn_diff_cross_section = get_differential_cross_section("data/endf/T/n.txt", 2)
n3Hen_diff_cross_section = get_differential_cross_section("data/endf/3He/n.txt", 2)


# Testing
if __name__ == "__main__":

    import matplotlib.pyplot as plotter
    from bosch_hale import DDn_cross_section as DDn_bosch_hale
    from bosch_hale import DDp_cross_section as DDp_bosch_hale
    from bosch_hale import DTn_cross_section as DTn_bosch_hale
    from bosch_hale import D3Hep_cross_section as D3Hep_bosch_hale
    from bosch_hale import DTn_high_energy_cross_section as DTn_high_energy_bosch_hale
    from bosch_hale import D3Hep_high_energy_cross_section as D3Hep_high_energy_bosch_hale

    # Compare fusion ENDF cross-sections to Bosch Hale
    data = [
        (DDn_cross_section, DDn_bosch_hale, "DDn Cross Section"),
        (DDp_cross_section, DDp_bosch_hale, "DDp Cross Section"),
        (DTn_cross_section, DTn_bosch_hale, "DTn Cross Section"),
        (D3Hep_cross_section, D3Hep_bosch_hale, "D3Hep Cross Section"),
        (DTn_cross_section, DTn_high_energy_bosch_hale, "DTn High Energy Cross Section"),
        (D3Hep_cross_section, D3Hep_high_energy_bosch_hale, "D3Hep High Energy Cross Section")
    ]

    for endf, bosch_hale, title in data:
        fig = plotter.figure()
        ax = fig.add_subplot()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel("\sigma (b)")
        ax.set_xlabel("CM Energy (MeV)")

        E = numpy.logspace(-3, 1, 1000)
        plotter.plot(E, endf(E), label="ENDF", color="k")
        plotter.plot(E, bosch_hale(E), label="Bosch Hale", color="r", linestyle="--")
        plotter.legend(loc="lower right")
        plotter.ion()
        plotter.show()
        plotter.pause(0.1)

    fig = plotter.figure()
    ax = fig.add_subplot()
    for sig in [nDn_cross_section, nTn_cross_section, n3Hen_cross_section]:

        E = numpy.logspace(-3, 2, 1000)
        plotter.plot(E, sig(E))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Neutron Scatter Cross Sections")
    ax.set_ylabel("\sigma (b)")
    ax.set_xlabel("CM Energy (MeV)")
    plotter.ion()
    plotter.show()
    plotter.pause(0.1)

    data = [
        (nDn_diff_cross_section, "Neutrons on D"),
        (nTn_diff_cross_section, "Neutrons on T"),
        (n3Hen_diff_cross_section, "Neutrons on 3He")
    ]
    for sig, title in data:
        fig = plotter.figure()
        ax = fig.add_subplot()

        mu = numpy.linspace(-1, 1, 1000)

        E = 0.01 * numpy.ones(mu.shape)
        plotter.plot(mu, sig(E, mu), label="10.0 keV")

        E = 0.1 * numpy.ones(mu.shape)
        plotter.plot(mu, sig(E, mu), label="100.0 keV")

        E = 1.0 * numpy.ones(mu.shape)
        plotter.plot(mu, sig(E, mu), label="1.0 MeV")

        E = 10.0 * numpy.ones(mu.shape)
        plotter.plot(mu, sig(E, mu), label="10.0 MeV")

        E = 50.0 * numpy.ones(mu.shape)
        plotter.plot(mu, sig(E, mu), label="50.0 MeV")


        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel("\sigma (b)")
        ax.set_xlabel("\cos (theta)")
        plotter.legend(loc="lower right")
        plotter.ion()
        plotter.show()
        plotter.pause(0.1)

    fig = plotter.figure()
    ax = fig.add_subplot()
    plotter.ioff()
    plotter.show()
