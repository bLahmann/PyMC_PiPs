import numpy as np
import functools
from scipy.interpolate import interp1d
from interpolation import logx_interp1d, logy_interp1d, loglog_interp1d, gamow_interp1d
import matplotlib.pyplot as plotter


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
    cross_section_data = data[3][mt]

    # Header
    zaid = cross_section_data[0][0]
    awr = cross_section_data[0][1]

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
    E = 1.0e6 * np.array(query_energies_mev)
    result = np.empty(E.shape)
    result[:] = np.NaN

    for i in range(len(interp_energies)):

        min_mask = E >= interp_energies[i][0]
        max_mask = E <= interp_energies[i][-1]
        mask = min_mask * max_mask

        f = interp_methods[interp_method_ids[i]](interp_energies[i], interp_cross_sections[i])
        result[mask] = f(E[mask])

    return result


def _get_angular_dist_tables(file, mt):

    data = _convert_file(file)
    angular_dist_data = data[4][mt]

    # Header
    zaid = angular_dist_data[0][0]
    awr = angular_dist_data[0][1]
    ltt = angular_dist_data[0][3]

    li = angular_dist_data[1][2]
    lct = angular_dist_data[1][3]

    nr = angular_dist_data[2][4]
    np = angular_dist_data[2][5]

    # If applicable, get the coefficient data
    pass


def get_cross_section(file, mt):
    """Returns the cross-section as a function of energy in MeV"""
    interp_energies, interp_cross_sections, method_ids = _get_cross_section_tables(file, mt)
    return functools.partial(_interpolate_cross_section_tables, interp_energies, interp_cross_sections, method_ids)


# Fusion cross-sections
DDn_cross_section = get_cross_section("data/endf/D/D.txt", 50)
DDp_cross_section = get_cross_section("data/endf/D/D.txt", 600)
DTn_cross_section = get_cross_section("data/endf/T/D.txt", 50)
D3Hep_cross_section = get_cross_section("data/endf/3He/D.txt", 600)

# Neutron scatter cross-sections
nDn_cross_section = get_cross_section("data/endf/D/n.txt", 2)
nTn_cross_section = get_cross_section("data/endf/T/n.txt", 2)
n3Hen_cross_section = get_cross_section("data/endf/3He/n.txt", 2)


if __name__ == "__main__":

    _get_angular_dist_tables("data/endf/D/n.txt", 2)

    """
    fig = plotter.figure()
    ax = fig.add_subplot()

    for sig in [DDn, DDp, DTn, D3Hep]:

        E = np.logspace(-3, 2, 1000)
        plotter.plot(E, sig(E))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Fusion Cross Sections")
    ax.set_ylabel("\sigma (b)")
    ax.set_xlabel("Energy (MeV)")
    plotter.show()

    fig = plotter.figure()
    ax = fig.add_subplot()

    for sig in [nDn, nTn, n3Hen]:

        E = np.logspace(-3, 2, 1000)
        plotter.plot(E, sig(E))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Neutron Scatter Cross Sections")
    ax.set_ylabel("\sigma (b)")
    ax.set_xlabel("Energy (MeV)")
    plotter.show()
    """

