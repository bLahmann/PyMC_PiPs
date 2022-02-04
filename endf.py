import numpy as np
import functools
from scipy.interpolate import interp1d
from interpolation import logx_interp1d, logy_interp1d, loglog_interp1d


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

    # Get the cross section table
    table = []
    for i in range(3, len(cross_section_data)):
        table.append((cross_section_data[i][0], cross_section_data[i][1]))
        table.append((cross_section_data[i][2], cross_section_data[i][3]))
        table.append((cross_section_data[i][4], cross_section_data[i][5]))

    # Remove any string entries
    while isinstance(table[-1][0], str):
        table.pop(-1)

    # Divide the table into interpolation sections
    cross_section_tables = []
    min_energies = []
    max_energies = []
    for i in range(len(interp_method_ids)):
        cross_section_tables.append(table[(interp_indexes[i]-1):interp_indexes[i+1]])
        min_energies.append(cross_section_tables[-1][0][0])
        max_energies.append(cross_section_tables[-1][-1][0])

    return cross_section_tables, interp_method_ids, min_energies, max_energies


def _interpolate_cross_section_tables(cross_section_tables, interp_method_ids, min_energies, max_energies, query_energies_mev):

    interp_methods = {
        1: functools.partial(interp1d, kind="nearest"),
        2: functools.partial(interp1d, kind="linear"),
        3: functools.partial(logx_interp1d, kind="linear"),
        4: functools.partial(logy_interp1d, kind="linear"),
        5: functools.partial(loglog_interp1d, kind="linear"),
    }

    # Init
    E = 1.0e6 * np.array(query_energies_mev)
    cross_sections = np.empty(E.shape)
    cross_sections[:] = np.NaN

    for i in range(len(cross_section_tables)):

        mask = min_energies[i] <= E <= max_energies[i]


    pass



tables, method_ids, min_energies, max_energies = _get_cross_section_tables("./data/endf/D/D.txt", 50)

interp_nearest = functools.partial(interp1d, kind="nearest")
f = interp_nearest([0, 1, 2, 3], [0, 1, 2, 3])
print(f([0.1, 0.2, 0.8, 0.9]))
# _interpolate_cross_section_tables(tables, method_ids, min_energies, max_energies, [0.1, 1, 10])
print("Hello")
