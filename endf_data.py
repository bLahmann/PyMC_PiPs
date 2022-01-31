from scipy.interpolate import interp1d
from scipy.special import eval_legendre
from utils import log_interp1d
import numpy
import functools
import sys


def fetch_endf_data(endf_file, mt):
    f = open(endf_file)
    lines = f.readlines()
    f.close()

    cross_section_data = []
    angular_dist_data = []
    for line in lines:
        entries = get_endf_line_entries(line)
        if entries[7] == 3 and entries[8] == mt:
            cross_section_data.append(entries)
        if entries[7] == 4 and entries[8] == mt:
            angular_dist_data.append(entries)

    return cross_section_data, angular_dist_data


def get_endf_line_entries(endf_line):
    return [
        convert_endf_number(endf_line[0:11]),
        convert_endf_number(endf_line[11:22]),
        convert_endf_number(endf_line[22:33]),
        convert_endf_number(endf_line[33:44]),
        convert_endf_number(endf_line[44:55]),
        convert_endf_number(endf_line[55:66]),
        int(endf_line[66:70]),                  # Target ID
        int(endf_line[70:72]),                  # MF Number
        int(endf_line[72:75]),                  # MT Number
        int(endf_line[75:80])                   # Line Number
    ]


def convert_endf_number(endf_number):

    # Handle the sign
    sign = +1
    if endf_number[0] == "-":
        sign = -1
    endf_number = endf_number[1:]

    # If it's a positive exponent
    if "+" in endf_number:
        temp = endf_number.split("+")
        try:
            base = float(temp[0])
            power = +int(temp[1])
            return sign * base * 10 ** power
        except ValueError:
            return endf_number

    # If it's a negative exponent
    if "-" in endf_number:
        temp = endf_number.split("-")
        try:
            base = float(temp[0])
            power = -int(temp[1])
            return sign * base * 10 ** power
        except ValueError:
            return endf_number

    # Check if it's an int before giving up
    try:
        return int(endf_number)
    except ValueError:
        return endf_number


def parse_cross_sections(cross_section_data):

    # Parse the header data
    target_zaid = round(cross_section_data[0][0])
    target_mass = cross_section_data[0][1]
    q_value = cross_section_data[1][0]
    nr = cross_section_data[1][5]
    np = cross_section_data[1][6]

    # Parse the cross-section data
    energies = []
    cross_sections = []
    for i in range(3, len(cross_section_data)):
        energies.append(cross_section_data[i][0])
        energies.append(cross_section_data[i][2])
        energies.append(cross_section_data[i][4])
        cross_sections.append(cross_section_data[i][1])
        cross_sections.append(cross_section_data[i][3])
        cross_sections.append(cross_section_data[i][5])

    # Remove any trailing strings
    while isinstance(energies[-1], str):
        energies.pop(-1)
        cross_sections.pop(-1)

    return 1.0e-6 * numpy.array(energies), numpy.array(cross_sections)


def parse_angular_dist_data(angular_dist_data):

    ltt = angular_dist_data[0][3]
    li = angular_dist_data[1][2]
    lct = angular_dist_data[1][3]

    # Parse coefficient data (if applicable)
    coefficient_energies = []
    coefficients = []
    coefficient_arrays = []
    if ltt == 1 or ltt == 3:

        num_energies = angular_dist_data[3][0]
        angular_dist_data = angular_dist_data[4:]                   # Remove header

        i = 0
        while len(coefficient_energies) < num_energies:
            coefficient_energies.append(angular_dist_data[i][1])
            num_coefficients = angular_dist_data[i][4]

            coefficients.append([])
            for j in range(0, num_coefficients):
                if j % 6 == 0:
                    i += 1
                coefficients[-1].append(angular_dist_data[i][j % 6])

            i += 1

        angular_dist_data = angular_dist_data[i:]                   # Remove unneeded data

        # Make arrays for easy interpolation
        coefficient_arrays = []
        for i in range(len(coefficients[-1])):
            coefficient_arrays.append(numpy.zeros(len(coefficients)))
            for j in range(len(coefficients)):
                try:
                    coefficient_arrays[i][j] = coefficients[j][i]
                except IndexError:
                    pass

    # Parse interpolation data (if applicable)
    interpolation_energies = []
    interpolation_mus = []
    interpolation_values = []
    interpolation_value_array = None
    if ltt == 2 or ltt == 3:

        if ltt == 2:
            angular_dist_data = angular_dist_data[2:]

        num_sections = angular_dist_data[0][5]
        angular_dist_data = angular_dist_data[2:]
        endf_interpolation_mus = []

        for _ in range(num_sections):

            num_points = angular_dist_data[0][5]
            interpolation_energies.append(angular_dist_data[0][1])
            endf_interpolation_mus.append([])
            interpolation_values.append([])
            angular_dist_data = angular_dist_data[2:]
            while len(interpolation_values[-1]) < num_points:
                endf_interpolation_mus[-1].append(angular_dist_data[0][0])
                endf_interpolation_mus[-1].append(angular_dist_data[0][2])
                endf_interpolation_mus[-1].append(angular_dist_data[0][4])
                interpolation_values[-1].append(angular_dist_data[0][1])
                interpolation_values[-1].append(angular_dist_data[0][3])
                interpolation_values[-1].append(angular_dist_data[0][5])
                angular_dist_data = angular_dist_data[1:]

            while isinstance(endf_interpolation_mus[-1][-1], str):
                endf_interpolation_mus[-1].pop(-1)
                interpolation_values[-1].pop(-1)

        # The final mu vector should be the most finely divided
        interpolation_mus = endf_interpolation_mus[-1]

        # Create a 2D array for easier interpolation
        interpolation_value_array = numpy.zeros((len(interpolation_energies), len(interpolation_mus)))
        for i in range(len(interpolation_values)):
            f = interp1d(endf_interpolation_mus[i], interpolation_values[i])
            interpolation_value_array[i] = f(interpolation_mus)

    return coefficient_energies, coefficient_arrays, \
        interpolation_energies, interpolation_mus, interpolation_value_array


def evaluate_angular_dist(coefficient_energies, coefficient_arrays,
                          interpolation_energies, interpolation_mus, interpolation_values,
                          energies, mus):

    result = numpy.zeros(numpy.array(mus).shape)

    min_interpolation_energy = sys.float_info.max
    if interpolation_energies:
        min_interpolation_energy = interpolation_energies[0]

    if coefficient_energies:
        a = []
        for i in range(len(coefficient_arrays)):
            f = interp1d(coefficient_energies, coefficient_arrays[i])
            a.append(f(energies))

        mask = energies < min_interpolation_energy
        result[mask] = 0.5
        for l in range(1, len(coefficient_arrays)+1):
            norm = (2*l + 1) / 2.0
            result[mask] += norm * a[l-1] * eval_legendre(l, mus)

    return result


class ENDFData:

    def __init__(self, file, mt):
        cross_section_data, angular_dist_data = fetch_endf_data(file, mt)

        if cross_section_data:
            # TODO: the interpolation scheme is defined in the ENDF file
            x, y = parse_cross_sections(cross_section_data)
            self.total_cross_section = log_interp1d(x, y, fill_value="extrapolate")
            # self.total_cross_section = interp1d(x, y, fill_value="extrapolate")
            self.min_cross_section_energy = x[0]
            self.max_cross_section_energy = x[-1]

        if angular_dist_data:
            coefficient_energies, coefficient_arrays, interpolation_energies, interpolation_mus, interpolation_values \
                = parse_angular_dist_data(angular_dist_data)
            self.angular_dist = functools.partial(evaluate_angular_dist, coefficient_energies, coefficient_arrays,
                                                  interpolation_energies, interpolation_mus, interpolation_values)

    def diff_cross_section(self, energy, mu):
        return self.total_cross_section(energy) * \
            self.angular_dist(energy, mu) / \
            (2 * numpy.pi)


# Fusion cross-sections
DDn = ENDFData("data/endf/D/D.txt", 50)
DDp = ENDFData("data/endf/D/D.txt", 600)
DTn = ENDFData("data/endf/T/D.txt", 50)
D3Hep = ENDFData("data/endf/3He/D.txt", 600)

# Neutron scatter cross-sections
nDn = ENDFData("data/endf/D/n.txt", 2)
nTn = ENDFData("data/endf/T/n.txt", 2)
n3Hen = ENDFData("data/endf/3He/n.txt", 2)



