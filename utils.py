import time
from sampling import make_sampler

def print_loading_message(message, stop):
    while not stop():
        dots = " "
        for _ in range(4):
            if stop():
                break
            print(message + dots, end="")
            time.sleep(0.5)
            print("\r", end="")
            dots += "."
    print(message + " : Done!")


def print_status_message(plasma_indexes, num_escaped, num_died, ):

    pass


def construct_radial_distribution(temperature_profiles, number_density_profiles, reactivity):

    # Rename for clarity
    n1 = number_density_profiles[0]
    n2 = number_density_profiles[1]
    sv = reactivity

    # In reality, if the temperatures are not the same you'd need sigma_v(T_1, T_2)
    # It's possible to construct that (by doing the integrals) but this is a bit much for such a niche capability
    # Despite that, choosing one temperature profile (i.e. assuming they're the same) feels a bit too arbitrary to me
    # so we'll take the average for now
    T = lambda r: 0.5 * (temperature_profiles[0](r) + temperature_profiles[1](r))

    return lambda r: r ** 2 * n1(r) * n2(r) * sv(T(r))


def get_prav_profiles(max_value, min_value, gamma):
    return lambda x: (max_value - min_value) * (1.0 - x ** 2) ** gamma + min_value
