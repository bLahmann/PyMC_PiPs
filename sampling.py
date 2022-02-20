import numpy as np
from scipy.interpolate import interp1d


_DEFAULT_NUM_INTERPOLATION_POINTS = 1000


def make_sampler(pdf, min_point, max_point, num_interpolation_points=_DEFAULT_NUM_INTERPOLATION_POINTS):
    x_s = np.linspace(min_point, max_point, num_interpolation_points)
    y_s = pdf(x_s)

    cum_sum = np.cumsum(y_s)
    cum_sum /= cum_sum[-1]
    ppf = interp1d(cum_sum, x_s, fill_value="extrapolate")
    return lambda n: ppf(np.random.uniform(size=n))


def normal_distribution_sampler(mu, sigma, num_interpolation_points=_DEFAULT_NUM_INTERPOLATION_POINTS):
    f = lambda x: np.exp(-0.5 * (x - mu)**2 / sigma)
    return make_sampler(f, mu - 6.0 * sigma, mu + 6.0 * sigma, num_interpolation_points)


def maxwell_energy_sampler(temperature, num_interpolation_points=_DEFAULT_NUM_INTERPOLATION_POINTS):
    f = lambda x: np.sqrt(x / temperature**3) * np.exp(-x/temperature)
    return make_sampler(f, 0, 10.0*temperature, num_interpolation_points)


def sample_directions(num_samples, fixed_theta=None, fixed_phi=None):

    if fixed_theta is None:
        theta_dir_dist = lambda theta: np.sin(theta)
        theta_dir = make_sampler(theta_dir_dist, 0.0, np.pi)(num_samples)
    else:
        theta_dir = fixed_theta * np.ones(num_samples)

    if fixed_phi is None:
        phi_dir_dist = lambda phi: np.ones(phi.shape)
        phi_dir = make_sampler(phi_dir_dist, 0.0, 2.*np.pi)(num_samples)
    else:
        phi_dir = fixed_phi * np.ones(num_samples)

    x_dir = np.sin(theta_dir) * np.cos(phi_dir)
    y_dir = np.sin(theta_dir) * np.sin(phi_dir)
    z_dir = np.cos(theta_dir)
    dir_vectors = np.array([x_dir, y_dir, z_dir]).T

    return dir_vectors

if __name__ == "__main__":

    import matplotlib.pyplot as plotter

    sampler = maxwell_energy_sampler(1.0)
    x_s = sampler(1000000)

    fig = plotter.figure()
    ax = fig.add_subplot()
    plotter.hist(x_s, bins=np.linspace(-5, 15, 1000))
    plotter.show()
