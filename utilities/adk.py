import numpy as np
from scipy.special import gamma


def n_s(Z, ionenergy):
    return 3.69 * Z / np.sqrt(ionenergy)


def ionization_rate(Z, ionenergy, field):
    ns = n_s(Z, ionenergy)
    ie = ionenergy ** (3 / 2) / field
    w1 = 1.52e15 * 4 ** ns * ionenergy / (ns * gamma(2 * ns))
    w2 = 20.5 * ie
    w3 = -6.83 * ie
    return w1 * w2 ** (2 * ns - 1) * np.exp(w3)  # Units: per second
