import numpy as np


def hist1d(x, bins=100, range=None, weights=None, density=None, normalized=False):
    hist, dx0 = np.histogram(x, bins=bins, range=range, weights=weights, density=density)
    xran = np.linspace(dx0[0], dx0[-1], hist.shape[0], endpoint=False)
    dx = xran[1] - xran[0]
    xran += dx / 2
    hist = np.transpose(hist)
    if normalized == True and density == False:
        hist /= dx
    return hist, xran


def hist2d(x, y, bins=[100, 100], range=None, weights=None, density=None, normalized=False):
    hist, dx0, dy0 = np.histogram2d(x, y, bins=bins, range=range, weights=weights, normed=density)
    xran = np.linspace(dx0[0], dx0[-1], hist.shape[1], endpoint=False)
    yran = np.linspace(dy0[0], dy0[-1], hist.shape[0], endpoint=False)
    dx = xran[1] - xran[0]
    dy = yran[1] - yran[0]
    xran += dx / 2
    yran += dy / 2
    hist = np.transpose(hist)
    if normalized == True and density == False:
        hist /= dx * dy
    return hist, xran, yran


def hist3d(x, y, z, bins=[100, 100, 100], range=None, weights=None, density=None, normalized=False):
    vals = np.zeros((x.shape[0], 3))
    vals[:, 0] = x[:]
    vals[:, 1] = y[:]
    vals[:, 2] = z[:]
    hist, edges = np.histogramdd(vals, bins=bins, range=range, weights=weights, normed=density)
    dx0 = edges[0]
    dy0 = edges[1]
    dz0 = edges[2]
    xran = np.linspace(dx0[0], dx0[-1], hist.shape[2], endpoint=False)
    yran = np.linspace(dy0[0], dy0[-1], hist.shape[1], endpoint=False)
    zran = np.linspace(dz0[0], dz0[-1], hist.shape[0], endpoint=False)
    dx = xran[1] - xran[0]
    dy = yran[1] - yran[0]
    dz = zran[1] - zran[0]
    xran += dx / 2
    yran += dy / 2
    zran += dz / 2
    hist = np.transpose(hist)
    if normalized == True and density == False:
        hist /= dx * dy
    return hist, xran, yran, zran


# %%


# %%
