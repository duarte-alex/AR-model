import h5py
import numpy as np
from tqdm import tqdm
from utilities.simplify_attrs import *
import sys
import copy as cp

"""
def grid_deposition_2d(file):
    LP = -1
    UP = 1

    def spline_s2(x):
        t0 = 0.5e0 - x
        t1 = 0.5e0 + x
        return np.array([0.5 * t0 ** 2, 0.5 + t0 * t1, 0.5 * t1 ** 2], dtype=np.float16)

    f = h5py.File(file, "r")
    xmin = f.attrs["XMIN"]
    xmax = f.attrs["XMAX"]
    time = f.attrs["TIME"]
    nx = f.attrs["NX"]
    x1range = np.linspace(xmin[0], xmax[0], nx[0], endpoint=False)
    x2range = np.linspace(xmin[1], xmax[1], nx[1], endpoint=False)
    dx1 = x1range[1] - x1range[0]
    dx2 = x2range[1] - x2range[0]
    q = f.get("q")[:]
    x1 = f.get("x1")[:]
    x2 = f.get("x2")[:]
    npar = len(q)
    rho = np.zeros((nx[0] + 2, nx[1] + 2))
    for n in range(npar):
        qq = q[n]
        x_1 = x1[n]
        x_2 = x2[n]
        nx1 = int(np.floor((x_1 - x1range[0]) / dx1))
        nx2 = int(np.floor((x_2 - x2range[0]) / dx2))
        deltax1 = np.float16((x_1 - x1range[nx1]) / dx1 - 0.5)
        deltax2 = np.float16((x_2 - x2range[nx2]) / dx2 - 0.5)
        w1 = spline_s2(deltax1)
        w2 = spline_s2(deltax2)
        for k2 in range(3):
            for k1 in range(3):
                rho[nx1 + k1, nx2 + k2] += qq * w1[k1] * w2[k2]
    return rho[1:-1, 1:-1]
"""

# return rho[1:-1,1:-1]


def grid_deposition_3d_spatial(file):
    LP = -1
    UP = 1

    def spline_s2(x):
        t0 = 0.5e0 - x
        t1 = 0.5e0 + x
        return np.array([0.5 * t0 ** 2, 0.5 + t0 * t1, 0.5 * t1 ** 2], dtype=np.float16)

    f = h5py.File(file, "r")
    xmin = f.attrs["XMIN"]
    xmax = f.attrs["XMAX"]
    time = f.attrs["TIME"]
    nx = f.attrs["NX"]
    x1range = np.linspace(xmin[0], xmax[0], nx[-1], endpoint=False)
    x2range = np.linspace(xmin[1], xmax[1], nx[-2], endpoint=False)
    x3range = np.linspace(xmin[2], xmax[2], nx[-3], endpoint=False)
    dx1 = x1range[1] - x1range[0]
    dx2 = x2range[1] - x2range[0]
    dx3 = x3range[1] - x3range[0]
    q = f.get("q")[:]
    x1 = f.get("x1")[:]
    x2 = f.get("x2")[:]
    x3 = f.get("x3")[:]
    npar = len(q)
    rho = np.zeros((nx[-1] + 2, nx[-2] + 2, nx[-3] + 2))
    for n in range(npar):
        if n % 10000 == 0:
            print(n, " of ", npar)
        qq = q[n]
        x_1 = x1[n]
        x_2 = x2[n]
        x_3 = x3[n]
        nx1 = int(np.floor((x_1 - x1range[0]) / dx1))
        nx2 = int(np.floor((x_2 - x2range[0]) / dx2))
        nx3 = int(np.floor((x_3 - x3range[0]) / dx3))
        deltax1 = np.float16((x_1 - x1range[nx1]) / dx1 - 0.5)
        deltax2 = np.float16((x_2 - x2range[nx2]) / dx2 - 0.5)
        deltax3 = np.float16((x_3 - x3range[nx3]) / dx3 - 0.5)
        w1 = spline_s2(deltax1)
        w2 = spline_s2(deltax2)
        w3 = spline_s2(deltax3)
        for k3 in range(3):
            for k2 in range(3):
                for k1 in range(3):
                    rho[nx1 + k1, nx2 + k2, nx3 + k3] += qq * w1[k1] * w2[k2] * w3[k3]
    return rho[1:-1, 1:-1, 1:-1]


def grid_deposition_cyl(file):
    LP = -1
    UP = 1

    def spline_s2(x):
        t0 = 0.5e0 - x
        t1 = 0.5e0 + x
        return np.array([0.5 * t0 ** 2, 0.5 + t0 * t1, 0.5 * t1 ** 2], dtype=np.float16)

    f = h5py.File(file, "r")
    xmin = f.attrs["XMIN"]
    xmax = f.attrs["XMAX"]
    time = f.attrs["TIME"]
    nx = f.attrs["NX"]
    x1range = np.linspace(xmin[0], xmax[0], nx[0], endpoint=False)
    x2range = np.linspace(xmin[1], xmax[1], nx[1], endpoint=False)
    dx1 = x1range[1] - x1range[0]
    dx2 = x2range[1] - x2range[0]
    x2range = np.linspace(xmin[1] + dx2 / 2, xmax[1] + dx2 / 2, nx[1], endpoint=False)
    x2node = np.linspace(xmin[1], xmax[1] + 2 * dx2, nx[1] + 2, endpoint=False)
    q = f.get("q")[:]
    x1 = f.get("x1")[:]
    x2 = f.get("x2")[:]
    npar = len(q)
    rho = np.zeros((nx[0] + 2, nx[1] + 2))
    for n in range(npar):
        qq = q[n]
        x_1 = x1[n]
        x_2 = x2[n]
        nx1 = int(np.floor((x_1 - x1range[0]) / dx1))
        nx2 = int(np.floor((x_2 - x2range[0]) / dx2))
        deltax1 = np.float16((x_1 - x1range[nx1]) / dx1 - 0.5)
        deltax2 = np.float16((x_2 - x2range[nx2]) / dx2 - 0.5)
        w1 = spline_s2(deltax1)
        w2 = spline_s2(deltax2)
        for k2 in range(3):
            for k1 in range(3):
                rho[nx1 + k1, nx2 + k2] += qq * w1[k1] * w2[k2]
    for n in range(len(x2node)):
        rho[:, n] = np.float16(rho[:, n]) / np.float(x2node[n])
    rho[:, 1] = np.float16(rho[:, 0]) + np.float16(rho[:, 1])
    rho[:, 0] = rho[:, 1]
    return rho[1:-1, 0:-2]


def grid_deposition_3d(file, quants, q1_grid=False, q2_grid=False, q3_grid=False, filter=False, sample=1, verbose=False):
    logical_filter = False
    qt_bk = quants
    qran = len(quants)
    if qran != 3:
        print("3D deposit requires three quantities. Aborting...")
        sys.exit()
    if filter is not False:
        if type(filter) is not dict:
            print('The filter must be a dictionary with the following format {"quants": ["x1", "x2"], "min": [-0.1, -0.1], "max": [0.1, 0.1]}')
            sys.exit()
        else:
            logical_filter = True
            num_filter = len(filter["quants"])
            quants.extend(filter["quants"])
            filter_min = filter["min"]
            filter_max = filter["max"]
    LP = -1
    UP = 1

    def spline_s2(x):
        t0 = 0.5e0 - x
        t1 = 0.5e0 + x
        return np.array([0.5 * t0 ** 2, 0.5 + t0 * t1, 0.5 * t1 ** 2], dtype=np.float16)

    f = h5py.File(file, "r+")
    attrs = {}
    for i in range(len(f.attrs)):
        attrs[keys(f.attrs)[i]] = values(f.attrs)[i]

    attrs["TYPE"] = "grid"
    k = keys(f)
    if "SIMULATION" in k:
        attrs1 = attrs
        attrs2 = f.get("SIMULATION").attrs
        attrs = {}
        for i in range(len(attrs1)):
            attrs[keys(attrs1)[i]] = values(attrs1)[i]
        for i in range(len(attrs2)):
            attrs[keys(attrs2)[i]] = values(attrs2)[i]
    q0 = f.get("q")[::sample]
    q1 = f.get(quants[0])[::sample]
    q2 = f.get(quants[1])[::sample]
    q3 = f.get(quants[2])[::sample]

    if logical_filter is True:
        qfil = []
        for i in range(num_filter):
            qfil.append(f.get(quants[i + 3])[::sample])
        for i in range(num_filter):
            ind1 = np.where((qfil[i] > filter_min[i]) & (qfil[i] < filter_max[i]))
            q0 = q0[ind1]
            q1 = q1[ind1]
            q2 = q2[ind1]
            q3 = q3[ind1]
            for j in range(num_filter):
                qfil[j] = qfil[j][ind1]

    if q1_grid is False:
        q1_grid = [0, 0, 0]
        q1_grid[0] = np.min(q1)
        q1_grid[1] = np.max(q1)
        q1_grid[2] = 100
    else:
        if len(q1_grid) != 3:
            print("q1_grid must have three elements (start, end, number of grid points). Aborting...")
            sys.exit()

        ind1 = np.where((q1 > q1_grid[0]) & (q1 < q1_grid[1]))
        q0 = q0[ind1]
        q1 = q1[ind1]
        q2 = q2[ind1]
        q3 = q3[ind1]

    if q2_grid is False:
        q2_grid = [0, 0, 0]
        q2_grid[0] = np.min(q2)
        q2_grid[1] = np.max(q2)
        q2_grid[2] = 100
    else:
        if len(q2_grid) != 3:
            print("q2_grid must have three elements (start, end, number of grid points). Aborting...")
            sys.exit()

        ind1 = np.where((q2 > q2_grid[0]) & (q2 < q2_grid[1]))
        q0 = q0[ind1]
        q1 = q1[ind1]
        q2 = q2[ind1]
        q3 = q3[ind1]

    if q3_grid is False:
        q3_grid = [0, 0, 0]
        q3_grid[0] = np.min(q3)
        q3_grid[1] = np.max(q3)
        q3_grid[2] = 100
    else:
        if len(q3_grid) != 3:
            print("q3_grid must have three elements (start, end, number of grid points). Aborting...")
            sys.exit()

        ind1 = np.where((q3 > q3_grid[0]) & (q3 < q3_grid[1]))
        q0 = q0[ind1]
        q1 = q1[ind1]
        q2 = q2[ind1]
        q3 = q3[ind1]
    quants = qt_bk
    x1range = np.linspace(q1_grid[0], q1_grid[1], q1_grid[2])
    x2range = np.linspace(q2_grid[0], q2_grid[1], q2_grid[2])
    x3range = np.linspace(q3_grid[0], q3_grid[1], q3_grid[2])
    nx = np.array([x1range.shape[0], x2range.shape[0], x3range.shape[0]])
    dx1 = x1range[1] - x1range[0]
    dx2 = x2range[1] - x2range[0]
    dx3 = x3range[1] - x3range[0]
    npar = len(q0)
    print("Total number of particles to be deposited ", npar)
    rho = np.zeros((nx[-1] + 2, nx[-2] + 2, nx[-3] + 2))
    if verbose is True:
        for n in tqdm(range(npar)):
            qq = np.abs(q0[n])
            x_1 = q1[n]
            x_2 = q2[n]
            x_3 = q3[n]
            nx1 = int(np.floor((x_1 - x1range[0]) / dx1))
            nx2 = int(np.floor((x_2 - x2range[0]) / dx2))
            nx3 = int(np.floor((x_3 - x3range[0]) / dx3))
            deltax1 = np.float16((x_1 - x1range[nx1]) / dx1 - 0.5)
            deltax2 = np.float16((x_2 - x2range[nx2]) / dx2 - 0.5)
            deltax3 = np.float16((x_3 - x3range[nx3]) / dx3 - 0.5)
            w1 = spline_s2(deltax1)
            w2 = spline_s2(deltax2)
            w3 = spline_s2(deltax3)
            for k3 in range(3):
                for k2 in range(3):
                    for k1 in range(3):
                        rho[nx3 + k3, nx2 + k2, nx1 + k1] += qq * w1[k1] * w2[k2] * w3[k3]
        return rho[1:-1, 1:-1, 1:-1], x1range, x2range, x3range, attrs
    else:
        for n in range(npar):
            qq = np.abs(q0[n])
            x_1 = q1[n]
            x_2 = q2[n]
            x_3 = q3[n]
            nx1 = int(np.floor((x_1 - x1range[0]) / dx1))
            nx2 = int(np.floor((x_2 - x2range[0]) / dx2))
            nx3 = int(np.floor((x_3 - x3range[0]) / dx3))
            deltax1 = np.float16((x_1 - x1range[nx1]) / dx1 - 0.5)
            deltax2 = np.float16((x_2 - x2range[nx2]) / dx2 - 0.5)
            deltax3 = np.float16((x_3 - x3range[nx3]) / dx3 - 0.5)
            w1 = spline_s2(deltax1)
            w2 = spline_s2(deltax2)
            w3 = spline_s2(deltax3)
            for k3 in range(3):
                for k2 in range(3):
                    for k1 in range(3):
                        rho[nx3 + k3, nx2 + k2, nx1 + k1] += qq * w1[k1] * w2[k2] * w3[k3]
        return rho[1:-1, 1:-1, 1:-1], x1range, x2range, x3range, attrs


def grid_deposition_2d(file, quants=["x1", "x2"], q1_grid=False, q2_grid=False, filter=False, sample=1, verbose=False):
    logical_filter = False
    qt_bk = quants[:]
    qran = len(quants)
    if qran != 2:
        print("2D deposit requires two quantities. Aborting...")
        sys.exit()
    if filter is not False:
        if type(filter) is not dict:
            print('The filter must be a dictionary with the following format {"quants": ["x1", "x2"], "min": [-0.1, -0.1], "max": [0.1, 0.1]}')
            sys.exit()
        else:
            logical_filter = True
            num_filter = len(filter["quants"])
            quants.extend(filter["quants"])
            filter_min = filter["min"]
            filter_max = filter["max"]
    LP = -1
    UP = 1

    def spline_s2(x):
        t0 = 0.5e0 - x
        t1 = 0.5e0 + x
        return np.array([0.5 * t0 ** 2, 0.5 + t0 * t1, 0.5 * t1 ** 2], dtype=np.float16)

    f = h5py.File(file, "r+")
    attrs = {}
    for i in range(len(f.attrs)):
        attrs[keys(f.attrs)[i]] = values(f.attrs)[i]

    attrs["TYPE"] = "grid"
    k = keys(f)
    if "SIMULATION" in k:
        attrs1 = attrs
        attrs2 = f.get("SIMULATION").attrs
        attrs = {}
        for i in range(len(attrs1)):
            attrs[keys(attrs1)[i]] = values(attrs1)[i]
        for i in range(len(attrs2)):
            attrs[keys(attrs2)[i]] = values(attrs2)[i]
    q0 = f.get("q")[::sample]
    q1 = f.get(quants[0])[::sample]
    q2 = f.get(quants[1])[::sample]

    if logical_filter is True:
        qfil = []
        for i in range(num_filter):
            qfil.append(f.get(quants[i + 2])[::sample])
        for i in range(num_filter):
            ind1 = np.where((qfil[i] > filter_min[i]) & (qfil[i] < filter_max[i]))
            q0 = q0[ind1]
            q1 = q1[ind1]
            q2 = q2[ind1]
            for j in range(num_filter):
                qfil[j] = qfil[j][ind1]

    if q1_grid is False:
        q1_grid = [0, 0, 0]
        q1_grid[0] = np.min(q1)
        q1_grid[1] = np.max(q1)
        q1_grid[2] = 100
    else:
        if len(q1_grid) != 3:
            print("q1_grid must have three elements (start, end, number of grid points). Aborting...")
            sys.exit()

        ind1 = np.where((q1 > q1_grid[0]) & (q1 < q1_grid[1]))
        q0 = q0[ind1]
        q1 = q1[ind1]
        q2 = q2[ind1]

    if q2_grid is False:
        q2_grid = [0, 0, 0]
        q2_grid[0] = np.min(q2)
        q2_grid[1] = np.max(q2)
        q2_grid[2] = 100
    else:
        if len(q2_grid) != 3:
            print("q2_grid must have three elements (start, end, number of grid points). Aborting...")
            sys.exit()

        ind1 = np.where((q2 > q2_grid[0]) & (q2 < q2_grid[1]))
        q0 = q0[ind1]
        q1 = q1[ind1]
        q2 = q2[ind1]

    quants = qt_bk[:]
    x1range = np.linspace(q1_grid[0], q1_grid[1], q1_grid[2])
    x2range = np.linspace(q2_grid[0], q2_grid[1], q2_grid[2])
    nx = np.array([x1range.shape[0], x2range.shape[0]])
    dx1 = x1range[1] - x1range[0]
    dx2 = x2range[1] - x2range[0]
    npar = len(q0)
    print("Total number of particles to be deposited ", npar)
    rho = np.zeros((nx[-1] + 2, nx[-2] + 2))
    if verbose is True:
        for n in tqdm(range(npar)):
            qq = np.abs(q0[n])
            x_1 = q1[n]
            x_2 = q2[n]
            nx1 = int(np.floor((x_1 - x1range[0]) / dx1))
            nx2 = int(np.floor((x_2 - x2range[0]) / dx2))
            deltax1 = np.float16((x_1 - x1range[nx1]) / dx1 - 0.5)
            deltax2 = np.float16((x_2 - x2range[nx2]) / dx2 - 0.5)
            w1 = spline_s2(deltax1)
            w2 = spline_s2(deltax2)
            for k2 in range(3):
                for k1 in range(3):
                    rho[nx2 + k2, nx1 + k1] += qq * w1[k1] * w2[k2]
        return rho[1:-1, 1:-1], x1range, x2range, attrs
    else:
        for n in range(npar):
            qq = np.abs(q0[n])
            x_1 = q1[n]
            x_2 = q2[n]
            nx1 = int(np.floor((x_1 - x1range[0]) / dx1))
            nx2 = int(np.floor((x_2 - x2range[0]) / dx2))
            deltax1 = np.float16((x_1 - x1range[nx1]) / dx1 - 0.5)
            deltax2 = np.float16((x_2 - x2range[nx2]) / dx2 - 0.5)
            w1 = spline_s2(deltax1)
            w2 = spline_s2(deltax2)
            for k2 in range(3):
                for k1 in range(3):
                    rho[nx2 + k2, nx1 + k1] += qq * w1[k1] * w2[k2]
        return rho[1:-1, 1:-1], x1range, x2range, attrs
