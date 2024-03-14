from matplotlib import rcParams
from cycler import cycler
import numpy as np
from utilities.find import find_nearest


def std_plot(**kwargs):
    rcParams["font.family"] = "Helvetica"
    rcParams["mathtext.default"] = "regular"
    rcParams["svg.fonttype"] = "none"
    rcParams["axes.linewidth"] = 2
    rcParams["figure.figsize"] = [10, 8]
    rcParams["font.size"] = 23
    rcParams["lines.linewidth"] = 2
    rcParams["lines.color"] = "k"
    rcParams["xtick.direction"] = "in"
    rcParams["xtick.top"] = True
    rcParams["xtick.minor.visible"] = True
    rcParams["ytick.direction"] = "in"
    rcParams["ytick.right"] = True
    rcParams["ytick.minor.visible"] = True
    rcParams["xtick.major.size"] = 10
    rcParams["xtick.minor.size"] = 5
    rcParams["xtick.major.width"] = 2
    rcParams["xtick.minor.width"] = 2
    rcParams["ytick.major.size"] = 10
    rcParams["ytick.minor.size"] = 5
    rcParams["ytick.major.width"] = 2
    rcParams["ytick.minor.width"] = 2
    # rcParams["figure.constrained_layout.use"] = True
    # rcParams["figure.constrained_layout.h_pad"] = 0.4
    # rcParams["figure.constrained_layout.w_pad"] = 0.4
    rcParams["figure.autolayout"] = True
    rcParams["axes.prop_cycle"] = cycler("color", ["k", (0.8, 0.0, 0.0), (0.0, 0.0, 0.8), (0.0, 0.8, 0.0), (0.8, 0.8, 0.0), (0.8, 0.0, 0.8), (0.0, 0.8, 0.8), "r", "b", "g", "c", "m", "y"])
    if kwargs:
        lenkeys = len(kwargs["keys"])
        for num in range(lenkeys):
            rcParams[kwargs["keys"][num]] = kwargs["values"][num]


def color_division(color_data, n_div):
    len_cd = len(color_data)
    range_cd = np.linspace(0, 1, len_cd, endpoint=False)
    dx_cd = range_cd[1] - range_cd[0]
    range_cd += dx_cd / 2
    range_div = np.linspace(0, 1, n_div, endpoint=False)
    dx_div = range_div[1] - range_div[0]
    range_div += dx_div / 2
    color_sp = np.zeros((n_div, 3), dtype=np.int16)
    color_sp[0] = color_data[0]
    color_sp[n_div - 1] = color_data[-1]
    for i in range(1, n_div - 1):
        idx = find_nearest(range_cd, range_div[i])
        if idx == 0:
            idx_2 = 0
            color_sp[i] = color_data[idx]
        elif idx == len_cd - 1:
            idx_2 = len_cd - 1
            color_sp[i] = color_data[idx]
        else:
            idx_2 = find_nearest(np.array([range_cd[idx - 1], range_cd[idx + 1]]), range_div[i])
            if idx_2 == 0:
                idx_2 = -1
            idx_2 += idx
            fac_1 = np.abs(range_div[i] - range_cd[idx]) / dx_cd
            fac_2 = np.abs(range_div[i] - range_cd[idx_2]) / dx_cd
            color_sp[i] = np.rint(fac_1 * color_data[idx] + fac_2 * color_data[idx])
    return color_sp
