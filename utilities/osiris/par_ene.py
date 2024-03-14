# %%

from utilities.find import find_string_match
from utilities.path import ospathup
from utilities.ask import askfolderexists_create
import numpy as np
from utilities.osiris.save import osiris_save_grid
from utilities.osiris.open import osiris_open_particle_data, osiris_alg_test, filetags
import sys
from tqdm import tqdm
import glob

# %%


def par_ene_raw(folder, x1range=False, x2range=False):
    filename, tags = filetags(folder)
    file = folder + filename + tags[0] + ".h5"
    alg = osiris_alg_test(file)
    quants = ["ene", "q"]
    if x2range:
        quants += ["x1", "x2"]
    elif x1range:
        quants += ["x1"]
    attrs, data = osiris_open_particle_data(file, quants)
    xmin = attrs["XMIN"]
    xmax = attrs["XMAX"]
    nx = attrs["NX"]
    lentags = len(tags)
    enepar = np.zeros(lentags)
    if alg == "1D":
        x1 = np.linspace(xmin[0], xmax[0], num=nx[0], endpoint=False)
        dx1 = x1[1] - x1[0]
        for t in tqdm(range(lentags)):
            file = folder + filename + tags[t] + ".h5"
            attrs, data = osiris_open_particle_data(file, quants)
            try:
                ene = data[0][:]
                q = data[1][:]
                if x1range:
                    x1 = data[2][:]
                    ind = np.where((x1 < x1range[1]) & (x1 > x1range[0]))
                    ene = ene[ind]
                    q = q[ind]
                enepar[t] = np.abs(np.sum(ene * q)) * dx1
            except ValueError:
                enepar[t] = 0
    elif alg == "2DCyl":
        x1 = np.linspace(xmin[0], xmax[0], num=nx[0], endpoint=False)
        dx1 = x1[1] - x1[0]
        x2 = np.linspace(xmin[1], xmax[1], num=nx[1], endpoint=False)
        dx2 = x2[1] - x2[0]
        for t in tqdm(range(lentags)):
            file = folder + filename + tags[t] + ".h5"
            attrs, data = osiris_open_particle_data(file, quants)
            try:
                ene = data[0][:]
                q = data[1][:]
                if x1range:
                    x1 = data[2][:]
                    if x2range:
                        x2 = data[3][:]
                        ind = np.where((x1 < x1range[1]) & (x1 > x1range[0]) & (x2 < x2range[1]) & (x2 > x2range[0]))
                        ene = ene[ind]
                        q = q[ind]
                    else:
                        ind = np.where((x1 < x1range[1]) & (x1 > x1range[0]))
                        ene = ene[ind]
                        q = q[ind]
                enepar[t] = np.abs(np.sum(ene * q)) * dx1 * dx2 * 2 * np.pi
            except ValueError:
                enepar[t] = 0
    elif alg == "2D":
        x1 = np.linspace(xmin[0], xmax[0], num=nx[0], endpoint=False)
        dx1 = x1[1] - x1[0]
        x2 = np.linspace(xmin[1], xmax[1], num=nx[1], endpoint=False)
        dx2 = x2[1] - x2[0]
        for t in tqdm(range(lentags)):
            file = folder + filename + tags[t] + ".h5"
            attrs, data = osiris_open_particle_data(file, quants)
            try:
                ene = data[0][:]
                q = data[1][:]
                if x1range:
                    x1 = data[2][:]
                    if x2range:
                        x2 = data[3][:]
                        ind = np.where((x1 < x1range[1]) & (x1 > x1range[0]) & (x2 < x2range[1]) & (x2 > x2range[0]))
                        ene = ene[ind]
                        q = q[ind]
                    else:
                        ind = np.where((x1 < x1range[1]) & (x1 > x1range[0]))
                        ene = ene[ind]
                        q = q[ind]
                enepar[t] = np.abs(np.sum(ene * q)) * dx1 * dx2
            except ValueError:
                enepar[t] = 0
    elif alg == "3D":
        x1 = np.linspace(xmin[0], xmax[0], num=nx[0], endpoint=False)
        dx1 = x1[1] - x1[0]
        x2 = np.linspace(xmin[1], xmax[1], num=nx[1], endpoint=False)
        dx2 = x2[1] - x2[0]
        x3 = np.linspace(xmin[2], xmax[2], num=nx[2], endpoint=False)
        dx3 = x3[1] - x3[0]
        for t in tqdm(range(lentags)):
            file = folder + filename + tags[t] + ".h5"
            attrs, data = osiris_open_particle_data(file, quants)
            try:
                ene = data[0][:]
                q = data[1][:]
                if x1range:
                    x1 = data[2][:]
                    if x2range:
                        x2 = data[3][:]
                        ind = np.where((x1 < x1range[1]) & (x1 > x1range[0]) & (x2 < x2range[1]) & (x2 > x2range[0]))
                        ene = ene[ind]
                        q = q[ind]
                    else:
                        ind = np.where((x1 < x1range[1]) & (x1 > x1range[0]))
                        ene = ene[ind]
                        q = q[ind]
                enepar[t] = np.abs(np.sum(ene * q)) * dx1 * dx2 * dx3
            except ValueError:
                enepar[t] = 0
    else:
        print("Algorithm not recognized. Aborting...")
        sys.exit()
    #
    return enepar


def osiris_particle_energy(file):
    iter, time, npar, data = np.loadtxt(file, skiprows=2, unpack=True)
    if "/HIST/" in file:
        sta, end = find_string_match("HIST/", file)
        askfolderexists_create(file[:sta] + "MS/")
        folderout = file[:sta] + "MS/ENE/"
    else:
        folderout = ospathup(file) + "ENE/"
    species = np.genfromtxt(file, max_rows=1, dtype="str")
    species = species[-1].replace('"', "")
    askfolderexists_create(folderout)
    filename = species + "_ene"
    comps = species
    long_name = "\epsilon_{" + species + "}"
    attrsdata = {"UNITS": "m_ec^2", "LONG_NAME": long_name}
    axattr = {"UNITS": b"1/\\omega_p", "NAME": "t", "LONG_NAME": "Time"}
    osiris_save_grid(
        folder=folderout,
        filename=filename,
        data=data,
        dataset_name=long_name,
        data_attrs=attrsdata,
        axis1=time,
        ax1attrs=axattr,
    )


# %%
def sum_osiris_particle_energy(folder):
    file_pattern = "par*_ene"
    files = glob.glob(folder + "/" + file_pattern)
    num_files = len(files)
    time, _, data = np.loadtxt(files[0], skiprows=2, unpack=True, usecols=(1, 2, 3))
    energy = np.zeros(data.shape)
    for i in range(num_files):
        _, _, data = np.loadtxt(files[i], skiprows=2, unpack=True, usecols=(1, 2, 3))
        energy[:] += data[:]
    if "/HIST/" in folder:
        sta, end = find_string_match("HIST/", folder)
        askfolderexists_create(folder[:sta] + "MS/")
        folderout = folder[:sta] + "MS/ENE/"
    else:
        folderout = ospathup(folder) + "ENE/"
    askfolderexists_create(folderout)
    filename = "total_par_ene"
    long_name = "\epsilon_{total}"
    attrsdata = {"UNITS": "m_ec^2", "LONG_NAME": long_name}
    axattr = {"UNITS": b"1/\\omega_p", "NAME": "t", "LONG_NAME": "Time"}
    osiris_save_grid(
        folder=folderout,
        filename=filename,
        data=energy,
        dataset_name=long_name,
        data_attrs=attrsdata,
        axis1=time,
        ax1attrs=axattr,
    )


# %%
