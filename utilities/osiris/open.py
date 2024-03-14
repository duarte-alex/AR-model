# %%
import h5py
import numpy as np
import sys
import os
from utilities.simplify_attrs import *
from utilities.find import find_nearest
from utilities.osiris import zdf

# %%


def osiris_open_grid_data(file):
    if file[-4:] == ".zdf":
        (data, info) = zdf.read(file)
        name = info.grid.__dict__["label"]
        time = info.iteration.__dict__["t"]
        iter = info.iteration.__dict__["n"]
        tunits = info.iteration.__dict__["tunits"]
        attrs = {"NAME": name, "TYPE": "grid", "TIME": time, "ITER": iter, "DT": 0.0, "TIME UNITS": tunits}
        ngrid = info.grid.__dict__["ndims"]
        axis = []

        class Attributevar(list):
            pass

        data2 = Attributevar(data)
        data2.attrs = {"UNITS": info.grid.__dict__["units"], "LONG_NAME": info.grid.__dict__["label"]}
        data2.shape = data.shape

        for i in range(ngrid):
            aux = info.grid.axis[i].__dict__
            ax_attrs = {"TYPE": "linear", "UNITS": aux["units"], "NAME": aux["label"], "LONG_NAME": aux["label"]}
            ax_vals = [aux["min"], aux["max"]]
            axq = Attributevar(ax_vals)
            axq.attrs = ax_attrs
            axis.append(axq)
        return attrs, axis, data2
    else:
        f = h5py.File(file, "r+")
        attrs = f.attrs
        k = keys(f)
        simtrue = False
        if "SIMULATION" in k:
            simtrue = True
            attrs1 = attrs
            attrs2 = f.get("SIMULATION").attrs
            attrs = {}
            for i in range(len(attrs1)):
                attrs[keys(attrs1)[i]] = values(attrs1)[i]
            for i in range(len(attrs2)):
                attrs[keys(attrs2)[i]] = values(attrs2)[i]
        ax = f.get(keys(f)[0])
        leanx = len(ax)
        axis = []
        for i in range(leanx):
            axis.append(ax.get(keys(ax)[i]))
        if simtrue:
            data = f.get(keys(f)[2])
            data.attrs["UNITS"] = attrs1["UNITS"]
            data.attrs["LONG_NAME"] = attrs1["LABEL"]
        else:
            data = f.get(keys(f)[1])
        return attrs, axis, data


# return attrs,axis,data


def osiris_open_particle_data(file, quants):
    f = h5py.File(file, "r")
    attrs = f.attrs
    k = keys(f)
    if "SIMULATION" in k:
        attrs1 = attrs
        attrs2 = f.get("SIMULATION").attrs
        attrs = {}
        for i in range(len(attrs1)):
            attrs[keys(attrs1)[i]] = values(attrs1)[i]
        for i in range(len(attrs2)):
            attrs[keys(attrs2)[i]] = values(attrs2)[i]
    qran = len(quants)
    data = []
    for i in range(qran):
        data.append(f.get(quants[i]))
    return attrs, data


# return attrs,data


def osiris_open_particle_data_all(file):
    f = h5py.File(file, "r")
    attrs = f.attrs
    k = keys(f)
    if "SIMULATION" in k:
        attrs1 = attrs
        attrs2 = f.get("SIMULATION").attrs
        attrs = {}
        for i in range(len(attrs1)):
            attrs[keys(attrs1)[i]] = values(attrs1)[i]
        for i in range(len(attrs2)):
            attrs[keys(attrs2)[i]] = values(attrs2)[i]
        # print(k)
        quants = [k[i] for i in range(1, len(k))]
        qran = len(quants)
        data = []
        for i in range(qran):
            data.append(f.get(quants[i]))
            # print(f.get(quants[i]).attrs)
        return quants, attrs, data
    else:
        quants = [i for i in k]
        qran = len(quants)
        data = []
        for i in range(qran):
            data.append(f.get(quants[i]))
        return quants, attrs, data


def osiris_open_tracks(file):
    f = h5py.File(file, "r")
    attrs = f.attrs
    data = f.get("data")
    itermap = f.get("itermap")
    return attrs, data, itermap


def osiris_open_tracks_allpar_1time(file, quants, t=0.0):
    lenq = len(quants)
    f = h5py.File(file, "r")
    k = keys(f)
    attrs = f.attrs
    if "SIMULATION" in k:
        simtrue = True
        attrs1 = attrs
        attrs2 = f.get("SIMULATION").attrs
        attrs = {}
        for i in range(len(attrs1)):
            attrs[keys(attrs1)[i]] = values(attrs1)[i]
        for i in range(len(attrs2)):
            attrs[keys(attrs2)[i]] = values(attrs2)[i]
    data = f.get("data")
    datash = data.shape
    dt = attrs["DT"][0]
    npar = attrs["NTRACKS"][0]
    ndump = attrs["NDUMP"][0]
    quants_file = list(attrs["QUANTS"])
    quants_file = [str(i, "utf-8") for i in quants_file]
    indx = []
    for i in range(lenq):
        try:
            indx.append(quants_file.index(quants[i]))
        except ValueError:
            print("Invalid quantity", quants[i], ". Aborting...")
            sys.exit()
    indx = [i - 1 for i in indx]
    tmin = np.min(data[:, 0])
    tmax = np.max(data[:, 0])
    num = (tmax - tmin) / dt + 1
    if t > tmax or t < tmin:
        print("Time out of the range of the tracks. Aborting...")
        sys.exit()
    idx = find_nearest(data[:, 0], t)
    #    if abs(data[idx,0]-data[idx+ndump,0])>dt/2:
    #        ndump+=1
    #    if abs(data[idx,0]-data[idx+ndump,0])>dt/2:
    #        print('Error calculating the separation between the tracks. Aborting...')
    #        sys.exit()
    time = data[idx, 0]
    ind = np.where((data[:, 0] < time + dt / 4) & (data[:, 0] > time - dt / 4))[0]
    indlen = len(ind)
    #    ind=[]
    #    for i in range(npar):
    #        ind.append(idx)
    #        idx+=ndump
    v = np.zeros((indlen, lenq))
    for tt in range(indlen):
        v[tt, :] = data[ind[tt], indx[:]]
    vecs = [v[:, i] for i in range(lenq)]
    return vecs


# return data


def osiris_open_tracks_1par_alltimes(file, quants, tag=None, x1=None, x2=None, x3=None, p1=None, p2=None, p3=None, t=0.0):
    lenq = len(quants)
    f = h5py.File(file, "r")
    attrs = f.attrs
    k = keys(f)
    if "SIMULATION" in k:
        simtrue = True
        attrs1 = attrs
        attrs2 = f.get("SIMULATION").attrs
        attrs = {}
        for i in range(len(attrs1)):
            attrs[keys(attrs1)[i]] = values(attrs1)[i]
        for i in range(len(attrs2)):
            attrs[keys(attrs2)[i]] = values(attrs2)[i]
    quants_file = list(attrs["QUANTS"])
    quants_file = [str(i, "utf-8") for i in quants_file]
    if tag == x1 == x2 == x3 == p1 == p2 == p3 == None:
        tag = 1
    elif not tag == None:
        pass
    else:
        xx = [x1, x2, x3, p1, p2, p3]
        lx = []
        if not x1 == None:
            lx.append("x1")
        if not x2 == None:
            try:
                quants_file.index("x2")
                lx.append("x2")
            except ValueError:
                print("A value of x2 was given, but the simulations is not 2D or 3D. Aborting...")
                sys.exit()
        if not x3 == None:
            try:
                quants_file.index("x3")
                lx.append("x3")
            except ValueError:
                print("A value of x3 was given, but the simulations is not 3D. Aborting...")
                sys.exit()
        if not p1 == None:
            lx.append("p1")
        if not p2 == None:
            lx.append("p2")
        if not p3 == None:
            lx.append("p3")
        print(lx)
        data_0 = osiris_open_tracks_allpar_1time(file, lx, t)
        val_x = [v for v in xx if v is not None]
        data_0 = np.array(data_0[:])
        data_min = np.zeros(data_0.shape[1])
        diff = np.zeros(data_0.shape[1])
        for i in range(len(lx)):
            data_temp = data_0[i, :]
            diff[:] = (data_temp[:] - val_x[i]) ** 2
            data_min[:] = data_min[:] + diff[:]
        ###
        ### NEEDS TO BE MADE MORE GENERAL FOR t!=0
        tag = find_nearest(data_min, 0) + 1
        ###
        print("tag =", tag)
    npar = attrs["NTRACKS"][0]
    ndump = attrs["NDUMP"][0]
    dt = attrs["DT"][0]
    if tag > npar - 1:
        print("Tag out of range. Valid value range is 1 to ", npar, "Aborting...")
        sys.exit()
    indx = []
    for i in range(lenq):
        try:
            indx.append(quants_file.index(quants[i]))
        except ValueError:
            print("Invalid quantity", quants[i], ". Aborting...")
            sys.exit()
    data = f.get("data")
    itermap = f.get("itermap")
    data = data[:]
    itermap = itermap[:]
    itermap_t = np.where(itermap[:, 0] == tag)[0]
    nvalues = np.sum(itermap[itermap_t, 1])
    v = np.zeros((nvalues, lenq))
    idx = 0
    num = 0
    for i in itermap_t:
        # iter_0=itermap_t[0]
        start = np.sum(itermap[:i, 1])
        end = start + itermap[i, 1]
        # print(start,end)
        num += itermap[i, 1]
        for j in range(lenq):
            v[idx:num, j] = data[start:end, indx[j] - 1]
        idx = num
    vecs = [v[:, i] for i in range(lenq)]
    return vecs


def filetags(folder, time=False, start=False):
    tags = []
    if time:
        t = []
    for filename in sorted(os.listdir(folder)):
        if start and not filename.startswith(start):
            continue
        if filename.endswith(".h5"):
            filetag = os.path.splitext(filename)[0][-6:]
            tags.append(filetag)
            nametag = os.path.splitext(filename)[0][:-6]
            if time:
                f = h5py.File(folder + filename, "r")
                t.append(f.attrs["TIME"][0])
                f.close()
    if time:
        return nametag, tags, t
    else:
        return nametag, tags


# return nametag, tags


def file_name_and_tags(folder, time=False):
    nametags = []
    tags = []
    if time:
        t = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".h5") or filename.endswith(".zdf"):
            filetag = os.path.splitext(filename)[0][-6:]
            tags.append(filetag)
            nametag = os.path.splitext(filename)[0][:-6]
            nametags.append(nametag)
            if time:
                f = h5py.File(folder + filename, "r")
                t.append(f.attrs["TIME"][0])
                f.close()
    if time:
        return nametags, tags, t
    else:
        return nametags, tags


# return nametag, tags


def osiris_alg_test(file):
    f = h5py.File(file, "r")
    x2 = f.get("x2")
    x3 = f.get("x3")
    x4 = f.get("x4")
    alg = False
    if x2 == None:
        alg = "1D"
    elif x3 == None:
        nx = f.attrs["NX"][1]
        xmin = f.attrs["XMIN"][1]
        xmax = f.attrs["XMAX"][1]
        x = np.linspace(xmin, xmax, num=nx, endpoint=False)
        dx = x[1] - x[0]
        x += dx / 2
        if x[0] == 0:
            alg = "2DCyl"
        else:
            alg = "2D"
    elif x4 == None:
        alg = "3D"
    else:
        alg = "q3D"
    return alg


# return alg


def open_beam_tags(filename):
    f = h5py.File(filename, "r")
    attrs = f.attrs
    data = f.get(keys(f)[0])
    data = data[:]
    d = [(data[i, 0], data[i, 1]) for i in range(data.shape[0])]
    return attrs, d


def change_type(file, newtype="particles"):
    f = h5py.File(file, "r+")
    f.attrs["TYPE"] = newtype
    f.close()


# %%
