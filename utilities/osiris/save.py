# %%
import h5py
import numpy as np
import os
import sys
from utilities.simplify_attrs import *
import progressbar
from collections import OrderedDict
from utilities.find import find_string_match, find_nearest


def osiris_save_grid_1d(
    folder,
    filename,
    data,
    axis,
    name,
    dataset,
    units,
    longname,
    axisunits="c/\omega_p",
    axisname="x1",
    axislong="x_1",
    time=0.0,
    iter=0,
    dt=0,
    datatype="<f4",
    compression="gzip",
):
    filename = os.path.splitext(filename)[0]
    file1 = os.path.join(folder, filename + ".h5")
    f = h5py.File(file1, "w")
    f.attrs["NAME"] = str(name)
    f.attrs["TYPE"] = "grid"
    f.attrs["TIME"] = time
    f.attrs["ITER"] = iter
    f.attrs["DT"] = dt
    f.attrs["TIME UNITS"] = "1 / \omega_p"
    x = np.array([axis[0], axis[-1]])
    ax = f.create_group("AXIS")
    ax1 = ax.create_dataset("AXIS1", (2,), "<f8", data=x)
    ax1.attrs["TYPE"] = "linear"
    ax1.attrs["UNITS"] = str(axisunits)
    ax1.attrs["NAME"] = str(axisname)
    ax1.attrs["LONG_NAME"] = str(axislong)
    dat = f.create_dataset(str(dataset), data.shape, datatype, data=data, compression=compression)
    dat.attrs["UNITS"] = str(units)
    dat.attrs["LONG_NAME"] = str(longname)
    f.close()
    return


def osiris_save_grid_2d(
    folder,
    filename,
    data,
    axis1,
    axis2,
    name,
    dataset,
    units,
    longname,
    axis1units="c/\omega_p",
    axis2units="c/\omega_p",
    axis1name="x1",
    axis2name="x2",
    axis1long="x_1",
    axis2long="x_2",
    time=0.0,
    iter=0,
    dt=0,
    datatype="<f4",
    compression="gzip",
):
    filename = os.path.splitext(filename)[0]
    file1 = os.path.join(folder, filename + ".h5")
    f = h5py.File(file1, "w")
    f.attrs["NAME"] = str(name)
    f.attrs["TYPE"] = "grid"
    f.attrs["TIME"] = time
    f.attrs["ITER"] = iter
    f.attrs["DT"] = dt
    f.attrs["TIME UNITS"] = "1 / \omega_p"
    x = np.array([axis1[0], axis1[-1]])
    y = np.array([axis2[0], axis2[-1]])
    ax = f.create_group("AXIS")
    ax1 = ax.create_dataset("AXIS1", (2,), "<f8", data=x)
    ax1.attrs["TYPE"] = "linear"
    ax1.attrs["UNITS"] = str(axis1units)
    ax1.attrs["NAME"] = str(axis1name)
    ax1.attrs["LONG_NAME"] = str(axis1long)
    ax2 = ax.create_dataset("AXIS2", (2,), "<f8", data=y)
    ax2.attrs["TYPE"] = "linear"
    ax2.attrs["UNITS"] = str(axis2units)
    ax2.attrs["NAME"] = str(axis2name)
    ax2.attrs["LONG_NAME"] = str(axis2long)
    dat = f.create_dataset(str(dataset), data.shape, datatype, data=data, compression=compression)
    dat.attrs["UNITS"] = str(units)
    dat.attrs["LONG_NAME"] = str(longname)
    f.close()
    return


def osiris_save_grid_copy_attrs_2d(
    folder,
    filename,
    data,
    dataattrs,
    dataset,
    axis1,
    axis1attrs,
    axis2,
    axis2attrs,
    attrs,
    datatype="<f4",
    compression="gzip",
):
    filename = os.path.splitext(filename)[0]
    file1 = os.path.join(folder, filename + ".h5")
    f = h5py.File(file1, "w")
    lenatt = len(attrs)
    for i in range(lenatt):
        f.attrs[keys(attrs)[i]] = values(attrs)[i]
    if not compression == "None":
        f.attrs["COMPRESS"] = "TRUE"
    x = np.array([axis1[0], axis1[-1]])
    y = np.array([axis2[0], axis2[-1]])
    ax = f.create_group("AXIS")
    ax1 = ax.create_dataset("AXIS1", (2,), "<f8", data=x)
    lenatt = len(axis1attrs)
    for i in range(lenatt):
        ax1.attrs[keys(axis1attrs)[i]] = values(axis1attrs)[i]
    ax2 = ax.create_dataset("AXIS2", (2,), "<f8", data=y)
    lenatt = len(axis2attrs)
    for i in range(lenatt):
        ax2.attrs[keys(axis2attrs)[i]] = values(axis2attrs)[i]
    dat = f.create_dataset(str(dataset), data.shape, datatype, data=data, compression=compression)
    lenatt = len(dataattrs)
    for i in range(lenatt):
        dat.attrs[keys(dataattrs)[i]] = values(dataattrs)[i]
    f.close()
    return


def osiris_save_grid_copy_attrs_and_axis_2d(
    folder,
    filename,
    data,
    dataattrs,
    dataset,
    axis1,
    axis1attrs,
    axis2,
    axis2attrs,
    attrs,
    datatype="<f4",
    compression="gzip",
):
    filename = os.path.splitext(filename)[0]
    file1 = os.path.join(folder, filename + ".h5")
    f = h5py.File(file1, "w")
    lenatt = len(attrs)
    for i in range(lenatt):
        f.attrs[keys(attrs)[i]] = values(attrs)[i]
    if not compression == "None":
        f.attrs["COMPRESS"] = "TRUE"
    x = np.array([axis1[0], axis1[-1]])
    y = np.array([axis2[0], axis2[-1]])
    ax = f.create_group("AXIS")
    ax1 = ax.create_dataset("AXIS1", (2,), "<f8", data=x)
    lenatt = len(axis1attrs)
    for i in range(lenatt):
        ax1.attrs[keys(axis1attrs)[i]] = values(axis1attrs)[i]
    ax2 = ax.create_dataset("AXIS2", (2,), "<f8", data=y)
    lenatt = len(axis2attrs)
    for i in range(lenatt):
        ax2.attrs[keys(axis2attrs)[i]] = values(axis2attrs)[i]
    dat = f.create_dataset(str(dataset), data.shape, datatype, data=data, compression=compression)
    for i in dataattrs:
        dat.attrs[i] = dataattrs[i]
    f.close()
    return


def osiris_save_grid(
    folder,
    filename,
    attrs=False,
    data=False,
    dataset_name=False,
    data_attrs=False,
    axis1=False,
    ax1attrs=False,
    axis2=False,
    ax2attrs=False,
    axis3=False,
    ax3attrs=False,
    datatype="<f4",
    compression="gzip",
):
    filename = os.path.splitext(filename)[0]
    file1 = os.path.join(folder, filename + ".h5")
    f = h5py.File(file1, "w")

    #
    def attrs_org(std, input):
        attrs0 = input
        attrs = std
        try:
            for attr in attrs_0.keys():
                try:
                    attrs[attr] = attrs0[attr]
                except KeyError:
                    attrs.add(attr, attrs0[attr])
        except AttributeError:
            pass
        return attrs

    #
    attrs_0 = attrs
    attrs = OrderedDict(
        {
            "NAME": dataset_name,
            "TYPE": "grid",
            "TIME": 0.0,
            "ITER": 0,
            "DT": 0.0,
            "TIME UNITS": b"1/\\omega_p",
        }
    )
    attrs = attrs_org(attrs, attrs_0)
    for attr in attrs.keys():
        f.attrs[attr] = attrs[attr]
    #
    ax = f.create_group("AXIS")
    #
    x = np.array([axis1[0], axis1[-1]])
    ax1 = ax.create_dataset("AXIS1", (2,), "<f8", data=x)
    attrs_0 = ax1attrs
    attrs = OrderedDict({"TYPE": "linear", "UNITS": b"c/\\omega_p", "NAME": "x1", "LONG_NAME": "x_1"})
    attrs = attrs_org(attrs, attrs_0)
    for attr in attrs.keys():
        ax1.attrs[attr] = attrs[attr]
    try:
        y = np.array([axis2[0], axis2[-1]])
        ax2 = ax.create_dataset("AXIS2", (2,), "<f8", data=y)
        attrs_0 = ax2attrs
        attrs = OrderedDict(
            {
                "TYPE": "linear",
                "UNITS": b"c/\\omega_p",
                "NAME": "x2",
                "LONG_NAME": "x_2",
            }
        )
        attrs = attrs_org(attrs, attrs_0)
        for attr in attrs.keys():
            ax2.attrs[attr] = attrs[attr]
        try:
            z = np.array([axis3[0], axis3[-1]])
            ax3 = ax.create_dataset("AXIS3", (2,), "<f8", data=z)
            attrs_0 = ax3attrs
            attrs = OrderedDict(
                {
                    "TYPE": "linear",
                    "UNITS": b"c/\\omega_p",
                    "NAME": "x3",
                    "LONG_NAME": "x_3",
                }
            )
            attrs = attrs_org(attrs, attrs_0)
            for attr in attrs.keys():
                ax3.attrs[attr] = attrs[attr]
        except TypeError:
            pass
    except TypeError:
        pass
    dat = f.create_dataset(str(dataset_name), data.shape, datatype, data=data, compression=compression)
    attrs_0 = data_attrs
    attrs = OrderedDict({"UNITS": "arb.unit", "LONG_NAME": dataset_name})
    attrs = attrs_org(attrs, attrs_0)
    for attr in attrs.keys():
        dat.attrs[attr] = attrs[attr]
    f.close()
    return


def osiris_save_grid_copy_attrs_3d(
    folder,
    filename,
    data,
    dataattrs,
    dataset,
    axis1,
    axis1attrs,
    axis2,
    axis2attrs,
    axis3,
    axis3attrs,
    attrs,
    datatype="<f4",
    compression="gzip",
):
    filename = os.path.splitext(filename)[0]
    file1 = os.path.join(folder, filename + ".h5")
    f = h5py.File(file1, "w")
    lenatt = len(attrs)
    for i in range(lenatt):
        f.attrs[keys(attrs)[i]] = values(attrs)[i]
    if not compression == "None":
        f.attrs["COMPRESS"] = "TRUE"
    x = np.array([axis1[0], axis1[-1]])
    y = np.array([axis2[0], axis2[-1]])
    z = np.array([axis3[0], axis3[-1]])
    ax = f.create_group("AXIS")
    ax1 = ax.create_dataset("AXIS1", (2,), "<f8", data=x)
    lenatt = len(axis1attrs)
    for i in range(lenatt):
        ax1.attrs[keys(axis1attrs)[i]] = values(axis1attrs)[i]
    ax2 = ax.create_dataset("AXIS2", (2,), "<f8", data=y)
    lenatt = len(axis2attrs)
    for i in range(lenatt):
        ax2.attrs[keys(axis2attrs)[i]] = values(axis2attrs)[i]
    ax3 = ax.create_dataset("AXIS3", (2,), "<f8", data=z)
    lenatt = len(axis3attrs)
    for i in range(lenatt):
        ax3.attrs[keys(axis3attrs)[i]] = values(axis3attrs)[i]
    dat = f.create_dataset(str(dataset), data.shape, datatype, data=data, compression=compression)
    lenatt = len(dataattrs)
    for i in range(lenatt):
        dat.attrs[keys(dataattrs)[i]] = values(dataattrs)[i]
    f.close()
    return


def osiris_save_particle(
    folder,
    filename,
    data,
    dataset,
    dataattrs=None,
    dx=0,
    datatype=list(["<f4"]),
    attrs=None,
    compression="gzip",
):
    if attrs == None:
        print("Attribute information not given. Aborting...")
        sys.exit()
    if not len(data) == len(dataset):
        print("The data, dataset, and data attributes must be lists of same length. Aborting...")
        sys.exit()
    lendata = len(data)
    if len(datatype) != 1 and len(datatype) != lendata:
        print("Warning, the length of the datatype neither matches the length of the datasets or it is not 1. Default value will be used")
        datatype = "<f4"
        flag_datatype = False
    elif len(datatype) == lendata:
        flag_datatype = True
    else:
        datatype = datatype[0]
        flag_datatype = False
    filename = os.path.splitext(filename)[0]
    file = os.path.join(folder, filename + ".h5")
    f = h5py.File(file, "w")
    try:
        keyattrs = keys(attrs)
        valattrs = values(attrs)
    except AttributeError:
        keyattrs = [row[0] for row in attrs]
        valattrs = [row[1] for row in attrs]
    lenat = len(keyattrs)
    for n in range(lenat):
        f.attrs[keyattrs[n]] = valattrs[n]
    if not compression == None:
        f.attrs["COMPRESS"] = "TRUE"
    try:
        if not dx == 0:
            pass
    except ValueError:
        f.attrs["DX"] = dx
    for n in range(lendata):
        if flag_datatype == True:
            dat = f.create_dataset(
                dataset[n],
                shape=data[n].shape,
                dtype=datatype[n],
                data=data[n],
                compression=compression,
            )
        else:
            dat = f.create_dataset(
                dataset[n],
                shape=data[n].shape,
                dtype=datatype,
                data=data[n],
                compression=compression,
            )
        if dataattrs:
            atr = dataattrs[n]
            if not atr == None:
                for natr in range(len(atr)):
                    dat.attrs[atr[natr][0]] = atr[natr][1]
    f.close()


# Custom attrs must give information about (for example) attrs=[['NAME',str(name)],['TYPE','particles'],['TIME',0.0],['ITER',0.0],['DT',0.0],['TIME UNITS','1 / \omega_p']]


def osiris_save_particle_empty(folder, filename, dataset, dataattrs, dx=0, datatype=list(["<f4"]), attrs=None):
    if attrs == None:
        print("Attribute information not given. Aborting...")
        sys.exit()
    if not len(dataset) == len(dataattrs):
        print("The dataset and data attributes must be lists of same length. Aborting...")
        sys.exit()
    lendata = len(dataset)
    if len(datatype) != 1 and len(datatype) != lendata:
        print("Warning, the length of the datatype neither matches the length of the datasets or it is not 1. Default value will be used")
        datatype = "<f4"
        flag_datatype = False
    elif len(datatype) == lendata:
        flag_datatype = True
    else:
        datatype = datatype[0]
        flag_datatype = False
    filename = os.path.splitext(filename)[0]
    file = os.path.join(folder, filename + ".h5")
    f = h5py.File(file, "w")
    try:
        keyattrs = keys(attrs)
        valattrs = values(attrs)
    except AttributeError:
        keyattrs = [row[0] for row in attrs]
        valattrs = [row[1] for row in attrs]
    lenat = len(keyattrs)
    for n in range(lenat):
        f.attrs[keyattrs[n]] = valattrs[n]
    try:
        if not dx == 0:
            pass
    except ValueError:
        f.attrs["DX"] = dx
    for n in range(lendata):
        if flag_datatype == True:
            dat = f.create_dataset(dataset[n], shape=(), dtype=datatype[n])
        else:
            dat = f.create_dataset(dataset[n], shape=(), dtype=datatype)
        atr = dataattrs[n]
        if not atr == None:
            for natr in range(len(atr)):
                dat.attrs[atr[natr][0]] = atr[natr][1]
    f.close()


# Custom attrs must give information about (for example) attrs=[['NAME',str(name)],['TYPE','particles'],['TIME',0.0],['ITER',0.0],['DT',0.0],['TIME UNITS','1 / \omega_p']]


def osiris_save_tracks(folder, filename, data, itermap, attrs, compression="gzip"):
    filename = os.path.splitext(filename)[0]
    file1 = os.path.join(folder, filename + ".h5")
    f = h5py.File(file1, "w")
    for i in range(len(attrs)):
        f.attrs[keys(attrs)[i]] = values(attrs)[i]
    dat = f.create_dataset("data", data.shape, "<f8", data=data, compression=compression)
    ite = f.create_dataset("itermap", itermap.shape, "<i4", data=itermap, compression=compression)
    f.close()
    return


def compress_files_in_folder(directory):
    pbar = progressbar.ProgressBar()
    for file in pbar(os.listdir(directory)):
        if file.endswith(".h5"):
            file2 = directory + os.path.split(file)[0] + "tmp_" + os.path.split(file)[1]
            os.rename(directory + file, file2)
            f = h5py.File(file2, "r")
            try:
                comp = f.attrs["COMPRESS"]
                os.rename(file2, directory + file)
                continue
            except KeyError:
                q = h5py.File(directory + file, "w")
                lenat = len(keys(f.attrs))
                for n in range(lenat):
                    q.attrs[keys(f.attrs)[n]] = values(f.attrs)[n]
                q.attrs["COMPRESS"] = "TRUE"
                try:
                    f.copy("AXIS", q["/"])
                    dat = f.get(keys(f)[1])
                    ds = q.create_dataset(keys(f)[1], data=dat, compression="gzip")
                    lenat = len(keys(dat.attrs))
                    for n in range(lenat):
                        ds.attrs[keys(dat.attrs)[n]] = values(dat.attrs)[n]
                except KeyError:
                    lenk = len(keys(f))
                    try:
                        for j in range(lenk):
                            dat = f.get(keys(f)[j])
                            ds = q.create_dataset(keys(f)[j], data=dat, compression="gzip")
                            lenat = len(keys(dat.attrs))
                            for n in range(lenat):
                                ds.attrs[keys(dat.attrs)[n]] = values(dat.attrs)[n]
                    except TypeError:
                        f.close()
                        q.close()
                        os.rename(file2, directory + file)
                        continue
            f.close()
            os.remove(file2)
            q.close()


def save_beam_tags(folder, beamtag, tags, attrs):
    filename = os.path.join(folder, "tags_" + beamtag + ".h5")
    tablist = np.array(tags)
    f = h5py.File(filename, "w")
    for i in range(len(attrs)):
        f.attrs[keys(attrs)[i]] = values(attrs)[i]
    dat = f.create_dataset("tags", tablist.shape, "<i4", data=tablist)
    f.close()
    return


# %%
import matplotlib.pyplot as plt


def osiris_save_points_to_grid(
    x,
    y,
    folder,
    filename,
    xrange=[-1, 1],
    Npoints=100,
    value=-1000,
    attrs=False,
    dataset_name=False,
    data_attrs=False,
    ax1attrs=False,
):
    xmin = xrange[0]
    xmax = xrange[1]
    x1 = np.linspace(xmin, xmax, Npoints, endpoint=True)
    res = np.full(x1.shape, value)
    for i in range(x.shape[0]):
        ind = find_nearest(x1, x[i])
        res[ind] = y[i]

    osiris_save_grid(
        folder,
        filename,
        attrs=attrs,
        data=res,
        dataset_name=dataset_name,
        data_attrs=data_attrs,
        axis1=x1,
        ax1attrs=ax1attrs,
    )


# %%
