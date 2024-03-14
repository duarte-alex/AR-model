from utilities.osiris.open import osiris_open_grid_data
import numpy as np
from utilities.find import find_nearest
from utilities.osiris.save import osiris_save_grid
import sys
from utilities.path import ospathup, ossplit
from utilities.ask import askfolderexists_create, askexists_skip
import multiprocessing as mp
from tqdm import tqdm


def slice_3D_to_2D(file, dir, pos=False, folderout=False):

    attrs, axes, data = osiris_open_grid_data(file)
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    x1 = np.linspace(ax1[0], ax1[1], data.shape[-1], endpoint=False)
    dx1 = x1[1] - x1[0]
    x1 += dx1 / 2
    x2 = np.linspace(ax2[0], ax2[1], data.shape[-2], endpoint=False)
    dx2 = x2[1] - x2[0]
    x2 += dx2 / 2
    x3 = np.linspace(ax3[0], ax3[1], data.shape[-3], endpoint=False)
    dx3 = x3[1] - x3[0]
    x3 += dx3 / 2

    attrs2 = {}
    for j in list(attrs.keys()):
        attrs2[j] = attrs[j]

    if dir == 1:
        axis1 = ax2
        axis2 = ax3
        rs = data.shape[-1]
    elif dir == 2:
        axis1 = ax1
        axis2 = ax3
        rs = data.shape[-2]
    elif dir == 3:
        axis1 = ax1
        axis2 = ax2
        rs = data.shape[-3]
    else:
        print("Dir must be 1, 2, or 3. Aborting...")
        sys.exit()

    if not folderout:
        folderout = ospathup(file, n=1) + "slice/"
    askfolderexists_create(folderout)

    def mainloop(ind, all):
        if all == True:
            fileout = ossplit(file)[1][:-3] + "-x{:d}-".format(dir) + "slice-{:06d}".format(ind)
            attrs2["ITER"] = ind

        else:
            fileout = "x{:d}-".format(dir) + "slice-{:06d}".format(ind) + "-" + ossplit(file)[1][:-3]

        if askexists_skip(folderout + fileout + ".h5"):
            return
        # print("Writing file", fileout, "@", folderout)

        if all == True:
            if dir == 1:
                slice = data_[:, :, ind]
            elif dir == 2:
                slice = data_[:, ind, :]
            elif dir == 3:
                slice = data_[ind, :, :]
        else:
            if dir == 1:
                slice = data[:, :, ind]
            elif dir == 2:
                slice = data[:, ind, :]
            elif dir == 3:
                slice = data[ind, :, :]

        osiris_save_grid(
            folder=folderout,
            filename=fileout,
            attrs=attrs2,
            data=slice,
            dataset_name=data.name[1:],
            data_attrs=data.attrs,
            axis1=axis1,
            ax1attrs=axis1.attrs,
            axis2=axis2,
            ax2attrs=axis2.attrs,
        )

    if pos is False:
        #        with mp.Pool(processes=3) as pool:
        #            pool.map(mainloop, range(rs))
        print("Opening 3D data...")
        data_ = data[:]
        print("Done. Starting mainloop.")
        for i in tqdm(range(rs)):
            mainloop(i, True)
    else:
        if dir == 1:
            ind = find_nearest(x1, pos)
        elif dir == 2:
            ind = find_nearest(x2, pos)
        elif dir == 3:
            ind = find_nearest(x3, pos)

        mainloop(ind, False)


def bin_ndarray(ndarray, new_shape, operation="mean"):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    operation = operation.lower()
    if not operation in ["sum", "mean"]:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape, new_shape))
    compression_pairs = [(d, c // d) for d, c in zip(new_shape, ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1 * (i + 1))
    return ndarray


def normalize_slices_3D(file, dir, folderout=False):

    attrs, axes, data = osiris_open_grid_data(file)
    data_ = bin_ndarray(data[:], new_shape=(500, 500, 500))
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    if dir == 1:
        rs = data_.shape[-1]
    elif dir == 2:
        rs = data_.shape[-2]
    elif dir == 3:
        rs = data_.shape[-3]
    else:
        print("Dir must be 1, 2, or 3. Aborting...")
        sys.exit()

    if not folderout:
        folderout = ospathup(file, n=1) + "normalized/"
    askfolderexists_create(folderout)
    fileout = ossplit(file)[1]
    for i in tqdm(range(rs)):
        if dir == 1:
            data_[:, :, i] = data_[:, :, i] / np.max(data_[:, :, i])
        elif dir == 2:
            data_[:, i, :] = data_[:, i, :] / np.max(data_[:, i, :])
        elif dir == 3:
            data_[i, :, :] = data_[i, :, :] / np.max(data_[i, :, :])

    osiris_save_grid(
        folder=folderout,
        filename=fileout,
        attrs=attrs,
        data=data_,
        dataset_name=data.name[1:],
        data_attrs=data.attrs,
        axis1=ax1,
        ax1attrs=ax1.attrs,
        axis2=ax2,
        ax2attrs=ax2.attrs,
        axis3=ax3,
        ax3attrs=ax3.attrs,
    )

    return


# file = "/Volumes/Drobo/Maser/Simulations/Ionization/OAM/01_00/MS/PHA/x1p3p2_|charge|/electrons/x1p3p2_|charge|-electrons-000100.h5"

# normalize_slices_3D(file, 3)
