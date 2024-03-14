from utilities.osiris.open import osiris_open_grid_data, filetags
import os
from utilities.path import ospathup
import numpy as np
from utilities.osiris.save import osiris_save_grid
from utilities.ask import askfolderexists_create
from tqdm import tqdm


def join_quasi3D_modes(folder, theta=0.0):
    thstr = "{:03d}".format(int(theta))
    theta *= np.pi / 180
    folder = ospathup(folder, n=0)
    folder_list = os.listdir(folder)
    modes_folder = [i + "/" for i in folder_list if "MODE-" in i]
    n_modes = int((len(modes_folder) - 1) / 2)
    f0 = ospathup(folder + modes_folder[0], n=0)
    phys_quants = os.listdir(ospathup(f0))
    for ph in range(len(phys_quants)):
        quant = phys_quants[ph].replace("_cyl_m", "")
        fol = [folder + modes_folder[0] + phys_quants[ph] + "/"]
        fout = folder + quant + "/"
        askfolderexists_create(fout)
        for n in range(1, n_modes + 1):
            fol.append(folder + modes_folder[2 * n - 1] + phys_quants[ph] + "/")
            fol.append(folder + modes_folder[2 * n] + phys_quants[ph] + "/")
        nametags = []
        for f in fol:
            nametag, _ = filetags(f)
            nametags.append(nametag)
        _, tags = filetags(fol[0])
        for t in tqdm(tags):
            file = fol[0] + nametags[0] + t + ".h5"
            attrs, axes, data = osiris_open_grid_data(file)
            data = data[:]
            for n in range(1, n_modes + 1):
                file = fol[2 * n - 1] + nametags[2 * n - 1] + t + ".h5"
                attrs, axes, data_im = osiris_open_grid_data(file)
                data += np.sin(n * (theta + np.pi)) * data_im[:]
                file = fol[2 * n] + nametags[2 * n] + t + ".h5"
                attrs, axes, data_re = osiris_open_grid_data(file)
                data += np.cos(n * (theta + np.pi)) * data_re[:]
            filout = quant + "-theta-" + thstr + "-" + t
            osiris_save_grid(fout, filout, attrs=attrs, data=data, axis1=[axes[0][0], axes[0][1]], ax1attrs=axes[0].attrs, axis2=[axes[1][0], axes[1][1]], ax2attrs=axes[1].attrs)


folder = "/Volumes/Drobo/Fireball/Simulations/Beam/quasi3D/HollowChannel/Fireball/MS/DENSITY/pbeam/"
folder = "/Volumes/Drobo/Fireball/Simulations/Laser/quasi3D/JV/quasi3D/MS/DENSITY/electrons/"
folder = "/Volumes/Drobo/Fireball/Simulations/Laser/quasi3D/w0_045.00/MS/DENSITY/pbeam/"
folder = "/Volumes/Drobo/Fireball/Simulations/Laser/quasi3D/w0_045.00/MS/FLD/"
join_quasi3D_modes(folder)
"""
attrs, axes, data = join_quasi3D_modes(folder)
oi[0].replace("_cyl_m", "")
oi

fout = "/Users/thales/Desktop/"
filout = "electrons"
axes[0].attrs

osiris_save_grid(fout, filout, attrs=attrs, data=data, axis1=[axes[0][0], axes[0][1]], ax1attrs=axes[0].attrs, axis2=[axes[1][0], axes[1][1]], ax2attrs=axes[1].attrs)

np.max(oi)
np.min(oi)

qq.modes_folder
qq.n_modes
"""
