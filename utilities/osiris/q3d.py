# %%
import numpy as np
from utilities.osiris.open import osiris_open_grid_data
from utilities.osiris.save import osiris_save_grid
from utilities.find import find_string_match
from utilities.ask import askfolderexists_create, askexists_skip
from utilities.path import osfile, ospathup, osfolderup
import sys
import os

# %%


def join_lineout(file, theta=0.0, n_modes=1, if_save=True):
    """
    file: path to the file mode 0. The code will search the others files automatically
    theta: angle of the slice wished in degrees
    """

    if if_save:
        if "FLD" in file:
            tag = osfile(file)[-9:-3]
            _, end = find_string_match("FLD/", file)
            _, end2 = find_string_match("MODE-0-RE/", file)
            filen = file[end2 : end2 + 2]
            folderout = file[:end] + filen + "-line/"
            askfolderexists_create(folderout)
            folderout += "theta_{:s}".format(str(int(theta)).zfill(3)) + "/"
            askfolderexists_create(folderout)
            filename = (osfile(file).replace("_cyl_m", "")).replace("-0-re", "")
            if askexists_skip(os.path.join(folderout, filename)):
                return
            th = theta * np.pi / 180
            attrs, axes, data = osiris_open_grid_data(file)

            f = data[:]
            fatrss = data.attrs
            file_gen = (file.replace("MODE-0-RE", "__folder__")).replace("0-re", "__file__")

            for i in range(1, n_modes + 1):
                tt0 = "MODE-{:d}-RE".format(i)
                tt1 = "{:d}-re".format(i)
                file_re = (file_gen.replace("__folder__", tt0)).replace("__file__", tt1)
                _, _, data_re = osiris_open_grid_data(file_re)

                tt0 = "MODE-{:d}-IM".format(i)
                tt1 = "{:d}-im".format(i)
                file_im = (file_gen.replace("__folder__", tt0)).replace("__file__", tt1)
                _, _, data_im = osiris_open_grid_data(file_im)

                f += data_re[:] * np.cos(i * th) - data_im[:] * np.sin(i * th)

            osiris_save_grid(folderout, filename, attrs=attrs, data=f, dataset_name=filen, data_attrs=data.attrs, axis1=axes[0], ax1attrs=axes[0].attrs)
            return
        elif "DENSITY" in file:
            _, end = find_string_match("DENSITY/", file)
            _, end2 = find_string_match("MODE-0-RE/", file)
            pp = ospathup(file, n=3)
            filen = osfolderup(ospathup(file, n=1)).replace("_cyl_m", "")
            folderout = pp + filen + "-line/"
            askfolderexists_create(folderout)
            folderout += "theta_{:s}".format(str(int(theta)).zfill(3)) + "/"
            askfolderexists_create(folderout)
            filename = (osfile(file).replace("_cyl_m", "")).replace("-0-re", "")
            if askexists_skip(os.path.join(folderout, filename)):
                return

            th = theta * np.pi / 180
            attrs, axes, data = osiris_open_grid_data(file)

            f = data[:]
            fatrss = data.attrs
            file_gen = (file.replace("MODE-0-RE", "__folder__")).replace("0-re", "__file__")

            for i in range(1, n_modes + 1):
                tt0 = "MODE-{:d}-RE".format(i)
                tt1 = "{:d}-re".format(i)
                file_re = (file_gen.replace("__folder__", tt0)).replace("__file__", tt1)
                _, _, data_re = osiris_open_grid_data(file_re)

                tt0 = "MODE-{:d}-IM".format(i)
                tt1 = "{:d}-im".format(i)
                file_im = (file_gen.replace("__folder__", tt0)).replace("__file__", tt1)
                _, _, data_im = osiris_open_grid_data(file_im)

                f += data_re[:] * np.cos(i * th) - data_im[:] * np.sin(i * th)

            osiris_save_grid(folderout, filename, attrs=attrs, data=f, dataset_name=filen, data_attrs=data.attrs, axis1=axes[0], ax1attrs=axes[0].attrs)
            return
        else:
            print("This was not done yet because for now it should never be necessary.")
            return
    else:
        th = theta * np.pi / 180
        attrs, axes, data = osiris_open_grid_data(file)

        f = data[:]
        fatrss = data.attrs
        file_gen = (file.replace("MODE-0-RE", "__folder__")).replace("0-re", "__file__")

        for i in range(1, n_modes + 1):
            tt0 = "MODE-{:d}-RE".format(i)
            tt1 = "{:d}-re".format(i)
            file_re = (file_gen.replace("__folder__", tt0)).replace("__file__", tt1)
            _, _, data_re = osiris_open_grid_data(file_re)

            tt0 = "MODE-{:d}-IM".format(i)
            tt1 = "{:d}-im".format(i)
            file_im = (file_gen.replace("__folder__", tt0)).replace("__file__", tt1)
            _, _, data_im = osiris_open_grid_data(file_im)

            f += data_re[:] * np.cos(i * th) - data_im[:] * np.sin(i * th)
        return attrs, axes, f, fatrss


def convert_to_2D(file, theta=0.0, n_modes=1, if_save=True):
    """
    file: path to the file mode 0. The code will search the others files automatically
    theta: angle of the slice wished in degrees
    """

    if if_save:
        if "FLD" in file:
            tag = osfile(file)[-9:-3]
            _, end = find_string_match("FLD/", file)
            _, end2 = find_string_match("MODE-0-RE/", file)
            filen = file[end2 : end2 + 2]
            folderout = file[:end] + filen + "/"
            askfolderexists_create(folderout)
            folderout += "theta_{:s}".format(str(int(theta)).zfill(3)) + "/"
            askfolderexists_create(folderout)
            filename = filen + "-" + tag + ".h5"
            if askexists_skip(os.path.join(folderout, filename)):
                return

            th = theta * np.pi / 180
            attrs, axes, data = osiris_open_grid_data(file)

            f = data[:]
            fatrss = data.attrs
            file_gen = (file.replace("MODE-0-RE", "__folder__")).replace("0-re", "__file__")

            for i in range(1, n_modes + 1):
                tt0 = "MODE-{:d}-RE".format(i)
                tt1 = "{:d}-re".format(i)
                file_re = (file_gen.replace("__folder__", tt0)).replace("__file__", tt1)
                _, _, data_re = osiris_open_grid_data(file_re)

                tt0 = "MODE-{:d}-IM".format(i)
                tt1 = "{:d}-im".format(i)
                file_im = (file_gen.replace("__folder__", tt0)).replace("__file__", tt1)
                _, _, data_im = osiris_open_grid_data(file_im)

                f += data_re[:] * np.cos(i * th) - data_im[:] * np.sin(i * th)

            osiris_save_grid(folderout, filename, attrs=attrs, data=f, dataset_name=filen, data_attrs=data.attrs, axis1=axes[0], ax1attrs=axes[0].attrs, axis2=axes[1], ax2attrs=axes[1].attrs)

            return

        elif "DENSITY" in file:
            tag = osfile(file)[-9:-3]
            _, end = find_string_match("DENSITY/", file)
            _, end2 = find_string_match("MODE-0-RE/", file)
            pp = ospathup(file, n=3)
            filen = osfolderup(ospathup(file, n=1)).replace("_cyl_m", "")
            folderout = pp + filen + "/"
            askfolderexists_create(folderout)
            folderout += "theta_{:s}".format(str(int(theta)).zfill(3)) + "/"
            askfolderexists_create(folderout)
            filename = filen + "-" + tag + ".h5"
            if askexists_skip(os.path.join(folderout, filename)):
                return

            th = theta * np.pi / 180
            attrs, axes, data = osiris_open_grid_data(file)

            f = data[:]
            fatrss = data.attrs
            file_gen = (file.replace("MODE-0-RE", "__folder__")).replace("0-re", "__file__")

            for i in range(1, n_modes + 1):
                tt0 = "MODE-{:d}-RE".format(i)
                tt1 = "{:d}-re".format(i)
                file_re = (file_gen.replace("__folder__", tt0)).replace("__file__", tt1)
                _, _, data_re = osiris_open_grid_data(file_re)

                tt0 = "MODE-{:d}-IM".format(i)
                tt1 = "{:d}-im".format(i)
                file_im = (file_gen.replace("__folder__", tt0)).replace("__file__", tt1)
                _, _, data_im = osiris_open_grid_data(file_im)

                f += data_re[:] * np.cos(i * th) - data_im[:] * np.sin(i * th)

            osiris_save_grid(folderout, filename, attrs=attrs, data=f, dataset_name=filen, data_attrs=data.attrs, axis1=axes[0], ax1attrs=axes[0].attrs, axis2=axes[1], ax2attrs=axes[1].attrs)
            return
        else:
            print("This was not done yet because for now it should never be necessary.")
            return
    else:
        th = theta * np.pi / 180
        attrs, axes, data = osiris_open_grid_data(file)

        f = data[:]
        fatrss = data.attrs
        file_gen = (file.replace("MODE-0-RE", "__folder__")).replace("0-re", "__file__")

        for i in range(1, n_modes + 1):
            tt0 = "MODE-{:d}-RE".format(i)
            tt1 = "{:d}-re".format(i)
            file_re = (file_gen.replace("__folder__", tt0)).replace("__file__", tt1)
            _, _, data_re = osiris_open_grid_data(file_re)

            tt0 = "MODE-{:d}-IM".format(i)
            tt1 = "{:d}-im".format(i)
            file_im = (file_gen.replace("__folder__", tt0)).replace("__file__", tt1)
            _, _, data_im = osiris_open_grid_data(file_im)

            f += data_re[:] * np.cos(i * th) - data_im[:] * np.sin(i * th)

        return attrs, axes, f, fatrss


# %%
# file = "/Volumes/Drobo/Maser/Simulations/SelfConsistent/He_SelfConsistent/q3d/LargerBox/MS/FLD/MODE-0-RE/e3_cyl_m-line/e3_cyl_m-line-0-re-x2-01-011026.h5"
# file = "/Volumes/Drobo/Maser/Simulations/SelfConsistent/He_SelfConsistent/q3d/LargerBox/MS/FLD/__folder__/e3_cyl_m-line/e3_cyl_m-line-__file__-x2-01-000000.h5"

# join_lineout(file, if_save=True)

# %%
