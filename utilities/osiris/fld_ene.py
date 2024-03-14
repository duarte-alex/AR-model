# %%
import numpy as np
from utilities.osiris.open import osiris_open_grid_data, filetags
from utilities.find import find_nearest, find_string_match
from utilities.ask import askfolderexists_create, askexists_skip
from utilities.osiris.save import osiris_save_grid_1d, osiris_save_grid
from utilities.path import ospathup
import progressbar
import sys
import os
from tqdm import tqdm

"""
def fld_ene_1d(file):
    attrs, axis, fld = osiris_open_grid_data(file)
    fld = fld[:]
    flsh = fld.shape
    ax1 = axis[0]
    x1 = np.linspace(ax1[0], ax1[1], fld.shape[0], endpoint=False)
    dx1 = x1[1] - x1[0]
    fld = fld ** 2
    enefld = np.sum(fld) * dx1 / 2
    return enefld  # NEVER TESTED
"""


class fld_ene_1d:
    def __init__(self, folder, x1range=None, fileout=None, if_save=False):
        self.fileout = fileout
        self.folder = folder
        self.filename, self.tags = filetags(folder)
        self.ini()
        self.idx_x1_min, self.idx_x1_max = self.range(x1range, self.x1)
        # print(self.idx_x1_min,self.idx_x1_max)
        # print(self.idx_x2_min,self.idx_x2_max)
        lentags = len(self.tags)
        self.time = np.zeros(lentags)
        self.enef = np.zeros(lentags)
        self.mainloop()
        if if_save:
            self.save()

    def ini(self):
        file_0 = self.folder + self.filename + self.tags[0] + ".h5"
        attrs, axis, fld = osiris_open_grid_data(file_0)
        self.attrs = attrs
        self.fldattrs = fld.attrs
        self.fldshape = fld.shape
        ax1 = axis[0]
        self.x1 = np.linspace(ax1[0], ax1[1], self.fldshape[0], endpoint=False)
        dx1 = self.x1[1] - self.x1[0]
        self.x1 += dx1 / 2
        self.dx1 = dx1

    def range(self, ran, x):
        if ran == None:
            return 0, len(x) + 1
        else:
            xmin = ran[0]
            if xmin == "":
                indmin = 0
            else:
                indmin = find_nearest(x, xmin)
            xmax = ran[1]
            if xmax == "":
                indmax = x.shape + 1
            else:
                indmax = find_nearest(x, xmax) + 1
            return indmin, indmax

    def mainloop(self):
        flag = -1
        # pbar = progressbar.ProgressBar()
        # for t in pbar(self.tags):
        for t in self.tags:
            flag += 1
            file = self.folder + self.filename + t + ".h5"
            attrs, axis, fld = osiris_open_grid_data(file)
            self.time[flag] = attrs["TIME"]
            fld = fld[self.idx_x1_min : self.idx_x1_max]
            fld = fld**2
            enefld = np.sum(fld) * self.dx1 / 2
            self.enef[flag] = enefld

    def save(self):
        if "/MS/" in self.folder:
            sta, end = find_string_match("/MS/", self.folder)
            folderout = self.folder[:end] + "ENE/"
        else:
            folderout = self.folder + "ENE/"
        askfolderexists_create(folderout)
        if not self.fileout:
            fileout = "ene_" + self.filename[:-1]
        else:
            fileout = self.fileout
        name = "ene_" + (self.attrs["NAME"][0]).decode("utf-8")
        dataset = name
        units = "m_ec^2"
        longname = "\epsilon_{" + (self.fldattrs["LONG_NAME"][0]).decode("utf-8") + "}"
        axisunits = "1/\omega_p"
        axisname = "t"
        axislong = "Time"
        if askexists_skip(folderout + fileout + ".h5"):
            os.remove(folderout + fileout + ".h5")
        print("Writing", fileout, "@", folderout)
        osiris_save_grid_1d(
            folderout,
            fileout,
            self.enef,
            self.time,
            name,
            dataset,
            units,
            longname,
            axisunits,
            axisname,
            axislong,
        )


# return enefld


class fld_ene_2d:
    def __init__(self, folder, cyl=False, x1range=None, x2range=None, fileout=None, if_save=False):
        self.fileout = fileout
        self.folder = folder
        self.cyl = cyl
        self.filename, self.tags = filetags(folder)
        self.ini()
        self.idx_x1_min, self.idx_x1_max = self.range(x1range, self.x1)
        self.idx_x2_min, self.idx_x2_max = self.range(x2range, self.x2)
        # print(self.idx_x1_min,self.idx_x1_max)
        # print(self.idx_x2_min,self.idx_x2_max)
        # self.tags = [self.tags[i] for i in range(0, len(self.tags), 10)]
        lentags = len(self.tags)
        self.time = np.zeros(lentags)
        self.enef = np.zeros(lentags)
        self.mainloop()
        if if_save:
            self.save()

    def ini(self):
        file_0 = self.folder + self.filename + self.tags[0] + ".h5"
        attrs, axis, fld = osiris_open_grid_data(file_0)
        self.attrs = attrs
        self.fldattrs = fld.attrs
        self.fldshape = fld.shape
        ax1 = axis[0]
        ax2 = axis[1]
        self.x1 = np.linspace(ax1[0], ax1[1], self.fldshape[1], endpoint=False)
        dx1 = self.x1[1] - self.x1[0]
        self.x1 += dx1 / 2
        self.x2 = np.linspace(ax2[0], ax2[1], self.fldshape[0], endpoint=False)
        dx2 = self.x2[1] - self.x2[0]
        self.x2 += dx2 / 2
        self.dx1 = dx1
        self.dx2 = dx2

    def range(self, ran, x):
        if ran == None:
            return 0, len(x) + 1
        else:
            xmin = ran[0]
            if xmin == "":
                indmin = 0
            else:
                indmin = find_nearest(x, xmin)
            xmax = ran[1]
            if xmax == "":
                indmax = x.shape + 1
            else:
                indmax = find_nearest(x, xmax) + 1
            return indmin, indmax

    def mainloop(self):
        flag = -1
        # pbar = progressbar.ProgressBar()
        # for t in pbar(self.tags):
        for t in tqdm(self.tags):
            flag += 1
            file = self.folder + self.filename + t + ".h5"
            attrs, axis, fld = osiris_open_grid_data(file)
            self.time[flag] = attrs["TIME"]
            fld = fld[self.idx_x2_min : self.idx_x2_max, self.idx_x1_min : self.idx_x1_max]
            fld = fld**2
            self.x2 = self.x2[self.idx_x2_min : self.idx_x2_max]
            if self.cyl:
                fld = 2 * np.pi * np.transpose(fld) * self.x2
            enefld = np.sum(fld) * self.dx1 * self.dx2 / 2
            self.enef[flag] = enefld

    def save(self):
        if "/MS/" in self.folder:
            sta, end = find_string_match("/MS/", self.folder)
            folderout = self.folder[:end] + "ENE/"
        else:
            folderout = self.folder + "ENE/"
        askfolderexists_create(folderout)
        if not self.fileout:
            fileout = self.filename[:-1] + "_ene"
        else:
            fileout = self.fileout
        name = "ene_" + (self.attrs["NAME"][0]).decode("utf-8")
        dataset = name
        units = "m_ec^2"
        longname = "\epsilon_{" + (self.fldattrs["LONG_NAME"][0]).decode("utf-8") + "}"
        axisunits = "1/\omega_p"
        axisname = "t"
        axislong = "Time"
        if askexists_skip(folderout + fileout + ".h5"):
            os.remove(folderout + fileout + ".h5")
        print("Writing", fileout, "@", folderout)
        osiris_save_grid_1d(
            folderout,
            fileout,
            self.enef,
            self.time,
            name,
            dataset,
            units,
            longname,
            axisunits,
            axisname,
            axislong,
        )


def fld_ene_3d_old(file):
    attrs, axis, fld = osiris_open_grid_data(file)
    fld = fld[:]
    flsh = fld.shape
    ax1 = axis[0]
    ax2 = axis[1]
    ax3 = axis[2]
    x1 = np.linspace(ax1[0], ax1[1], fld.shape[2], endpoint=False)
    dx1 = x1[1] - x1[0]
    x2 = np.linspace(ax2[0], ax2[1], fld.shape[1], endpoint=False)
    dx2 = x2[1] - x2[0]
    x3 = np.linspace(ax3[0], ax3[1], fld.shape[0], endpoint=False)
    dx3 = x3[1] - x3[0]
    fld = fld**2
    enefld = np.sum(fld) * dx1 * dx2 * dx3 / 2
    print(np.max(fld))
    return enefld  # NEVER TESTED


class fld_ene_3d:
    def __init__(self, folder, x1range=None, x2range=None, x3range=None, fileout=None, if_save=False):
        self.fileout = fileout
        self.folder = folder
        self.filename, self.tags = filetags(folder)
        self.ini()
        self.idx_x1_min, self.idx_x1_max = self.range(x1range, self.x1)
        self.idx_x2_min, self.idx_x2_max = self.range(x2range, self.x2)
        self.idx_x3_min, self.idx_x3_max = self.range(x3range, self.x3)
        # print(self.idx_x1_min,self.idx_x1_max)
        # print(self.idx_x2_min,self.idx_x2_max)
        # self.tags = [self.tags[i] for i in range(0, len(self.tags), 10)]
        lentags = len(self.tags)
        self.time = np.zeros(lentags)
        self.enef = np.zeros(lentags)
        self.mainloop()
        if if_save:
            self.save()

    def ini(self):
        file_0 = self.folder + self.filename + self.tags[0] + ".h5"
        attrs, axis, fld = osiris_open_grid_data(file_0)
        self.attrs = attrs
        self.fldattrs = fld.attrs
        self.fldshape = fld.shape
        ax1 = axis[0]
        ax2 = axis[1]
        ax3 = axis[2]
        self.x1 = np.linspace(ax1[0], ax1[1], self.fldshape[2], endpoint=False)
        dx1 = self.x1[1] - self.x1[0]
        self.x1 += dx1 / 2
        self.x2 = np.linspace(ax2[0], ax2[1], self.fldshape[1], endpoint=False)
        self.x3 = np.linspace(ax3[0], ax3[1], self.fldshape[0], endpoint=False)
        dx2 = self.x2[1] - self.x2[0]
        dx3 = self.x3[1] - self.x3[0]
        self.x2 += dx2 / 2
        self.x3 += dx3 / 2
        self.dx1 = dx1
        self.dx2 = dx2
        self.dx3 = dx3

    def range(self, ran, x):
        if ran == None:
            return 0, len(x) + 1
        else:
            xmin = ran[0]
            if xmin == "":
                indmin = 0
            else:
                indmin = find_nearest(x, xmin)
            xmax = ran[1]
            if xmax == "":
                indmax = x.shape + 1
            else:
                indmax = find_nearest(x, xmax) + 1
            return indmin, indmax

    def mainloop(self):
        flag = -1
        # pbar = progressbar.ProgressBar()
        # for t in pbar(self.tags):
        for t in tqdm(self.tags):
            flag += 1
            file = self.folder + self.filename + t + ".h5"
            attrs, axis, fld = osiris_open_grid_data(file)
            self.time[flag] = attrs["TIME"]
            fld = fld[self.idx_x3_min : self.idx_x3_max, self.idx_x2_min : self.idx_x2_max, self.idx_x1_min : self.idx_x1_max]
            fld = fld**2
            enefld = np.sum(fld) * self.dx1 * self.dx2 * self.dx3 / 2
            self.enef[flag] = enefld
            print(enefld)
            # print('ok I am here')

    def save(self):
        if "/MS/" in self.folder:
            sta, end = find_string_match("/MS/", self.folder)
            folderout = self.folder[:end] + "ENE/"
        else:
            folderout = self.folder + "ENE/"
        askfolderexists_create(folderout)
        if not self.fileout:
            fileout = self.filename[:-1] + "_ene"
        else:
            fileout = self.fileout
        name = "ene_" + (self.attrs["NAME"][0]).decode("utf-8")
        dataset = name
        units = "m_ec^2"
        longname = "\epsilon_{" + (self.fldattrs["LONG_NAME"][0]).decode("utf-8") + "}"
        axisunits = "1/\omega_p"
        axisname = "t"
        axislong = "Time"
        if askexists_skip(folderout + fileout + ".h5"):
            os.remove(folderout + fileout + ".h5")
        print("Writing", fileout, "@", folderout)
        osiris_save_grid_1d(
            folderout,
            fileout,
            self.enef,
            self.time,
            name,
            dataset,
            units,
            longname,
            axisunits,
            axisname,
            axislong,
        )


# return enefld


def osiris_field_energy(file):
    time, b1, b2, b3, e1, e2, e3 = np.loadtxt(file, skiprows=2, usecols=(1, 2, 3, 4, 5, 6, 7), unpack=True)
    _, indx = np.unique(time, return_index=True)
    if not time.shape == indx.shape:
        # iter = iter[indx]
        time = time[indx]
        b1 = b1[indx]
        b2 = b2[indx]
        b3 = b3[indx]
        e1 = e1[indx]
        e2 = e2[indx]
        e3 = e3[indx]
    if "/HIST/" in file:
        sta, end = find_string_match("HIST/", file)
        askfolderexists_create(file[:sta] + "MS/")
        folderout = file[:sta] + "MS/ENE/"
    else:
        folderout = ospathup(file) + "ENE/"
    askfolderexists_create(folderout)
    filename = {
        "b1": "b1_ene",
        "b2": "b2_ene",
        "b3": "b3_ene",
        "e1": "e1_ene",
        "e2": "e2_ene",
        "e3": "e3_ene",
        "fld": "fld_ene",
    }
    comps = ["b1", "b2", "b3", "e1", "e2", "e3", "fld"]
    long_name = {
        "b1": "\epsilon_{B_1}",
        "b2": "\epsilon_{B_2}",
        "b3": "\epsilon_{B_3}",
        "e1": "\epsilon_{E_1}",
        "e2": "\epsilon_{E_2}",
        "e3": "\epsilon_{E_3}",
        "fld": "\epsilon_{E+B}",
    }
    for i in comps:
        if i == "b1":
            data = b1
        elif i == "b2":
            data = b2
        elif i == "b3":
            data = b3
        elif i == "e1":
            data = e1
        elif i == "e2":
            data = e2
        elif i == "e3":
            data = e3
        elif i == "fld":
            data = b1 + b2 + b3 + e1 + e2 + e3
        attrsdata = {"UNITS": "m_ec^2", "LONG_NAME": long_name[i]}
        axattr = {"UNITS": b"1/\\omega_p", "NAME": "t", "LONG_NAME": "Time"}
        osiris_save_grid(
            folder=folderout,
            filename=filename[i],
            data=data,
            dataset_name=i,
            data_attrs=attrsdata,
            axis1=time,
            ax1attrs=axattr,
        )


# %%
