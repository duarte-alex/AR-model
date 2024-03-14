import numpy as np
from utilities.osiris.open import osiris_open_grid_data
from utilities.osiris.save import osiris_save_grid_copy_attrs_2d, osiris_save_grid_copy_attrs_3d
from utilities.find import find_nearest
from utilities.ask import askfolderexists_create
from utilities.path import ospathup
import os


class cutgrid2d:
    def __init__(self, file, x1range=None, x2range=None):
        self.attrs, self.axis, self._data_ = osiris_open_grid_data(file)
        self.ax1 = self.axis[0]
        self.ax2 = self.axis[1]
        x1 = np.linspace(self.ax1[0], self.ax1[1], self._data_.shape[-1], endpoint=False)
        dx1 = x1[1] - x1[0]
        x1 += dx1 / 2
        x2 = np.linspace(self.ax2[0], self.ax2[1], self._data_.shape[0], endpoint=False)
        dx2 = x2[1] - x2[0]
        x2 += dx2 / 2
        self.data_cut = self._data_
        if not x1range == None:
            try:
                indx1min = find_nearest(x1, x1range[0])
            except TypeError:
                indx1min = 0
            try:
                indx1max = find_nearest(x1, x1range[1])
            except TypeError:
                indx1max = x1.shape[0] - 1
            self.data_cut = self.data_cut[:, indx1min : (indx1max + 1)]
            self.x1r = np.array([x1[indx1min] - dx1 / 2, x1[indx1max] + dx1 / 2])
        else:
            self.x1r = np.array([x1[0], x1[-1]])
        if not x2range == None:
            try:
                indx2min = find_nearest(x2, x2range[0])
            except TypeError:
                indx2min = 0
            try:
                indx2max = find_nearest(x2, x2range[1])
            except TypeError:
                indx2max = x2.shape[0] - 1
            self.data_cut = self.data_cut[indx2min : (indx2max + 1), :]
            self.x2r = np.array([x2[indx2min] - dx2 / 2, x2[indx2max] + dx2 / 2])
        else:
            self.x2r = np.array([x2[0], x2[-1]])
        self.folderout = os.path.join(os.path.split(file)[0], "cut")
        self.fileout = os.path.split(file)[1]

    def save(self, folderout=None, fileout=None):
        if folderout == None:
            folderout = self.folderout
        if fileout == None:
            fileout = self.fileout
        askfolderexists_create(folderout)
        osiris_save_grid_copy_attrs_2d(folderout, fileout, self.data_cut, self._data_.attrs, self._data_.name[1:], self.x1r, self.ax1.attrs, self.x2r, self.ax2.attrs, self.attrs)

    def data(self):
        foldertemp = ospathup(self.folderout) + "/.h5temp"
        self.save(folderout=foldertemp, fileout="temp.h5")
        file = foldertemp + "/temp.h5"
        attrs, axis, data = osiris_open_grid_data(file)
        os.remove(file)
        os.rmdir(foldertemp)
        return attrs, axis, data


# def cutgrid2d(file,folderout,fileout,x1range=None,x2range=None):
# attrs,axis,data=osiris_open_grid_data(file)
# ax1=axis[0]
# ax2=axis[1]
# x1=np.linspace(ax1[0],ax1[1],data.shape[-1],endpoint=False);dx1=x1[1]-x1[0];x1+=dx1/2
# x2=np.linspace(ax2[0],ax2[1],data.shape[ 0],endpoint=False);dx2=x2[1]-x2[0];x2+=dx2/2
# data_v=data[:]
# if not x1range==None:
# indx1min=find_nearest(x1,x1range[0])
# indx1max=find_nearest(x1,x1range[1])
# data_v=data_v[:,indx1min:indx1max]
# x1r=np.array([x1[indx1min],x1[indx1max]])
# else:
# x1r=np.array([x1[0],x1[-1]])
# if not x2range==None:
# indx2min=find_nearest(x2,x2range[0])
# indx2max=find_nearest(x2,x2range[1])
# data_v=data_v[indx2min:indx2max,:]
# x2r=np.array([x2[indx2min],x2[indx2max]])
# else:
# x2r=np.array([x2[0],x2[-1]])
# osiris_save_grid_copy_attrs_2d(folderout,fileout,data_v,data.attrs,data.name[1:],x1r,ax1.attrs,x2r,ax2.attrs,attrs)

"""
def cutgrid3d(file, folderout, fileout, x1range=None, x2range=None, x3range=None):
    attrs, axis, data = osiris_open_grid_data(file)
    ax1 = axis[0]
    ax2 = axis[1]
    ax3 = axis[2]
    x1 = np.linspace(ax1[0], ax1[1], data.shape[2], endpoint=False)
    dx1 = x1[1] - x1[0]
    x1 += dx1 / 2
    x2 = np.linspace(ax2[0], ax2[1], data.shape[1], endpoint=False)
    dx2 = x2[1] - x2[0]
    x2 += dx2 / 2
    x3 = np.linspace(ax3[0], ax3[1], data.shape[0], endpoint=False)
    dx3 = x3[1] - x3[0]
    x3 += dx3 / 2
    data_v = data[:]
    if not x1range == None:
        indx1min = find_nearest(x1, x1range[0])
        indx1max = find_nearest(x1, x1range[1])
        data_v = data_v[:, :, indx1min:indx1max]
        x1r = np.array([x1[indx1min], x1[indx1max]])
    else:
        x1r = np.array([x1[0], x1[-1]])
    if not x2range == None:
        indx2min = find_nearest(x2, x2range[0])
        indx2max = find_nearest(x2, x2range[1])
        data_v = data_v[:, indx2min:indx2max, :]
        x2r = np.array([x2[indx2min], x2[indx2max]])
    else:
        x2r = np.array([x2[0], x2[-1]])
    if not x3range == None:
        indx3min = find_nearest(x3, x3range[0])
        indx3max = find_nearest(x3, x3range[1])
        data_v = data_v[indx3min:indx3max, :, :]
        x3r = np.array([x3[indx3min], x3[indx3max]])
    else:
        x3r = np.array([x3[0], x3[-1]])
    print(data_v.shape)
    osiris_save_grid_copy_attrs_3d(folderout, fileout, data_v, data.attrs, data.name[1:], x1r, ax1.attrs, x2r, ax2.attrs, x3r, ax3.attrs, attrs)
    return
"""

class cutgrid3d:
    def __init__(self, file, x1range=None, x2range=None, x3range=None):
        self.attrs, self.axis, self._data_ = osiris_open_grid_data(file)
        self.ax1 = self.axis[0]
        self.ax2 = self.axis[1]
        self.ax3 = self.axis[2]
        x1 = np.linspace(self.ax1[0], self.ax1[1], self._data_.shape[-1], endpoint=False)
        dx1 = x1[1] - x1[0]
        x1 += dx1 / 2
        x2 = np.linspace(self.ax2[0], self.ax2[1], self._data_.shape[-2], endpoint=False)
        dx2 = x2[1] - x2[0]
        x2 += dx2 / 2
        x3 = np.linspace(self.ax3[0], self.ax3[1], self._data_.shape[-3], endpoint=False)
        dx3 = x3[1] - x3[0]
        x3 += dx3 / 2
        self.data_cut = self._data_
        if not x1range == None:
            try:
                indx1min = find_nearest(x1, x1range[0])
            except TypeError:
                indx1min = 0
            try:
                indx1max = find_nearest(x1, x1range[1])
            except TypeError:
                indx1max = x1.shape[0] - 1
            self.data_cut = self.data_cut[:, :, indx1min : (indx1max + 1)]
            self.x1r = np.array([x1[indx1min] - dx1 / 2, x1[indx1max] + dx1 / 2])
        else:
            self.x1r = np.array([x1[0], x1[-1]])
        if not x2range == None:
            try:
                indx2min = find_nearest(x2, x2range[0])
            except TypeError:
                indx2min = 0
            try:
                indx2max = find_nearest(x2, x2range[1])
            except TypeError:
                indx2max = x2.shape[0] - 1
            self.data_cut = self.data_cut[:, indx2min : (indx2max + 1), :]
            self.x2r = np.array([x2[indx2min] - dx2 / 2, x2[indx2max] + dx2 / 2])
        else:
            self.x2r = np.array([x2[0], x2[-1]])
        if not x3range == None:
            try:
                indx3min = find_nearest(x3, x3range[0])
            except TypeError:
                indx3min = 0
            try:
                indx3max = find_nearest(x3, x3range[1])
            except TypeError:
                indx3max = x3.shape[0] - 1
            self.data_cut = self.data_cut[indx3min : (indx3max + 1), :, :]
            self.x3r = np.array([x3[indx3min] - dx3 / 2, x3[indx3max] + dx3 / 2])
        else:
            self.x3r = np.array([x3[0], x3[-1]])
        self.folderout = os.path.join(os.path.split(file)[0], "cut")
        self.fileout = os.path.split(file)[1]

    def save(self, folderout=None, fileout=None):
        if folderout == None:
            folderout = self.folderout
        if fileout == None:
            fileout = self.fileout
        askfolderexists_create(folderout)
        osiris_save_grid_copy_attrs_3d(
            folderout, fileout, self.data_cut, self._data_.attrs, self._data_.name[1:], self.x1r, self.ax1.attrs, self.x2r, self.ax2.attrs, self.x3r, self.ax3.attrs, self.attrs
        )

    def data(self):
        foldertemp = ospathup(self.folderout) + "/.h5temp"
        self.save(folderout=foldertemp, fileout="temp.h5")
