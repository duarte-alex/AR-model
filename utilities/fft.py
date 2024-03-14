from numpy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
import numpy as np
import sys
from utilities.plot import std_plot
from utilities.round import base10_min_max_nonzero, round_to_1
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utilities.find import find_nearest


class FFT2D:
    def __init__(self, data, axis, **kwargs):
        self.datashape = data.shape
        self.data = data
        self.check_axis(axis)
        self.fft_data = fftshift(fft2(data, norm="forward"))
        self.wave_vector()
        try:
            self.dataattrs = data.attrs
            self.units = (self.dataattrs["UNITS"][0]).decode("utf-8")
            self.long_name = (self.dataattrs["LONG_NAME"][0]).decode("utf-8")
        except AttributeError:
            self.dataattrs = False
        self.plot = False
        self.check_kwargs(kwargs)

    def check_axis(self, axis):
        try:
            self.ax1 = [axis[0][0], axis[0][-1]]
            self.ax2 = [axis[1][0], axis[1][-1]]
            x1 = np.linspace(self.ax1[0], self.ax1[1], self.datashape[1], endpoint=False)
            x2 = np.linspace(self.ax2[0], self.ax2[1], self.datashape[0], endpoint=False)
            self.dx1 = x1[1] - x1[0]
            self.dx2 = x2[1] - x2[0]
            self.x1 = x1 - self.dx1
            self.x2 = x2 - self.dx2
            return
        except IndexError or TypeError:
            print("Bad definition of axis. It must be a list or array of the style axis=[[x1min,x1max],[x2min,x2max]]. Aborting...")
            sys.exit()

    def wave_vector(self):
        self.kx1 = 2 * np.pi * fftshift(fftfreq(self.datashape[1], d=self.dx1))
        self.kx2 = 2 * np.pi * fftshift(fftfreq(self.datashape[0], d=self.dx2))
        return

    def plot_fft(self, kind="abs", **kwargs):
        if not kind == "abs" and not kind == "real" and not kind == "imag":
            print("Valid values for kind are: abs, real, or imag. Aborting...")
            sys.exit()
        std_plot()
        self.fftkind = kind
        exec("self.ffp=self." + kind + "()")
        self.plot = True
        self.check_kwargs(kwargs)
        fig, ax = plt.subplots(figsize=(10, 8), tight_layout=True, nrows=1, ncols=1)
        im = ax.imshow(self.ffp, aspect="auto", origin="lower", cmap=self.cmap, norm=self.norm, vmin=self.vmin, vmax=self.vmax, extent=[self.kx1[0], self.kx1[-1], self.kx2[0], self.kx2[-1]])
        ax.set_xlabel("$k_1\ [\omega_p/c]$")
        ax.set_ylabel("$k_2\ [\omega_p/c]$")
        if not self.k1range == False:
            ax.set_xlim(self.k1range)
        if not self.k2range == False:
            ax.set_ylim(self.k2range)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, ticks=self.cbar_ticks)
        cbar.ax.set_ylabel("$|FFT(" + self.long_name + ")|\ [" + self.units + "]$")
        cbar.ax.minorticks_off()
        plt.show()

    def abs(self):
        return np.abs(self.fft_data)

    def real(self):
        return np.real(self.fft_data)

    def imag(self):
        return np.imag(self.fft_data)

    def mode_value(self, kx, ky, **kwargs):
        if kx < np.min(self.kx1) or ky < np.min(self.kx2) or kx > np.max(self.kx1) or ky > np.max(self.kx2):
            print("Value of k out of the range. Aborting...")
            sys.exit()
        ind_kx = find_nearest(self.kx1, kx)
        ind_ky = find_nearest(self.kx2, ky)
        return self.fft_data[ind_ky, ind_kx]

    def filter(self, kx, ky, exclude=False):
        fft_data_filt = np.zeros(self.datashape, dtype=np.complex64)
        kxmin = kx[0]
        kxmax = kx[-1]
        kymin = ky[0]
        kymax = ky[-1]
        id3 = ind_kx1_min_negative = find_nearest(self.kx1, -kxmax)
        id4 = ind_kx1_max_negative = find_nearest(self.kx1, -kxmin)
        id7 = ind_kx1_min_positive = find_nearest(self.kx1, kxmin)
        id8 = ind_kx1_max_positive = find_nearest(self.kx1, kxmax)
        id1 = ind_kx2_min_negative = find_nearest(self.kx2, -kymax)
        id2 = ind_kx2_max_negative = find_nearest(self.kx2, -kymin)
        id5 = ind_kx2_min_positive = find_nearest(self.kx2, kymin)
        id6 = ind_kx2_max_positive = find_nearest(self.kx2, kymax)
        fft_data_filt[id1:id2, id3:id4] = self.fft_data[id1:id2, id3:id4]
        fft_data_filt[id5:id6, id7:id8] = self.fft_data[id5:id6, id7:id8]
        if exclude == True:
            fft_data_filt[:, :] = self.fft_data[:, :] - self.fft_data_filt[:, :]
        self.fft_data[:, :] = fft_data_filt[:, :]
        return

    """
    def save_osiris(self):
        kx1,kx2=self.wave_vector()
    """

    def check_kwargs(self, kwargs):
        if not self.dataattrs:
            if "UNITS" in kwargs:
                self.units = kwargs["UNITS"]
            else:
                self.units = "a.u."
            if "LONG_NAME" in kwargs:
                self.long_name = kwargs["LONG_NAME"]
            else:
                self.long_name = "DATA"
        if self.plot:
            self.plot = False
            if "cmap" in kwargs:
                self.cmap = kwargs["cmap"]
            else:
                self.cmap = "rainbow"
            if "LogPlot" in kwargs:
                self.logplot = kwargs["LogPlot"]
            else:
                if self.fftkind == "abs":
                    self.logplot = True
                else:
                    self.logplot = False
            if "fftrange" in kwargs and self.logplot:
                r1 = kwargs["fftrange"][0]
                r2 = kwargs["fftrange"][1]
                self.norm = LogNorm(vmin=r1, vmax=r2)
                self.vmin = None
                self.vmax = None
                self.cbar_ticks = np.logspace(np.log10(r1), np.log10(r2), np.log10(r2) - np.log10(r1) + 1)
            elif "fftrange" in kwargs and not self.logplot:
                r1 = kwargs["fftrange"][0]
                r2 = kwargs["fftrange"][1]
                self.norm = None
                self.vmin = r1
                self.vmax = r2
            elif self.logplot:
                if self.fftkind == "abs":
                    r1, r2 = base10_min_max_nonzero(self.ffp)
                    self.norm = LogNorm(vmin=r1, vmax=r2)
                    self.vmin = None
                    self.vmax = None
                else:
                    print("Warning: taking absolute value of the data to make log plot.")
                    self.ffp = np.abs(self.ffp)
                    r1, r2 = base10_min_max_nonzero(self.ffp)
                    self.norm = LogNorm(vmin=r1, vmax=r2)
                    self.vmin = None
                    self.vmax = None
                self.cbar_ticks = np.logspace(np.log10(r1), np.log10(r2), np.log10(r2) - np.log10(r1) + 1)
            elif not self.logplot:
                max = np.max(np.array([np.max(self.ffp), -np.min(self.ffp)]))
                self.norm = None
                self.vmin = -max
                self.vmax = max
            self.k1range = False
            self.k2range = False
            xrlist = ["xrange", "x1range", "k1range"]
            for i in xrlist:
                try:
                    self.k1range = [kwargs[i][0], kwargs[i][1]]
                    break
                except KeyError:
                    pass
            yrlist = ["yrange", "x2range", "k2range"]
            for i in yrlist:
                try:
                    self.k2range = [kwargs[i][0], kwargs[i][1]]
                    break
                except KeyError:
                    pass


def fft_filter_2D(kx, ky, data, axis, attrs=None, plot=False, exclusive=False):
    fft_data_filt = np.zeros(data.shape, dtype=np.complex64)
    ax1 = [axis[0][0], axis[0][-1]]
    ax2 = [axis[1][0], axis[1][-1]]
    x1 = np.linspace(ax1[0], ax1[1], data.shape[1], endpoint=False)
    dx1 = x1[1] - x1[0]
    x2 = np.linspace(ax2[0], ax2[1], data.shape[0], endpoint=False)
    dx2 = x2[1] - x2[0]
    fft_data = fft2(data)
    kx1 = 2 * np.pi * fftfreq(data.shape[1], d=dx1)
    kx2 = 2 * np.pi * fftfreq(data.shape[0], d=dx2)
    kx1 = fftshift(kx1)
    kx2 = fftshift(kx2)
    fft_data = fftshift(fft_data)
    kxmin = kx[0]
    kxmax = kx[-1]
    kymin = ky[0]
    kymax = ky[-1]
    id3 = ind_kx1_min_negative = find_nearest(kx1, -kxmax)
    id4 = ind_kx1_max_negative = find_nearest(kx1, -kxmin)
    id7 = ind_kx1_min_positive = find_nearest(kx1, kxmin)
    id8 = ind_kx1_max_positive = find_nearest(kx1, kxmax)
    id1 = ind_kx2_min_negative = find_nearest(kx2, -kymax)
    id2 = ind_kx2_max_negative = find_nearest(kx2, -kymin)
    id5 = ind_kx2_min_positive = find_nearest(kx2, kymin)
    id6 = ind_kx2_max_positive = find_nearest(kx2, kymax)
    fft_data_filt[id1:id2, id3:id4] = fft_data[id1:id2, id3:id4]
    fft_data_filt[id5:id6, id7:id8] = fft_data[id5:id6, id7:id8]
    if exclusive == True:
        fft_data_filt[:, :] = fft_data[:, :] - fft_data_filt[:, :]
    fft_data_filt = ifftshift(fft_data_filt)
    data_filt = ifft2(fft_data_filt)
    data_filt = np.real(data_filt)
    if plot == True:
        std_plot()
        if attrs == None:
            str_fld = "Data"
            str_fft = "FFT"
        else:
            str_fld = "$" + (data.attrs["LONG_NAME"][0]).decode("utf-8") + "\ [" + (data.attrs["UNITS"][0]).decode("utf-8") + "]$"
            str_fft = "$|FFT(" + (data.attrs["LONG_NAME"][0]).decode("utf-8") + ")|\ [" + (data.attrs["UNITS"][0]).decode("utf-8") + "]$"
        hist, edges = np.histogram(np.abs(data[:]))
        vm = 2 * np.average(edges[1:-1], weights=hist[1:])
        vm = round_to_1(vm)
        fig, ax = plt.subplots(figsize=(15, 12), tight_layout=True, nrows=2, ncols=2)
        im = ax[0, 1].imshow(data, origin="lower", cmap="seismic", vmin=-vm, vmax=vm, extent=[ax1[0], ax1[-1], ax2[0], ax2[-1]])
        ax[0, 1].set_xlabel("$x_1\ [c/\omega_p]$")
        ax[0, 1].set_ylabel("$x_2\ [c/\omega_p]$")
        divider = make_axes_locatable(ax[0, 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, ticks=np.arange(-vm, 1.0000001 * vm, vm / 2).tolist())
        cbar.ax.set_ylabel(str_fld)
        minval, maxval = base10_min_max_nonzero(np.abs(fft_data))
        im = ax[0, 0].imshow(np.abs(fft_data), origin="lower", cmap="rainbow", norm=LogNorm(vmin=minval, vmax=maxval), extent=[kx1[0], kx1[-1], kx2[0], kx2[-1]])
        ax[0, 0].set_xlabel("$k_1\ [\omega_p/c]$")
        ax[0, 0].set_ylabel("$k_2\ [\omega_p/c]$")
        divider = make_axes_locatable(ax[0, 0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, ticks=np.logspace(np.log10(minval), np.log10(maxval), np.log10(maxval) - np.log10(minval) + 1))
        cbar.ax.set_ylabel(str_fft)
        cbar.ax.minorticks_off()
        fft_data_filt_vis = np.zeros(data.shape, dtype=np.complex64)
        fft_data_filt_vis[:] = fft_data_filt[:]
        fft_data_filt_vis[np.where(fft_data_filt_vis == 0)] = 1e-25
        fft_data_filt_vis = fftshift(fft_data_filt_vis)
        im = ax[1, 0].imshow(np.abs(fft_data_filt_vis), origin="lower", cmap="rainbow", norm=LogNorm(vmin=minval, vmax=maxval), extent=[kx1[0], kx1[-1], kx2[0], kx2[-1]])
        ax[1, 0].set_xlabel("$k_1\ [\omega_p/c]$")
        ax[1, 0].set_ylabel("$k_2\ [\omega_p/c]$")
        divider = make_axes_locatable(ax[1, 0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, ticks=np.logspace(np.log10(minval), np.log10(maxval), np.log10(maxval) - np.log10(minval) + 1))
        str = "$|FFT(" + (data.attrs["LONG_NAME"][0]).decode("utf-8") + ")|\ [" + (data.attrs["UNITS"][0]).decode("utf-8") + "]$"
        cbar.ax.set_ylabel(str)
        cbar.ax.minorticks_off()
        im = ax[1, 1].imshow(data_filt, origin="lower", cmap="seismic", vmin=-vm, vmax=vm, extent=[ax1[0], ax1[-1], ax2[0], ax2[-1]])
        ax[1, 1].set_xlabel("$x_1\ [c/\omega_p]$")
        ax[1, 1].set_ylabel("$x_2\ [c/\omega_p]$")
        divider = make_axes_locatable(ax[1, 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, ticks=np.arange(-vm, 1.0000001 * vm, vm / 2).tolist())
        str = "$" + (data.attrs["LONG_NAME"][0]).decode("utf-8") + "[" + (data.attrs["UNITS"][0]).decode("utf-8") + "]$"
        cbar.ax.set_ylabel(str)
        plt.show()
    return data_filt
