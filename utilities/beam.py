import numpy as np
from utilities.hist import hist1d, hist2d
from matplotlib.figure import Figure
from matplotlib.pylab import get_current_fig_manager
from utilities.round import base10_min_max_nonzero
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import sys
from utilities.osiris.open import (
    osiris_open_particle_data,
    osiris_alg_test,
    filetags,
    osiris_open_grid_data,
)

from utilities.osiris.save import osiris_save_grid_1d
from utilities.units import plasma_parameters
from utilities.ask import (
    askfolder_path,
    ask_user_input,
    askfolderexists_create,
    askfile_path,
    askexists_skip,
)
from utilities.path import ospathup, osfile, osfolderup
from utilities.find import find_string_match
import progressbar
from utilities.plot import std_plot
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk
from tkinter.font import Font
import shapely.geometry as sh


def beam_profile(
    x,
    y,
    q,
    bins=[100, 100],
    interpolation=None,
    filternorm=1,
    filterrad=4.0,
    extent=None,
    range=None,
    cmap="jet",
    xlabel=r"$x$",
    ylabel=r"$y$",
    showplot=False,
    savepath="~/Desktop/fig.svg",
):
    hist, xran, yran = hist2d(
        x,
        y,
        bins=bins,
        range=[[range[0], range[1]], [range[2], range[3]]],
        weights=q,
        normalized=True,
    )
    histx1, x1ran = hist1d(x, bins=bins[0], range=[range[0], range[1]], weights=q, density=True)
    histy1, y1ran = hist1d(y, bins=bins[1], range=[range[2], range[3]], weights=q, density=True)
    vmin, vmax = base10_min_max_nonzero(hist)
    cbarticks = 10.0 ** (np.arange(int(np.log10(vmin)), int(np.log10(vmax)) + 1, 1))

    fig = plt.figure(figsize=(18, 12))
    ax1 = fig.add_axes([0.15, 0.2, 0.7, 0.7])
    ax1.patch.set_alpha(0)
    ax2 = fig.add_axes([0.15, 0.2, 0.7, 0.15])
    ax2.patch.set_alpha(0)
    ax3 = fig.add_axes([0.15, 0.2, 0.15, 0.7])
    ax3.patch.set_alpha(0)
    ax4 = fig.add_axes([0.86, 0.2, 0.05, 0.7])
    ax3.patch.set_alpha(0)

    phist = ax1.imshow(
        hist,
        cmap=cmap,
        extent=[xran[0], xran[-1], yran[0], yran[-1]],
        aspect="auto",
        origin="lower",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        interpolation=interpolation,
        filternorm=filternorm,
        filterrad=filterrad,
    )
    ax1.set_xlabel(xlabel, fontsize=50, labelpad=0)
    ax1.set_ylabel(ylabel, fontsize=50, labelpad=0)
    ax1.tick_params(labelsize=50, which="both", direction="in", top=True, right=True)
    ax1.tick_params(which="major", length=30, width=2)
    ax1.tick_params(which="minor", length=15, width=2)
    ax1.minorticks_on()
    if not extent == None:
        ax1.set_xlim(extent[0], extent[1])
        ax1.set_ylim(extent[2], extent[3])

    px1 = ax2.plot(x1ran, histx1, lw=2.0, c=[1 / 255, 157 / 255, 224 / 255])
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(0, 1 * np.max(histx1))
    ax2.axis("off")

    pene = ax3.plot(histy1, y1ran, lw=2.0, c=[205 / 255, 29 / 255, 19 / 255])
    ax3.set_xlim(0, 1 * np.max(histy1))
    ax3.set_ylim(ax1.get_ylim())
    ax3.axis("off")

    cbar = fig.colorbar(phist, cax=ax4, ax=ax1, ticks=cbarticks)
    cbar.ax.tick_params(labelsize=50, which="both", direction="in", left=True)
    cbar.ax.tick_params(which="major", length=20, width=2)
    savepath = os.path.expanduser(savepath)
    fig.savefig(savepath, transparent=True)
    if showplot == True:
        plt.show()
    plt.close()


"""
"""


class BeamParameters:
    def __init__(self, file, *args, **kwargs):
        self.alg = osiris_alg_test(file)
        if self.alg == "3D":
            quants = ["ene", "x1", "x2", "x3", "p1", "p2", "p3", "q"]
        elif self.alg == "q3D":
            quants = ["ene", "x1", "x3", "x4", "p1", "p2", "p3", "q"]
        elif self.alg == "2DCyl":
            quants = ["ene", "x1", "x2", "x2", "p1", "p2", "p3", "q"]
        else:
            print("Invalid algorithm. Aborting...")
            sys.exit()
        self.attrs, data = osiris_open_particle_data(file, quants)
        self.ene, self.ene_attrs = self.separate_attrs(data[0])
        self.x1, self.x1_attrs = self.separate_attrs(data[1])
        self.x2, self.x2_attrs = self.separate_attrs(data[2])
        self.x3, self.x3_attrs = self.separate_attrs(data[3])
        self.p1, self.p1_attrs = self.separate_attrs(data[4])
        self.p2, self.p2_attrs = self.separate_attrs(data[5])
        self.p3, self.p3_attrs = self.separate_attrs(data[6])
        self.q, self.q_attrs = self.separate_attrs(data[7])
        if self.alg == "2DCyl":
            dim = self.ene.shape[0]
            enec = self.ene
            x1c = self.x1
            x2c = self.x2
            x3c = self.x3
            p1c = self.p1
            p2c = self.p2
            p3c = self.p3
            qc = self.q
            self.ene = np.zeros(2 * dim)
            self.x1 = np.zeros(2 * dim)
            self.x2 = np.zeros(2 * dim)
            self.x3 = np.zeros(2 * dim)
            self.p1 = np.zeros(2 * dim)
            self.p2 = np.zeros(2 * dim)
            self.p3 = np.zeros(2 * dim)
            self.q = np.zeros(2 * dim)
            self.ene[:dim] = enec[:]
            self.ene[dim:] = enec[:]
            self.x1[:dim] = x1c[:]
            self.x1[dim:] = x1c[:]
            self.x2[:dim] = x2c[:]
            self.x2[dim:] = -x2c[:]
            self.x3[:dim] = x3c[:]
            self.x3[dim:] = -x3c[:]
            self.p1[:dim] = p1c[:]
            self.p1[dim:] = p1c[:]
            self.p2[:dim] = p2c[:]
            self.p2[dim:] = -p2c[:]
            self.p3[:dim] = p3c[:]
            self.p3[dim:] = -p3c[:]
            self.q[:dim] = qc[:]
            self.q[dim:] = qc[:]
        if self.p1.all() == 0:
            self.divx = np.array(0)
            self.divy = np.array(0)
        else:
            self.divx = self.p2 / self.p1
            self.divy = self.p3 / self.p1
        self.weights = self.q
        self.check_kwargs(args, kwargs)
        if self.if_n0:
            self.no_units = False
            self.units = plasma_parameters(self.n0)
        else:
            self.no_units = True
            self.units = plasma_parameters()

    def check_kwargs(self, args, kwargs):
        self.if_n0 = False
        self.if_dx = False
        if args:
            self.n0 = np.float(args[0])
            self.if_n0 = True
        if "n0" in kwargs:
            self.n0 = np.float(kwargs["n0"])
            self.if_n0 = True
        if "dx" in kwargs:
            self.dx = kwargs["dx"]
            self.if_dx = True
        if "warning" in kwargs and kwargs["warning"] == True:
            print("Warning: n0 should be given in cm^-3.")

    def check_n0(self, n0):
        if self.if_n0 and not n0 == None:
            print("Warning. Double definition of n0. Using", n0)
            self.units = plasma_parameters(n0)
        elif self.if_n0:
            return
        elif n0 == None:
            self.no_units = True
            return
        else:
            self.units = plasma_parameters(n0)
            return

    def check_empty(self, var):
        try:
            vvv = var[:]
            return vvv
        except ValueError or RuntimeError:
            return np.array(0)

    def separate_attrs(self, quant):
        quant_val = self.check_empty(quant)
        quant_attrs = quant.attrs
        return quant_val, quant_attrs

    def average(self, quant):
        if quant.shape:
            return np.average(quant, weights=self.weights)
        elif not quant.shape:
            return 0

    def charge(self, n0=None, norm=False, unit="pC"):
        self.check_n0(n0)
        q = np.abs(self.q)
        if np.sum(q) == 0.0:
            return 0.0
        else:
            if self.if_dx:
                dx1 = self.dx[0]
                dx2 = self.dx[1]
                dx3 = self.dx[-1]
            else:
                dx = self.attrs["DX"]
                dx1 = dx[0]
                dx2 = dx[1]
                dx3 = dx[-1]
            if self.alg == "3D":
                charge = dx1 * dx2 * dx3 * np.sum(q)
            elif self.alg == "q3D":
                charge = 2 * np.pi * dx1 * dx2 * np.sum(q)
            elif self.alg == "2DCyl":
                charge = np.pi * dx1 * dx2 * np.sum(q)
        if not self.no_units and not norm:
            charge *= self.units.charge(unit=unit)
        return charge

    def energy(self, norm=False, unit="MeV"):
        mean_ene = self.average(self.ene)
        if not norm:
            mean_ene *= self.units.electron_rest_energy(unit=unit)
        return mean_ene

    def energy_spread(self):
        mean_ene = self.energy(norm=True)
        if mean_ene == 0:
            return 0
        ene_spread = np.sqrt(self.average((self.ene - mean_ene) ** 2)) / mean_ene * 100
        return ene_spread

    def size_rms_x(self, n0=None, norm=False, unit="um"):
        self.check_n0(n0)
        mean = self.average(self.x2)
        s = (self.x2 - mean) ** 2
        s = np.sqrt(self.average(s))
        if self.alg == "2DCyl":
            s /= np.sqrt(2)
        if not self.no_units and not norm:
            s *= self.units.cwp(unit=unit)
        return s

    def size_rms_y(self, n0=None, norm=False, unit="um"):
        self.check_n0(n0)
        mean = self.average(self.x3)
        s = (self.x3 - mean) ** 2
        s = np.sqrt(self.average(s))
        if self.alg == "2DCyl":
            s /= np.sqrt(2)
        if not self.no_units and not norm:
            s *= self.units.cwp(unit=unit)
        return s

    def size_rms_z(self, n0=None, norm=False, unit="fs"):
        self.check_n0(n0)
        mean = self.average(self.x1)
        s = (self.x1 - mean) ** 2
        s = np.sqrt(self.average(s))
        if not self.no_units and not norm:
            s *= self.units.wp_inv(unit=unit)
        return s

    def size_z(self, n0=None, norm=False, unit="fs"):
        self.check_n0(n0)
        s = np.max(self.x1) - np.min(self.x1)
        if not self.no_units and not norm:
            s *= self.units.wp_inv(unit=unit)
        return s

    def div_rms_x(self):
        div = self.divx
        mean = self.average(div)
        div -= mean
        div = np.sqrt(self.average(div**2))
        """
        pz2=self.average(self.p1**2)
        px2=self.average(self.p2**2)
        div=np.arctan(np.sqrt(px2/pz2))
        """
        return div

    def div_rms_y(self):
        div = self.divy
        mean = self.average(div)
        div -= mean
        div = np.sqrt(self.average(div**2))
        return div

    def emit_tr_x(self, n0=None, norm=False, unit="um"):
        self.check_n0(n0)
        size2 = self.size_rms_x(norm=True) ** 2
        dive2 = self.div_rms_x() ** 2
        size_dive_m = self.average(self.x2 * self.divx)
        size_m = self.average(self.x2)
        dive_m = self.average(self.divx)
        size_dive_m2 = (size_dive_m - size_m * dive_m) ** 2
        emit = np.sqrt(size2 * dive2 - size_dive_m2)
        """
        x2m=self.average(self.x2**2)
        xp2=self.average(self.divx**2)
        xxp=self.average(self.x2*self.divx)
        emit=np.sqrt(x2m*xp2-xxp**2)
        """
        if not self.no_units and not norm:
            emit *= self.units.cwp(unit=unit)
        return emit

    def emit_tr_y(self, n0=None, norm=False, unit="um"):
        self.check_n0(n0)
        size2 = self.size_rms_y(norm=True) ** 2
        dive2 = self.div_rms_y() ** 2
        size_dive_m = self.average(self.x3 * self.divy)
        size_m = self.average(self.x3)
        dive_m = self.average(self.divy)
        size_dive_m2 = (size_dive_m - size_m * dive_m) ** 2
        emit = np.sqrt(size2 * dive2 - size_dive_m2)
        if not self.no_units and not norm:
            emit *= self.units.cwp(unit=unit)
        return emit

    def emit_ph_n_x(self, n0=None, norm=False, unit="um"):
        self.check_n0(n0)
        size2 = self.size_rms_x(norm=True) ** 2
        pm = self.average(self.p2)
        p = self.p2 - pm
        dive2 = self.average(p**2)
        size_dive_m = self.average(self.x2 * self.p2)
        size_m = self.average(self.x2)
        dive_m = pm
        size_dive_m2 = (size_dive_m - size_m * dive_m) ** 2
        emit = np.sqrt(size2 * dive2 - size_dive_m2)
        if not self.no_units and not norm:
            emit *= self.units.cwp(unit=unit)
        return emit

    def emit_ph_n_y(self, n0=None, norm=False, unit="um"):
        self.check_n0(n0)
        size2 = self.size_rms_y(norm=True) ** 2
        pm = self.average(self.p3)
        p = self.p3 - pm
        dive2 = self.average(p**2)
        size_dive_m = self.average(self.x3 * self.p3)
        size_m = self.average(self.x3)
        dive_m = pm
        size_dive_m2 = (size_dive_m - size_m * dive_m) ** 2
        emit = np.sqrt(size2 * dive2 - size_dive_m2)
        if not self.no_units and not norm:
            emit *= self.units.cwp(unit=unit)
        return emit

    def gamma_twiss_x(self, n0=None, norm=False, emitunit="m"):
        self.check_n0(n0)
        div = self.div_rms_x()
        tr_emit = self.emit_tr_x(norm=norm, unit=emitunit)
        if tr_emit == 0:
            gamma = 0
        else:
            gamma = div**2 / tr_emit
        return gamma

    def gamma_twiss_y(self, n0=None, norm=False, emitunit="m"):
        self.check_n0(n0)
        div = self.div_rms_y()
        tr_emit = self.emit_tr_y(norm=norm, unit=emitunit)
        if tr_emit == 0:
            gamma = 0
        else:
            gamma = div**2 / tr_emit
        return gamma

    """
    def current(self,n0=None,norm=False):
        q=self.charge()
        sz=self.size_z()
        return q/sz
    """

    def z_pos(self, n0=None, norm=False, unit="um"):
        self.check_n0(n0)
        zpos = self.attrs["XMAX"][0]
        #        if self.alg == "q3D":
        time = self.attrs["TIME"][0]
        # zpos += time
        zpos = time
        if not self.no_units and not norm:
            zpos *= self.units.cwp(unit=unit)
        return zpos

    def time(self, n0=None, norm=False, unit="ps"):
        self.check_n0(n0)
        time = self.attrs["TIME"][0]
        if not self.no_units and not norm:
            time *= self.units.wp_inv(unit=unit)
        return time

    def npar(self):
        try:
            npar = self.x1.shape[0]
        except IndexError:
            npar = 0
        return npar


"""
"""


class DataBeamParameters:
    def __init__(
        self,
        n0=False,
        x1=False,
        x2=False,
        x3=False,
        p1=False,
        p2=False,
        p3=False,
        ene=False,
        q=False,
        dx1=False,
        dx2=False,
        dx3=False,
        alg="3D",
    ):
        self.alg = alg
        self.ene = ene
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.q = q
        self.dx1 = dx1
        self.dx2 = dx2
        self.dx3 = dx3
        if type(self.p1) == bool or type(self.p2) == bool:
            self.divx = False
        else:
            self.divx = self.p2 / self.p1
        if type(self.p1) == bool or type(self.p3) == bool:
            self.divy = False
        else:
            self.divy = self.p3 / self.p1
        if not n0:
            n0 = ask_user_input("Enter the plasma density in cm^-3 (eg: 1e19)\nn0 = ")
        self.units = plasma_parameters(n0)

    def average(self, quant):
        if quant.shape:
            return np.average(quant, weights=self.q)
        elif not quant.shape:
            return 0

    def check_require(self, quants):
        flag = False
        for i in quants:
            if type(quants[i]) == bool:
                if not quants[i]:
                    flag = True
                    print("Missing value of", i, "for the calculation of the", quants["param"])
        if flag:
            print("Aborting...")
            sys.exit()

    def charge(self, unit="pC"):
        if self.alg == "3D":
            require = {
                "dx1": self.dx1,
                "dx2": self.dx2,
                "dx3": self.dx3,
                "q": self.q,
                "param": "charge",
            }
            self.check_require(require)
            charge = self.dx1 * self.dx2 * self.dx3 * np.sum(self.q)
        elif self.alg == "q3D":
            require = {"dx1": self.dx1, "dx2": self.dx2, "q": self.q, "param": "charge"}
            self.check_require(require)
            charge = 2 * np.pi * self.dx1 * self.dx2 * np.sum(self.q)
        charge *= self.units.charge(unit=unit)
        return np.abs(charge)

    def energy(self, unit="MeV"):
        require = {"ene": self.ene, "q": self.q, "param": "energy"}
        self.check_require(require)
        mean_ene = self.average(self.ene)
        mean_ene *= self.units.electron_rest_energy(unit=unit)
        return mean_ene

    def energy_spread(self):
        require = {"ene": self.ene, "q": self.q, "param": "energy spread"}
        self.check_require(require)
        mean_ene = self.average(self.ene)
        ene_spread = np.sqrt(self.average((self.ene - mean_ene) ** 2)) / mean_ene * 100
        return ene_spread

    def size_rms_x(self, norm=False, unit="um"):
        require = {"x2": self.x2, "q": self.q, "param": "size rms x"}
        self.check_require(require)
        mean = self.average(self.x2)
        s = (self.x2 - mean) ** 2
        s = np.sqrt(self.average(s))
        if not norm:
            s *= self.units.cwp(unit=unit)
        return s

    def size_rms_y(self, norm=False, unit="um"):
        require = {"x3": self.x3, "q": self.q, "param": "size rms y"}
        self.check_require(require)
        mean = self.average(self.x3)
        s = (self.x3 - mean) ** 2
        s = np.sqrt(self.average(s))
        if not norm:
            s *= self.units.cwp(unit=unit)
        return s

    def size_rms_z(self, unit="fs"):
        require = {"x1": self.x1, "q": self.q, "param": "size rms z"}
        self.check_require(require)
        mean = self.average(self.x1)
        s = (self.x1 - mean) ** 2
        s = np.sqrt(self.average(s))
        s *= self.units.wp_inv(unit=unit)
        return s

    def size_z(self, unit="fs"):
        require = {"x1": self.x1, "q": self.q, "param": "size z"}
        self.check_require(require)
        s = np.max(self.x1) - np.min(self.x1)
        s *= self.units.wp_inv(unit=unit)
        return s

    def div_rms_x(self):
        require = {"divx": self.divx, "q": self.q, "param": "divergence rms x"}
        self.check_require(require)
        div = self.divx
        mean = self.average(div)
        div -= mean
        div = np.sqrt(self.average(div**2))
        return div

    def div_rms_y(self):
        require = {"divy": self.divy, "q": self.q, "param": "divergence rms y"}
        self.check_require(require)
        div = self.divy
        mean = self.average(div)
        div -= mean
        div = np.sqrt(self.average(div**2))
        return div

    def emit_tr_x(self, unit="um"):
        require = {"x2": self.x2, "divx": self.divx, "q": self.q, "param": "trace emittance x"}
        self.check_require(require)
        size2 = self.size_rms_x(norm=True) ** 2
        dive2 = self.div_rms_x() ** 2
        size_dive_m = self.average(self.x2 * self.divx)
        size_m = self.average(self.x2)
        dive_m = self.average(self.divx)
        size_dive_m2 = (size_dive_m - size_m * dive_m) ** 2
        emit = np.sqrt(size2 * dive2 - size_dive_m2)
        emit *= self.units.cwp(unit=unit)
        return emit

    def emit_tr_y(self, unit="um"):
        require = {"x3": self.x3, "divy": self.divy, "q": self.q, "param": "trace emittance y"}
        self.check_require(require)
        size2 = self.size_rms_y(norm=True) ** 2
        dive2 = self.div_rms_y() ** 2
        size_dive_m = self.average(self.x3 * self.divy)
        size_m = self.average(self.x3)
        dive_m = self.average(self.divy)
        size_dive_m2 = (size_dive_m - size_m * dive_m) ** 2
        emit = np.sqrt(size2 * dive2 - size_dive_m2)
        emit *= self.units.cwp(unit=unit)
        return emit

    def emit_ph_n_x(self, unit="um"):
        require = {
            "x2": self.x2,
            "p2": self.p2,
            "q": self.q,
            "param": "normalized phase emittance x",
        }
        self.check_require(require)
        size2 = self.size_rms_x(norm=True) ** 2
        pm = self.average(self.p2)
        p = self.p2 - pm
        dive2 = self.average(p**2)
        size_dive_m = self.average(self.x2 * self.p2)
        size_m = self.average(self.x2)
        dive_m = pm
        size_dive_m2 = (size_dive_m - size_m * dive_m) ** 2
        emit = np.sqrt(size2 * dive2 - size_dive_m2)
        emit *= self.units.cwp(unit=unit)
        return emit

    def emit_ph_n_y(self, unit="um"):
        require = {
            "x3": self.x3,
            "p3": self.p3,
            "q": self.q,
            "param": "normalized phase emittance y",
        }
        self.check_require(require)
        size2 = self.size_rms_y(norm=True) ** 2
        pm = self.average(self.p3)
        p = self.p3 - pm
        dive2 = self.average(p**2)
        size_dive_m = self.average(self.x3 * self.p3)
        size_m = self.average(self.x3)
        dive_m = pm
        size_dive_m2 = (size_dive_m - size_m * dive_m) ** 2
        emit = np.sqrt(size2 * dive2 - size_dive_m2)
        emit *= self.units.cwp(unit=unit)
        return emit

    def gamma_twiss_x(self, emitunit="m"):
        div = self.div_rms_x()
        tr_emit = self.emit_tr_x(unit=emitunit)
        if tr_emit == 0:
            gamma = 0
        else:
            gamma = div**2 / tr_emit
        return gamma

    def gamma_twiss_y(self, emitunit="m"):
        div = self.div_rms_y()
        tr_emit = self.emit_tr_y(unit=emitunit)
        if tr_emit == 0:
            gamma = 0
        else:
            gamma = div**2 / tr_emit
        return gamma


"""
"""


class BeamChar:
    def __init__(self, n0=False, folder=False, create_files=True, if_plot=True, use_old=True):
        if not folder:
            folder = askfolder_path("/Volumes/EXT/Thales/EuPRAXIA/runs", title="Choose the directory")

        if not n0:
            n0 = ask_user_input("Enter the plasma density in cm^-3 (eg: 1e19)\nn0 = ")
        try:
            self.n0 = np.float128(n0)
        except ValueError:
            self.n0 = 2.8239587232849527e19
            print("**WARNING** || n0 is not defined. Using n0=2.824e19cm^-3 such that c/wp = 1um")

        self.nametag, self.tags = filetags(folder)
        self.folder = folder
        self.lentags = len(self.tags)
        lentags = range(self.lentags)
        if "/MS/" in self.folder:
            sta, end = find_string_match("/MS/", self.folder)
            self.beamcharfolder = self.folder[:end] + osfolderup(self.folder).upper() + "CHAR"
        else:
            self.beamcharfolder = os.path.join(self.folder, "BEAMCHAR")

        self.params_create()
        self.flag = 0
        self.pbar = progressbar.ProgressBar(maxval=self.lentags)
        self.pbar.start()
        old_vals = askexists_skip(self.beamcharfolder + "/Energy.h5")
        if old_vals == True and use_old == True:
            self.use_previous()
        else:
            self.mainloop(lentags)
        self.sort_data()
        if create_files:
            self.create_files()
        if if_plot:
            self.pl()
        self.pbar.finish()

    def zeros(self):
        return np.zeros(self.lentags)

    def params_create(self):
        self.z_pos = self.zeros()
        self.time = self.zeros()
        self.charge = self.zeros()
        self.energy = self.zeros()
        self.e_spread = self.zeros()
        self.emit_tr_x = self.zeros()
        self.emit_tr_y = self.zeros()
        self.emit_ph_n_x = self.zeros()
        self.emit_ph_n_y = self.zeros()
        self.div_x = self.zeros()
        self.div_y = self.zeros()
        self.sx = self.zeros()
        self.sy = self.zeros()
        self.sz = self.zeros()
        self.gamma_x = self.zeros()
        self.gamma_y = self.zeros()
        self.npar = self.zeros()
        self.tags_ = np.full(self.lentags, -1)

    def params_update(self):
        self.z_pos[self.flag] = self.bp.z_pos()
        self.time[self.flag] = self.bp.time()
        self.charge[self.flag] = self.bp.charge()
        self.energy[self.flag] = self.bp.energy()
        self.e_spread[self.flag] = self.bp.energy_spread()
        self.emit_tr_x[self.flag] = self.bp.emit_tr_x()
        self.emit_tr_y[self.flag] = self.bp.emit_tr_y()
        self.emit_ph_n_x[self.flag] = self.bp.emit_ph_n_x()
        self.emit_ph_n_y[self.flag] = self.bp.emit_ph_n_y()
        self.div_x[self.flag] = self.bp.div_rms_x()
        self.div_y[self.flag] = self.bp.div_rms_y()
        self.sx[self.flag] = self.bp.size_rms_x()
        self.sy[self.flag] = self.bp.size_rms_y()
        self.sz[self.flag] = self.bp.size_z()
        self.gamma_x[self.flag] = self.bp.gamma_twiss_x()
        self.gamma_y[self.flag] = self.bp.gamma_twiss_y()
        self.npar[self.flag] = self.bp.npar()

    def use_previous(self):
        old_file = self.beamcharfolder + "/parameters.txt"
        data = np.loadtxt(old_file)
        for jj in range(data.shape[0]):
            self.time[self.flag] = data[jj, 0]
            self.z_pos[self.flag] = data[jj, 1]
            self.energy[self.flag] = data[jj, 2]
            self.e_spread[self.flag] = data[jj, 3]
            self.charge[self.flag] = data[jj, 4]
            self.emit_tr_x[self.flag] = data[jj, 5]
            self.emit_tr_y[self.flag] = data[jj, 6]
            self.emit_ph_n_x[self.flag] = data[jj, 7]
            self.emit_ph_n_y[self.flag] = data[jj, 8]
            self.div_x[self.flag] = data[jj, 9]
            self.div_y[self.flag] = data[jj, 10]
            self.sx[self.flag] = data[jj, 11]
            self.sy[self.flag] = data[jj, 12]
            self.sz[self.flag] = data[jj, 13]
            self.gamma_x[self.flag] = data[jj, 14]
            self.gamma_y[self.flag] = data[jj, 15]
            self.tags_[self.flag] = data[jj, 16]
            self.flag += 1
            self.pbar.update(self.flag)
        test_repeat = np.isin(np.array([np.int(i) for i in self.tags]), self.tags_)
        idx = np.where(test_repeat == False)
        self.mainloop(idx[0])

    def mainloop(self, id):
        for t in id:
            beamfile = self.folder + self.nametag + self.tags[t] + ".h5"
            self.bp = BeamParameters(beamfile, n0=self.n0)
            self.tags_[self.flag] = np.int(self.tags[t])
            self.params_update()
            self.flag += 1
            self.pbar.update(self.flag)

    def sort_data(self):
        ind = np.argsort(self.tags_)
        self.time = self.time[ind]
        self.z_pos = self.z_pos[ind]
        self.energy = self.energy[ind]
        self.e_spread = self.e_spread[ind]
        self.charge = self.charge[ind]
        self.emit_tr_x = self.emit_tr_x[ind]
        self.emit_tr_y = self.emit_tr_y[ind]
        self.emit_ph_n_x = self.emit_ph_n_x[ind]
        self.emit_ph_n_y = self.emit_ph_n_y[ind]
        self.div_x = self.div_x[ind]
        self.div_y = self.div_y[ind]
        self.sx = self.sx[ind]
        self.sy = self.sy[ind]
        self.sz = self.sz[ind]
        self.gamma_x = self.gamma_x[ind]
        self.gamma_y = self.gamma_y[ind]
        self.tags_ = self.tags_[ind]

    def create_files(self):
        beamcharfolder = self.beamcharfolder
        askfolderexists_create(beamcharfolder)
        osiris_save_grid_1d(
            beamcharfolder,
            "Energy",
            self.energy,
            self.z_pos,
            "Energy",
            "Energy",
            "MeV",
            "Energy",
            "\mu m",
            "z",
            "z",
        )
        osiris_save_grid_1d(
            beamcharfolder,
            "Energy_spread",
            self.e_spread,
            self.z_pos,
            "Energy Spread",
            "EnergySpread",
            "%",
            "Energy Spread",
            "\mu m",
            "z",
            "z",
        )
        osiris_save_grid_1d(
            beamcharfolder,
            "Charge",
            self.charge,
            self.z_pos,
            "Charge",
            "Charge",
            "pC",
            "Charge",
            "\mu m",
            "z",
            "z",
        )
        osiris_save_grid_1d(
            beamcharfolder,
            "Emittance-trace-x",
            self.emit_tr_x,
            self.z_pos,
            "Trace emittance x",
            "emit-tr-x",
            "mm mrad",
            "\epsilon_{tr,x}",
            "\mu m",
            "z",
            "z",
        )
        osiris_save_grid_1d(
            beamcharfolder,
            "Emittance-trace-y",
            self.emit_tr_y,
            self.z_pos,
            "Trace emittance y",
            "emit-tr-y",
            "mm mrad",
            "\epsilon_{tr,y}",
            "\mu m",
            "z",
            "z",
        )
        osiris_save_grid_1d(
            beamcharfolder,
            "Emittance-phase-normalized-x",
            self.emit_ph_n_x,
            self.z_pos,
            "Normalized phase emittance x",
            "emit-ph-x",
            "mm mrad",
            "\epsilon_{ph,n,x}",
            "\mu m",
            "z",
            "z",
        )
        osiris_save_grid_1d(
            beamcharfolder,
            "Emittance-phase-normalized-y",
            self.emit_ph_n_y,
            self.z_pos,
            "Normalized phase emittance y",
            "emit-ph-y",
            "mm mrad",
            "\epsilon_{ph,n,y}",
            "\mu m",
            "z",
            "z",
        )
        osiris_save_grid_1d(
            beamcharfolder,
            "Divergence-x",
            self.div_x,
            self.z_pos,
            "Divergence x",
            "divx",
            "rad",
            "Divergence x",
            "\mu m",
            "z",
            "z",
        )
        osiris_save_grid_1d(
            beamcharfolder,
            "Divergence-y",
            self.div_y,
            self.z_pos,
            "Divergence y",
            "divy",
            "rad",
            "Divergence y",
            "\mu m",
            "z",
            "z",
        )
        osiris_save_grid_1d(
            beamcharfolder,
            "Size-x",
            self.sx,
            self.z_pos,
            "Size x",
            "sizex",
            "\mu m",
            "\sigma_x",
            "\mu m",
            "z",
            "z",
        )
        osiris_save_grid_1d(
            beamcharfolder,
            "Size-y",
            self.sy,
            self.z_pos,
            "Size y",
            "sizey",
            "\mu m",
            "\sigma_y",
            "\mu m",
            "z",
            "z",
        )
        osiris_save_grid_1d(
            beamcharfolder,
            "Size-z",
            self.sz,
            self.z_pos,
            "Size z",
            "sizez",
            "fs",
            "\sigma_z",
            "\mu m",
            "z",
            "z",
        )
        osiris_save_grid_1d(
            beamcharfolder,
            "Gamma-x",
            self.gamma_x,
            self.z_pos,
            "Gamma x",
            "gammax",
            "m^{-1}",
            "\gamma_{Twiss,x}",
            "\mu m",
            "z",
            "z",
        )
        osiris_save_grid_1d(
            beamcharfolder,
            "Gamma-y",
            self.gamma_y,
            self.z_pos,
            "Gamma y",
            "gammay",
            "m^{-1}",
            "\gamma_{Twiss,y}",
            "\mu m",
            "z",
            "z",
        )
        list0 = "{:^14}".format("Time") + "{:^14}".format("z") + "{:^18}".format("Energy") + "{:^16}".format("ESpread") + "{:^12}".format("Charge") + "{:^18}".format("Emit tr x") + "{:^12}".format("Emit tr y") + "{:^18}".format("Emit ph n x") + "{:^12}".format("Emit ph n y") + "{:^18}".format("Divx") + "{:^12}".format("Divy") + "{:^18}".format("SX") + "{:^12}".format("SY") + "{:^16}".format("SZ") + "{:^12}".format("Gamma x (Tw)") + "{:^18}".format("Gamma y (Tw)") + "{:^12}".format("TAG")
        list1 = "{:^14}".format("wp^-1") + "{:^14}".format("mu m") + "{:^18}".format("MeV") + "{:^16}".format("%") + "{:^12}".format("pC") + "{:^18}".format("mm mrad") + "{:^12}".format("mm mrad") + "{:^18}".format("mm mrad") + "{:^12}".format("mm mrad") + "{:^18}".format("rad") + "{:^12}".format("rad") + "{:^18}".format("mu m") + "{:^12}".format("mu m") + "{:^16}".format("fs") + "{:^12}".format("m^-1") + "{:^18}".format("m^-1") + "{:^12}".format("NA")
        list2 = []
        for i in range(self.lentags):
            list3 = [
                self.time[i],
                self.z_pos[i],
                self.energy[i],
                self.e_spread[i],
                self.charge[i],
                self.emit_tr_x[i],
                self.emit_tr_y[i],
                self.emit_ph_n_x[i],
                self.emit_ph_n_y[i],
                self.div_x[i],
                self.div_y[i],
                self.sx[i],
                self.sy[i],
                self.sz[i],
                self.gamma_x[i],
                self.gamma_y[i],
                self.tags_[i],
            ]
            list2.append(list3)
        np.savetxt(
            beamcharfolder + "/parameters.txt",
            np.array(list2),
            fmt="".join(["  %10.4f     "] + ["%11.4f     "] + ["%10.4f     "] * 14 + ["%6i"]),
            delimiter="     ",
            header="\n".join([list0, list1]),
            footer="\n".join([list0, list1]),
        )

    def pl(self):
        std_plot()
        fig1, ax1 = plt.subplots(figsize=(25, 15), tight_layout=True, nrows=4, ncols=4, sharex=True)
        ax1[0, 0].plot(self.z_pos, self.energy, lw=2, c="k")
        ax1[0, 0].set_xlim([self.z_pos[0], self.z_pos[-1]])
        ax1[0, 0].set_ylim([0, 1.05 * max(self.energy)])
        ax1[0, 0].set_xticks(np.arange(self.z_pos[0], self.z_pos[-1], 1000))
        ax1[0, 0].set_yticks(np.arange(0, 1.05 * max(self.energy), 50))
        # xlabel='z [$\mu m$]'
        ylabel = "Energy [MeV]"
        # ax1[0,0].set_xlabel(xlabel,fontsize=25)
        ax1[0, 0].set_ylabel(ylabel, fontsize=25)

        ax1[0, 1].plot(self.z_pos, self.e_spread, lw=2, c="k")
        ax1[0, 1].set_xlim([self.z_pos[0], self.z_pos[-1]])
        ax1[0, 1].set_ylim([0, 1.15 * np.max(self.e_spread)])
        ax1[0, 1].set_xticks(np.arange(self.z_pos[0], self.z_pos[-1], 1000))
        # ax1[0,1].set_yticks(np.arange(0,1.05*max(self.energy),50))
        # xlabel='z [$\mu m$]'
        ylabel = "E spread [%]"
        # ax1[0,1].set_xlabel(xlabel,fontsize=25)
        ax1[0, 1].set_ylabel(ylabel, fontsize=25)

        ax1[0, 2].plot(self.z_pos, self.charge, lw=2, c="k")
        ax1[0, 2].set_xlim([self.z_pos[0], self.z_pos[-1]])
        ax1[0, 2].set_ylim([0, 1.05 * max(self.charge)])
        ax1[0, 2].set_xticks(np.arange(self.z_pos[0], self.z_pos[-1], 1000))
        # ax1[0,1].set_yticks(np.arange(0,1.05*max(self.energy),50))
        # xlabel='z [$\mu m$]'
        ylabel = "Charge [pC]"
        # ax1[0,2].set_xlabel(xlabel,fontsize=25)
        ax1[0, 2].set_ylabel(ylabel, fontsize=25)

        ax1[0, 3].plot(self.z_pos, self.sz, lw=2, c="k")
        ax1[0, 3].set_xlim([self.z_pos[0], self.z_pos[-1]])
        ax1[0, 3].set_ylim([0, 1.05 * max(self.sz)])
        ax1[0, 3].set_xticks(np.arange(self.z_pos[0], self.z_pos[-1], 1000))
        # ax1[0,1].set_yticks(np.arange(0,1.05*max(self.energy),50))
        # xlabel='z [$\mu m$]'
        ylabel = "Size z [fs]"
        # ax1[0,2].set_xlabel(xlabel,fontsize=25)
        ax1[0, 3].set_ylabel(ylabel, fontsize=25)

        ax1[1, 2].plot(self.z_pos, self.sx, lw=2, c="k")
        ax1[1, 2].set_xlim([self.z_pos[0], self.z_pos[-1]])
        ax1[1, 2].set_ylim([0, 1.05 * max(self.sx)])
        ax1[1, 2].set_xticks(np.arange(self.z_pos[0], self.z_pos[-1], 1000))
        # ax1[0,1].set_yticks(np.arange(0,1.05*max(self.energy),50))
        # xlabel='z [$\mu m$]'
        ylabel = "Size x [$\mu m$]"
        # ax1[0,2].set_xlabel(xlabel,fontsize=25)
        ax1[1, 2].set_ylabel(ylabel, fontsize=25)

        ax1[1, 3].plot(self.z_pos, self.sy, lw=2, c="k")
        ax1[1, 3].set_xlim([self.z_pos[0], self.z_pos[-1]])
        ax1[1, 3].set_ylim([0, 1.05 * max(self.sy)])
        ax1[1, 3].set_xticks(np.arange(self.z_pos[0], self.z_pos[-1], 1000))
        # ax1[0,1].set_yticks(np.arange(0,1.05*max(self.energy),50))
        # xlabel='z [$\mu m$]'
        ylabel = "Size y [$\mu m$]"
        # ax1[0,2].set_xlabel(xlabel,fontsize=25)
        ax1[1, 3].set_ylabel(ylabel, fontsize=25)

        ax1[1, 0].semilogy(self.z_pos, self.div_x, lw=2, c="k")
        ax1[1, 0].set_xlim([self.z_pos[0], self.z_pos[-1]])
        # ax1[1,0].set_ylim([0,1.05*max(self.div_x)])
        ax1[1, 0].set_xticks(np.arange(self.z_pos[0], self.z_pos[-1], 1000))
        # ax1[0,1].set_yticks(np.arange(0,1.05*max(self.energy),50))
        # xlabel='z [$\mu m$]'
        ylabel = "Div x [rad]"
        # ax1[0,2].set_xlabel(xlabel,fontsize=25)
        ax1[1, 0].set_ylabel(ylabel, fontsize=25)

        ax1[1, 1].semilogy(self.z_pos, self.div_y, lw=2, c="k")
        ax1[1, 1].set_xlim([self.z_pos[0], self.z_pos[-1]])
        # ax1[1,1].set_ylim([0,1.05*max(self.div_y)])
        ax1[1, 1].set_xticks(np.arange(self.z_pos[0], self.z_pos[-1], 1000))
        # ax1[0,1].set_yticks(np.arange(0,1.05*max(self.energy),50))
        # xlabel='z [$\mu m$]'
        ylabel = "Div y [rad]"
        # ax1[0,2].set_xlabel(xlabel,fontsize=25)
        ax1[1, 1].set_ylabel(ylabel, fontsize=25)

        ax1[2, 2].plot(self.z_pos, self.emit_ph_n_x, lw=2, c="k")
        ax1[2, 2].set_xlim([self.z_pos[0], self.z_pos[-1]])
        ax1[2, 2].set_ylim([0, 1.05 * max(self.emit_ph_n_x)])
        ax1[2, 2].set_xticks(np.arange(self.z_pos[0], self.z_pos[-1], 1000))
        # ax1[0,1].set_yticks(np.arange(0,1.05*max(self.energy),50))
        # xlabel='z [$\mu m$]'
        ylabel = "Emit ph x [$\mu m$]"
        # ax1[0,2].set_xlabel(xlabel,fontsize=25)
        ax1[2, 2].set_ylabel(ylabel, fontsize=25)

        ax1[2, 3].plot(self.z_pos, self.emit_ph_n_y, lw=2, c="k")
        ax1[2, 3].set_xlim([self.z_pos[0], self.z_pos[-1]])
        ax1[2, 3].set_ylim([0, 1.05 * max(self.emit_ph_n_y)])
        ax1[2, 3].set_xticks(np.arange(self.z_pos[0], self.z_pos[-1], 1000))
        # ax1[0,1].set_yticks(np.arange(0,1.05*max(self.energy),50))
        # xlabel='z [$\mu m$]'
        ylabel = "Emit ph y [$\mu m$]"
        # ax1[0,2].set_xlabel(xlabel,fontsize=25)
        ax1[2, 3].set_ylabel(ylabel, fontsize=25)

        ax1[2, 0].semilogy(self.z_pos, self.emit_tr_x, lw=2, c="k")
        ax1[2, 0].set_xlim([self.z_pos[0], self.z_pos[-1]])
        # ax1[1,0].set_ylim([0,1.05*max(self.div_x)])
        ax1[2, 0].set_xticks(np.arange(self.z_pos[0], self.z_pos[-1], 1000))
        # ax1[0,1].set_yticks(np.arange(0,1.05*max(self.energy),50))
        # xlabel='z [$\mu m$]'
        ylabel = "Emit tr x [$\mu m$]"
        # ax1[0,2].set_xlabel(xlabel,fontsize=25)
        ax1[2, 0].set_ylabel(ylabel, fontsize=25)

        ax1[2, 1].semilogy(self.z_pos, self.emit_tr_y, lw=2, c="k")
        ax1[2, 1].set_xlim([self.z_pos[0], self.z_pos[-1]])
        # ax1[1,1].set_ylim([0,1.05*max(self.div_y)])
        ax1[2, 1].set_xticks(np.arange(self.z_pos[0], self.z_pos[-1], 1000))
        # ax1[0,1].set_yticks(np.arange(0,1.05*max(self.energy),50))
        # xlabel='z [$\mu m$]'
        ylabel = "Emit tr y [$\mu m$]"
        # ax1[0,2].set_xlabel(xlabel,fontsize=25)
        ax1[2, 1].set_ylabel(ylabel, fontsize=25)

        ax1[3, 2].plot(self.z_pos, self.gamma_x, lw=2, c="k")
        ax1[3, 2].set_xlim([self.z_pos[0], self.z_pos[-1]])
        ax1[3, 2].set_ylim([0, 1.05 * max(self.gamma_x)])
        ax1[3, 2].set_xticks(np.arange(self.z_pos[0], self.z_pos[-1], 1000))
        # ax1[0,1].set_yticks(np.arange(0,1.05*max(self.energy),50))
        xlabel = "z [$\mu m$]"
        ylabel = "$\gamma$ Tw x [$m^{-1}$]"
        ax1[3, 2].set_xlabel(xlabel, fontsize=25)
        ax1[3, 2].set_ylabel(ylabel, fontsize=25)

        ax1[3, 3].plot(self.z_pos, self.gamma_y, lw=2, c="k")
        ax1[3, 3].set_xlim([self.z_pos[0], self.z_pos[-1]])
        ax1[3, 3].set_ylim([0, 1.05 * max(self.gamma_y)])
        ax1[3, 3].set_xticks(np.arange(self.z_pos[0], self.z_pos[-1], 1000))
        xlabel = "z [$\mu m$]"
        ylabel = "$\gamma$ Tw y [$m^{-1}$]"
        ax1[3, 3].set_xlabel(xlabel, fontsize=25)
        ax1[3, 3].set_ylabel(ylabel, fontsize=25)

        ax1[3, 0].semilogy(self.z_pos, self.gamma_x, lw=2, c="k")
        ax1[3, 0].set_xlim([self.z_pos[0], self.z_pos[-1]])
        # ax1[1,0].set_ylim([0,1.05*max(self.div_x)])
        ax1[3, 0].set_xticks(np.arange(self.z_pos[0], self.z_pos[-1], 1000))
        # ax1[0,1].set_yticks(np.arange(0,1.05*max(self.energy),50))
        xlabel = "z [$\mu m$]"
        ylabel = "$\gamma$ Tw x [$m^{-1}$]"
        ax1[3, 0].set_xlabel(xlabel, fontsize=25)
        ax1[3, 0].set_ylabel(ylabel, fontsize=25)

        ax1[3, 1].semilogy(self.z_pos, self.gamma_y, lw=2, c="k")
        ax1[3, 1].set_xlim([self.z_pos[0], self.z_pos[-1]])
        # ax1[1,1].set_ylim([0,1.05*max(self.div_y)])
        ax1[3, 1].set_xticks(np.arange(self.z_pos[0], self.z_pos[-1], 1000))
        # ax1[0,1].set_yticks(np.arange(0,1.05*max(self.energy),50))
        xlabel = "z [$\mu m$]"
        ylabel = "$\gamma$ Tw y [$m^{-1}$]"
        ax1[3, 1].set_xlabel(xlabel, fontsize=25)
        ax1[3, 1].set_ylabel(ylabel, fontsize=25)

        # tm = get_current_fig_manager()
        # tm.window.wm_geometry("+0+0")
        plt.show()


"""
"""
# params = {"TAG": "000000", "FLD": 2, ""}


class SelectBeamTags:
    def __init__(
        self,
        folder=False,
        initdir=False,
        plot=True,
        fracplot=False,
        fieldsel=True,
        enesel=True,
        radiussel=True,
        longsel=True,
        halorem=False,
        params={},
    ):
        # params = {"TAG": "000000", "FLD": 2, "ENE": 10.0, "RAD": 0.2, "X1MIN": 0.0, "X1MAX": 0.0}
        if not folder:
            if not initdir:
                initdir = "/Users/thales/"
            folder = askfolder_path(initdir, title="Choose the directory")
        try:
            self.nametag, self.tags = filetags(folder)
        except FileNotFoundError:
            folder = askfolder_path(folder, title="Choose the directory")
        self.folder = folder
        self.plot = plot
        if type(fracplot) == int:
            self.interval = fracplot
        else:
            self.interval = 1

        try:
            self.sel_tag = params["TAG"]
            self.filetag = params["TAG"]
        except KeyError:
            self.sel_tag = ""
            self.select_initial_tag()

        self.file = folder + self.nametag + self.filetag + ".h5"
        self.alg = osiris_alg_test(self.file)
        print, self.alg
        if self.alg not in ["q3D", "3D", "2DCyl"]:
            print("Invalid algorithm. Aborting...")
            sys.exit()

        self.open_raw_data()

        try:
            self.sel_fld = params["FLD"]
            bkplot = self.plot
            self.plot = False
            self.e1_based_cutoff(opt=self.sel_fld)
            self.plot = bkplot
        except KeyError:
            self.sel_fld = False
            if fieldsel:
                self.e1_based_cutoff()

        try:
            self.sel_ene = params["ENE"]
            bkplot = self.plot
            self.plot = False
            self.ene_based_cutoff(ene_min=self.sel_ene)
            self.plot = bkplot
        except KeyError:
            self.sel_ene = False
            if enesel:
                self.ene_based_cutoff()

        try:
            self.sel_rad = params["RAD"]
            bkplot = self.plot
            self.plot = False
            self.radius_based_cutoff(ans=self.sel_rad)
            self.plot = bkplot
        except KeyError:
            self.sel_rad = False
            if radiussel:
                self.radius_based_cutoff()

        try:
            self.sel_x1min = params["X1MIN"]
            self.sel_x1max = params["X1MAX"]
            bkplot = self.plot
            self.plot = False
            self.longitudinal_position_based_cutoff(x1min=self.sel_x1min, x1max=self.sel_x1max)
            self.plot = bkplot
        except KeyError:
            self.sel_x1min = False
            self.sel_x1max = False
            if longsel:
                self.longitudinal_position_based_cutoff()

        if halorem:
            print("Select several points in the plot to define a region where to cut the halo. When you are finished, press enter.")
            self.halo_removal()

        print("Final number of beam particles = ", len(self.x1))
        ss = np.lexsort((self.tag[:, 0], self.tag[:, 1]))
        self.tagbeam = self.tag[ss]
        if self.alg == "3D":
            dx1 = (self.xmax[0] - self.xmin[0]) / self.nx[0]
            dx2 = (self.xmax[1] - self.xmin[1]) / self.nx[1]
            dx3 = (self.xmax[2] - self.xmin[2]) / self.nx[2]
            self.dx = np.array([dx1, dx2, dx3])
        elif self.alg == "q3D" or self.alg == "2DCyl":
            dx1 = (self.xmax[0] - self.xmin[0]) / self.nx[0]
            dx2 = (self.xmax[1] - self.xmin[1]) / self.nx[1]
            self.dx = np.array([dx1, dx2])

    def select_initial_tag(self):
        filetag = ask_user_input(Message="Select the RAW file you wish to use to select the beam particles\nDefault: last time. Valid values from " + str(int(self.tags[0])) + " to " + str(int(self.tags[-1])) + "\nFile tag = ")
        if filetag == "":
            filetag = str(int(self.tags[-1]))
        filetag = filetag.zfill(6)
        self.filetag = filetag
        self.sel_tag = filetag
        return

    def open_raw_data(self):
        if self.alg == "3D":
            self.quants = ["ene", "x1", "x2", "p1", "p2", "p3", "q", "tag", "x3"]
            attrs, data = osiris_open_particle_data(self.file, self.quants)
            self.x3 = data[8][:]
        elif self.alg == "q3D":
            self.quants = ["ene", "x1", "x2", "p1", "p2", "p3", "q", "tag", "x3", "x4"]
            attrs, data = osiris_open_particle_data(self.file, self.quants)
            self.x3 = data[8][:]
            self.x4 = data[9][:]
        elif self.alg == "2DCyl":
            self.quants = ["ene", "x1", "x2", "p1", "p2", "p3", "q", "tag"]
            attrs, data = osiris_open_particle_data(self.file, self.quants)
        self.ene = data[0][:]
        self.x1 = data[1][:]
        self.x2 = data[2][:]
        self.p1 = data[3][:]
        self.p2 = data[4][:]
        self.p3 = data[5][:]
        self.q = data[6][:]
        self.tag = data[7][:]
        self.nx = attrs["NX"]
        self.xmin = attrs["XMIN"]
        self.xmax = attrs["XMAX"]

    def filter_particles(self, tg):
        lenold = len(self.x1)
        tg = tg[0]
        if not len(tg) == lenold:
            if self.alg == "q3D":
                self.x4 = self.x4[tg]
                self.x3 = self.x3[tg]
            elif self.alg == "3D":
                self.x3 = self.x3[tg]
            self.ene = self.ene[tg]
            self.x1 = self.x1[tg]
            self.x2 = self.x2[tg]
            self.p1 = self.p1[tg]
            self.p2 = self.p2[tg]
            self.p3 = self.p3[tg]
            self.q = self.q[tg]
            self.tag = self.tag[tg]
        lennew = len(self.x1)
        print(lennew, "of", lenold, "particles remain.")

    def e1_based_cutoff(self, opt=False):
        fldfile = self.select_e1_folder()
        x1, e1l = self.e1_lineout_data(fldfile)
        self.e1_select_cross(x1, e1l, opt)
        return

    def select_e1_folder(self):
        if "/MS/" in self.folder:
            sta, end = find_string_match("/MS/", self.folder)
            fldfolder = self.folder[:end] + "FLD/"
        else:
            fldfolder = askfolder_path(self.folder, title="Choose the FLD directory")
        if self.alg == "3D":
            fldfolder += "e1-slice/e1-slice-x3-01-" + self.filetag + ".h5"
        elif self.alg == "2DCyl":
            fldfolder += "e1/e1-" + self.filetag + ".h5"
        elif self.alg == "q3D":
            fldfolder += "MODE-0-RE/e1_cyl_m/e1_cyl_m-0-re-" + self.filetag + ".h5"
        return fldfolder

    def e1_lineout_data(self, fldfile):
        attrs, axis, e1 = osiris_open_grid_data(fldfile)
        e1sh = e1.shape
        if self.alg == "3D":
            x2c = int(e1sh[0] / 2 - 1)
        elif self.alg == "q3D" or self.alg == "2DCyl":
            x2c = 0
        e1l = e1[x2c, :]
        ax1 = axis[0]
        x1 = np.linspace(ax1[0], ax1[1], num=e1sh[1], endpoint=False)
        dx1 = x1[1] - x1[0]
        x1 += dx1 / 2
        return x1, e1l

    def e1_select_cross(self, x1, e1l, opt):
        for i in range(e1l.shape[0]):
            if abs(e1l[e1l.shape[0] - 1 - i]) > 0.01:
                break
        x1m = e1l.shape[0] - 1 - i
        x1 = x1[0:x1m]
        e1l = e1l[0:x1m]
        cross = 0
        xcross = []
        for i in range(x1m - 1):
            e1lnew = e1l[x1m - i - 1]
            e1lold = e1l[x1m - i - 2]
            if e1lold > 0.0 and e1lnew < 0.0:
                xold = x1[x1m - i - 2]
                cross += 1
                xcross.append(xold)
        print("The code detected " + str(cross) + " time(s) that the field e1 crossed x1=0 with negative derivative.")
        print("Please select the option which has the appropriate value. Most of the times will be option 1.")
        cross += 1
        xcross.append(x1[0])
        for i in range(cross):
            print(i + 1, xcross[i])
        maxe1 = max(abs(e1l))
        #
        if self.plot:
            std_plot(keys=["xtick.minor.visible"], values=[False])
            fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
            ax.plot(x1, e1l, c="k", lw=3)
            ax.set_xticks(xcross)
            ax.xaxis.grid(True, which="major", color="r")
            ax.yaxis.grid(True, which="major", color="r")
            ax.set_xlim([x1[0], x1[-1]])
            ax.set_ylim([-1.05 * maxe1, 1.05 * maxe1])
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
            xlabel = "$x_1 [c\omega_p^{-1}]$"
            ylabel = "$E_1 [m_ec\omega_pe^{-1}]$"
            ax.set_xlabel(xlabel, fontsize=20)
            ax.set_ylabel(ylabel, fontsize=20)
            plt.draw()
            plt.pause(0.01)
        if not opt:
            opt = ask_user_input(Message="Select the option: ")
        if self.plot:
            plt.close(fig)
        if opt == "":
            opt = 0
        else:
            opt = int(opt)
            opt -= 1
            if opt < 0:
                print("Invalid option. Aborting")
                sys.exit()
        x1min = xcross[opt]
        self.sel_fld = opt + 1
        tg = np.where(self.x1 > x1min)
        self.filter_particles(tg)

    def ene_based_cutoff(self, ene_min=False):
        if self.plot:
            std_plot()
            fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
            ax.scatter(self.x1[:: self.interval], self.ene[:: self.interval], s=0.5, c="k")
            ax.set_yticks(np.linspace(0.95 * np.min(self.ene), 1.05 * np.max(self.ene), num=5))
            ax.xaxis.grid(True, which="major", color="r")
            ax.yaxis.grid(True, which="major", color="r")
            ax.set_xlim([min(self.x1), max(self.x1)])
            ax.set_ylim([min(self.ene), max(self.ene)])
            xlabel = "$x_1 [c\omega_p^{-1}]$"
            ylabel = "$Ene [m_ec^2]$"
            ax.set_xlabel(xlabel, fontsize=20)
            ax.set_ylabel(ylabel, fontsize=20)
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
            plt.draw()
            plt.pause(0.01)
        if not ene_min:
            ene_min = ask_user_input(Message="Select the minimum energy to be considered (default is 10): ")
        if self.plot:
            plt.close(fig)
        if ene_min == "":
            ene_min = 10.0
        ene_min = float(ene_min)
        self.sel_ene = ene_min
        tg = np.where(self.ene > ene_min)
        self.filter_particles(tg)

    def radius_based_cutoff(self, ans=False):
        if self.alg == "3D":
            meanx3 = np.mean(self.x3)
            stdx3 = np.std(self.x3)
            meanx2 = np.mean(self.x2)
            stdx2 = np.std(self.x2)
        elif self.alg == "2DCyl" or self.alg == "q3D":
            meanx2 = 0
            stdx2 = np.sqrt(np.mean(self.x2**2))

        sfac = 16
        flag = True
        if not ans:
            while flag:
                if self.plot:
                    if self.alg == "3D":
                        fig, ax = plt.subplots(figsize=(10, 6), ncols=2, nrows=2, sharex=True)  # , constrained_layout=True)
                        ax[0, 0].scatter(self.x1[:: self.interval], self.x2[:: self.interval], s=0.5, c="k")
                        ax[0, 0].set_xlim([np.min(self.x1), np.max(self.x1)])
                        ax[0, 0].set_ylim([np.min(self.x2), np.max(self.x2)])
                        xlabel = "$x_1 [c\omega_p^{-1}]$"
                        ylabel = "$x_2 [c\omega_p^{-1}]$"
                        # ax[0,0].set_xlabel(xlabel,fontsize=20)
                        ax[0, 0].set_ylabel(ylabel, fontsize=20)
                        ax[0, 1].scatter(self.x1[:: self.interval], self.x3[:: self.interval], s=0.5, c="k")
                        ax[0, 1].set_xlim([np.min(self.x1), np.max(self.x1)])
                        ax[0, 1].set_ylim([np.min(self.x3), np.max(self.x3)])
                        xlabel = "$x_1 [c\omega_p^{-1}]$"
                        ylabel = "$x_3 [c\omega_p^{-1}]$"
                        # ax[0,1].set_xlabel(xlabel,fontsize=20)
                        ax[0, 1].set_ylabel(ylabel, fontsize=20)
                        ax[1, 0].scatter(self.x1[:: self.interval], self.x2[:: self.interval], s=0.5, c="k")
                        ax[1, 0].set_xlim([np.min(self.x1), np.max(self.x1)])
                        ax[1, 0].set_ylim([meanx2 - sfac * stdx2, meanx2 + sfac * stdx2])
                        xlabel = "$x_1 [c\omega_p^{-1}]$"
                        ylabel = "$x_2 [c\omega_p^{-1}]$"
                        ax[1, 0].set_xlabel(xlabel, fontsize=20)
                        ax[1, 0].set_ylabel(ylabel, fontsize=20)
                        ax[1, 1].scatter(self.x1[:: self.interval], self.x3[:: self.interval], s=0.5, c="k")
                        ax[1, 1].set_xlim([np.min(self.x1), np.max(self.x1)])
                        ax[1, 1].set_ylim([meanx3 - sfac * stdx3, meanx3 + sfac * stdx3])
                        xlabel = "$x_1 [c\omega_p^{-1}]$"
                        ylabel = "$x_3 [c\omega_p^{-1}]$"
                        ax[1, 1].set_xlabel(xlabel, fontsize=20)
                        ax[1, 1].set_ylabel(ylabel, fontsize=20)
                        plt.draw()
                        plt.pause(0.01)
                    elif self.alg == "2DCyl" or self.alg == "q3D":
                        fig, ax = plt.subplots(figsize=(10, 6), ncols=1, nrows=2, sharex=True)  # , constrained_layout=True)
                        ax[0].scatter(self.x1[:: self.interval], self.x2[:: self.interval], s=0.5, c="k")
                        ax[0].set_xlim([np.min(self.x1), np.max(self.x1)])
                        ax[0].set_ylim([0, np.max(self.x2)])
                        xlabel = "$x_1 [c\omega_p^{-1}]$"
                        ylabel = "$x_2 [c\omega_p^{-1}]$"
                        # ax[0,0].set_xlabel(xlabel,fontsize=20)
                        ax[0].set_ylabel(ylabel, fontsize=20)
                        ax[1].scatter(self.x1[:: self.interval], self.x2[:: self.interval], s=0.5, c="k")
                        ax[1].set_xlim([np.min(self.x1), np.max(self.x1)])
                        ax[1].set_ylim([0, sfac * stdx2])
                        xlabel = "$x_1 [c\omega_p^{-1}]$"
                        ylabel = "$x_2 [c\omega_p^{-1}]$"
                        ax[1].set_xlabel(xlabel, fontsize=20)
                        ax[1].set_ylabel(ylabel, fontsize=20)
                        plt.draw()
                        plt.pause(0.01)
                ans = ask_user_input(Message="Enter a new factor to be considered. Use fac > 1 if you want a bigger area to be considered and 0 < fac < 1 if want a smaller area.\nDefault answer [Not necessary]: ")
                try:
                    newfac = float(ans)
                    sfac *= newfac
                    if self.plot:
                        plt.close(fig)
                except ValueError:
                    if self.alg == "3D":
                        x2min = meanx2 - sfac * stdx2
                        x2max = meanx2 + sfac * stdx2
                        x3min = meanx3 - sfac * stdx3
                        x3max = meanx3 + sfac * stdx3
                    elif self.alg == "2DCyl" or self.alg == "q3D":
                        x2min = 0
                        x2max = sfac * stdx2
                    break
        else:
            if self.alg == "3D":
                sfac *= ans
                x2min = meanx2 - sfac * stdx2
                x2max = meanx2 + sfac * stdx2
                x3min = meanx3 - sfac * stdx3
                x3max = meanx3 + sfac * stdx3
            elif self.alg == "2DCyl" or self.alg == "q3D":
                sfac *= ans
                x2min = 0
                x2max = sfac * stdx2
        if self.plot:
            plt.close(fig)
        self.sel_rad = sfac / 16
        if self.alg == "3D":
            tg = np.where((self.x2 > x2min) & (self.x2 < x2max) & (self.x3 > x3min) & (self.x3 < x3max))
        elif self.alg == "2DCyl" or self.alg == "q3D":
            tg = np.where((self.x2 > x2min) & (self.x2 < x2max))
        self.filter_particles(tg)

    def halo_removal(self):
        if self.alg == "3D":
            plt.scatter(self.x2, self.x3, s=0.01, marker="o")
            plt.draw()
            plt.waitforbuttonpress()
            pts = np.array(plt.ginput(-1, timeout=-1))
            plt.close()
            polygon = sh.polygon.Polygon(pts)
            list = []
            for i in range((self.x2).shape[0]):
                list.append((polygon.contains(sh.Point(self.x2[i], self.x3[i]))))
            tg = np.where(np.array(list) == True)
            self.filter_particles(tg)
        else:
            return

    def longitudinal_position_based_cutoff(self, x1min=False, x1max=False):
        x1min_ran = np.min(self.x1)
        x1max_ran = np.max(self.x1)
        x1range = np.linspace(x1min_ran, x1max_ran, num=3)
        if self.plot:
            if self.alg == "3D":
                fig, ax = plt.subplots(figsize=(15, 8), ncols=2, nrows=2, constrained_layout=True)
                ax[0, 0].scatter(self.x1[:: self.interval], self.x2[:: self.interval], s=0.5, c="k")
                ax[0, 0].set_xticks(x1range)
                ax[0, 0].xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
                ax[0, 0].set_xlim([np.min(self.x1), np.max(self.x1)])
                ax[0, 0].set_ylim([np.min(self.x2), np.max(self.x2)])
                xlabel = "$x_1 [c\omega_p^{-1}]$"
                ylabel = "$x_2 [c\omega_p^{-1}]$"
                ax[0, 0].set_xlabel(xlabel, fontsize=20)
                ax[0, 0].set_ylabel(ylabel, fontsize=20)
                ax[0, 1].scatter(self.x1[:: self.interval], self.x3[:: self.interval], s=0.5, c="k")
                ax[0, 1].set_xlim([np.min(self.x1), np.max(self.x1)])
                ax[0, 1].set_ylim([np.min(self.x3), np.max(self.x3)])
                ax[0, 1].set_xticks(x1range)
                ax[0, 1].xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
                xlabel = "$x_1 [c\omega_p^{-1}]$"
                ylabel = "$x_3 [c\omega_p^{-1}]$"
                ax[0, 1].set_xlabel(xlabel, fontsize=20)
                ax[0, 1].set_ylabel(ylabel, fontsize=20)
                ax[1, 0].axis("off")
                ax[1, 1].axis("off")
                plt.draw()
                plt.pause(0.01)
            elif self.alg == "q3D" or self.alg == "2DCyl":
                fig, ax = plt.subplots(figsize=(15, 8), ncols=1, nrows=2, constrained_layout=True)
                ax[0].scatter(self.x1[:: self.interval], self.x2[:: self.interval], s=0.5, c="k")
                ax[0].set_xticks(x1range)
                ax[0].xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
                ax[0].set_xlim([np.min(self.x1), np.max(self.x1)])
                ax[0].set_ylim([0, np.max(self.x2)])
                xlabel = "$x_1 [c\omega_p^{-1}]$"
                ylabel = "$x_2 [c\omega_p^{-1}]$"
                ax[0].set_xlabel(xlabel, fontsize=20)
                ax[0].set_ylabel(ylabel, fontsize=20)
                ax[1].axis("off")
                plt.draw()
                plt.pause(0.01)
        """
        """
        x1mm = np.min(self.x1)
        x1mp = np.max(self.x1)
        flag = True
        flagmin = True
        flagmax = True
        if not x1min or not x1max:
            while flag:
                x1min = ask_user_input(Message="Enter the new lowest value of x1.\nDefault: x1min = [LAST VALUE USED]\nx1min = ")
                try:
                    x1min = float(x1min)
                    x1mm = x1min
                    flagmin = True
                except ValueError:
                    x1min = x1mm
                    flagmin = False
                x1max = ask_user_input(Message="Enter the new highest value of x1.\nDefault: x1max = [LAST VALUE USED]\nx1max = ")
                try:
                    x1max = float(x1max)
                    x1mp = x1max
                    flagmax = True
                except ValueError:
                    x1max = x1mp
                    flagmax = False
                if not flagmax and not flagmin:
                    self.sel_x1min = x1min
                    self.sel_x1max = x1max
                    tg = np.where((self.x1 > x1min) & (self.x1 < x1max))
                    self.filter_particles(tg)
                    break
                else:
                    if self.plot:
                        if self.alg == "3D":
                            ax[1, 0].cla()
                            ax[1, 1].cla()
                            x1range = np.linspace(x1min, x1max, num=3)
                            ax[1, 0].scatter(self.x1[:: self.interval], self.x2[:: self.interval], s=0.5, c="k")
                            ax[1, 0].set_xticks(x1range)
                            ax[1, 0].xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
                            ax[1, 0].set_xlim([x1min, x1max])
                            ax[1, 0].set_ylim([np.min(self.x2), np.max(self.x2)])
                            xlabel = "$x_1 [c\omega_p^{-1}]$"
                            ylabel = "$x_2 [c\omega_p^{-1}]$"
                            ax[1, 0].set_xlabel(xlabel, fontsize=20)
                            ax[1, 0].set_ylabel(ylabel, fontsize=20)
                            ax[1, 1].scatter(self.x1[:: self.interval], self.x3[:: self.interval], s=0.5, c="k")
                            ax[1, 1].set_xticks(x1range)
                            ax[1, 1].set_xlim([x1min, x1max])
                            ax[1, 1].set_ylim([np.min(self.x3), np.max(self.x3)])
                            ax[1, 1].xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
                            xlabel = "$x_1 [c\omega_p^{-1}]$"
                            ylabel = "$x_3 [c\omega_p^{-1}]$"
                            ax[1, 1].set_xlabel(xlabel, fontsize=20)
                            ax[1, 1].set_ylabel(ylabel, fontsize=20)
                            plt.draw()
                            plt.pause(0.01)
                        elif self.alg == "q3D" or self.alg == "2DCyl":
                            ax[1].cla()
                            x1range = np.linspace(x1min, x1max, num=3)
                            ax[1].scatter(self.x1[:: self.interval], self.x2[:: self.interval], s=0.5, c="k")
                            ax[1].set_xticks(x1range)
                            ax[1].xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
                            ax[1].set_xlim([x1min, x1max])
                            ax[1].set_ylim([0, np.max(self.x2)])
                            xlabel = "$x_1 [c\omega_p^{-1}]$"
                            ylabel = "$x_2 [c\omega_p^{-1}]$"
                            ax[1].set_xlabel(xlabel, fontsize=20)
                            ax[1].set_ylabel(ylabel, fontsize=20)
                            plt.draw()
                            plt.pause(0.01)
        else:
            self.sel_x1min = x1min
            self.sel_x1max = x1max
            tg = np.where((self.x1 > x1min) & (self.x1 < x1max))
            self.filter_particles(tg)
        if self.plot:
            plt.close(fig="all")
        # plt.destroy()


"""
"""


class SelectBeamTagsFollow:
    def __init__(self, folder=False, n0=False):
        self.n0 = n0
        if not folder:
            folder = askfolder_path("/Volumes/EXT/Thales/EuPRAXIA/runs", title="Choose the directory")
        self.nametag, self.tags = filetags(folder)
        self.folder = folder
        self.select_initial_tag()
        self.file = folder + self.nametag + self.filetag + ".h5"
        self.alg = osiris_alg_test(self.file)
        if self.alg not in ["q3D", "3D"]:
            print("Invalid algorithm. Aborting...")
            sys.exit()
        self.open_raw_data()
        std_plot()
        self.radius_based_cutoff()
        self.longitudinal_position_based_cutoff()
        print("Final number of beam particles = ", len(self.x1))
        ss = np.lexsort((self.tag[:, 0], self.tag[:, 1]))
        self.tagbeam = self.tag[ss]

    def select_initial_tag(self):
        filetag = ask_user_input(Message="Select the RAW file you wish to use to select the beam particles\nDefault: last time. Valid values from " + str(int(self.tags[0])) + " to " + str(int(self.tags[-1])) + "\nFile tag = ")
        if filetag == "":
            filetag = str(int(self.tags[-1]))
        filetag = filetag.zfill(6)
        self.filetag = filetag
        return

    def open_raw_data(self):
        if self.alg == "3D":
            self.quants = ["ene", "x1", "x2", "x3", "p1", "p2", "p3", "q", "tag"]
            attrs, data = osiris_open_particle_data(self.file, self.quants)
        elif self.alg == "q3D":
            self.quants = ["ene", "x1", "x2", "x3", "p1", "p2", "p3", "q", "tag", "x4"]
            attrs, data = osiris_open_particle_data(self.file, self.quants)
            self.x4 = data[9][:]
        self.ene = data[0][:]
        self.x1 = data[1][:]
        self.x2 = data[2][:]
        self.x3 = data[3][:]
        self.p1 = data[4][:]
        self.p2 = data[5][:]
        self.p3 = data[6][:]
        self.q = data[7][:]
        self.tag = data[8][:]
        self.dx = attrs["DX"]
        self.dx1 = self.dx[0]
        self.dx2 = self.dx[1]
        self.dx3 = self.dx[2]

    def filter_particles(self, tg):
        lenold = len(self.x1)
        tg = tg[0]
        if self.alg == "q3D":
            self.x4 = self.x4[tg]
        self.ene = self.ene[tg]
        self.x1 = self.x1[tg]
        self.x2 = self.x2[tg]
        self.x3 = self.x3[tg]
        self.p1 = self.p1[tg]
        self.p2 = self.p2[tg]
        self.p3 = self.p3[tg]
        self.q = self.q[tg]
        self.tag = self.tag[tg]
        lennew = len(self.x1)
        print(lennew, "of", lenold, "particles remain.")

    def radius_based_cutoff(self):
        meanx2 = np.mean(self.x2)
        meanx3 = np.mean(self.x3)
        stdx2 = np.std(self.x2)
        stdx3 = np.std(self.x3)
        sfac = 16
        flag = True
        while flag:
            fig, ax = plt.subplots(figsize=(10, 6), ncols=2, nrows=2, sharex=True, constrained_layout=True)
            ax[0, 0].scatter(self.x1, self.x2, s=0.5, c="k")
            ax[0, 0].set_xlim([np.min(self.x1), np.max(self.x1)])
            ax[0, 0].set_ylim([np.min(self.x2), np.max(self.x2)])
            xlabel = "$x_1 [c\omega_p^{-1}]$"
            ylabel = "$x_2 [c\omega_p^{-1}]$"
            # ax[0,0].set_xlabel(xlabel,fontsize=20)
            ax[0, 0].set_ylabel(ylabel, fontsize=20)
            ax[0, 1].scatter(self.x1, self.x3, s=0.5, c="k")
            ax[0, 1].set_xlim([np.min(self.x1), np.max(self.x1)])
            ax[0, 1].set_ylim([np.min(self.x3), np.max(self.x3)])
            xlabel = "$x_1 [c\omega_p^{-1}]$"
            ylabel = "$x_3 [c\omega_p^{-1}]$"
            # ax[0,1].set_xlabel(xlabel,fontsize=20)
            ax[0, 1].set_ylabel(ylabel, fontsize=20)
            ax[1, 0].scatter(self.x1, self.x2, s=0.5, c="k")
            ax[1, 0].set_xlim([np.min(self.x1), np.max(self.x1)])
            ax[1, 0].set_ylim([meanx2 - sfac * stdx2, meanx2 + sfac * stdx2])
            xlabel = "$x_1 [c\omega_p^{-1}]$"
            ylabel = "$x_2 [c\omega_p^{-1}]$"
            ax[1, 0].set_xlabel(xlabel, fontsize=20)
            ax[1, 0].set_ylabel(ylabel, fontsize=20)
            ax[1, 1].scatter(self.x1, self.x3, s=0.5, c="k")
            ax[1, 1].set_xlim([np.min(self.x1), np.max(self.x1)])
            ax[1, 1].set_ylim([meanx3 - sfac * stdx3, meanx3 + sfac * stdx3])
            xlabel = "$x_1 [c\omega_p^{-1}]$"
            ylabel = "$x_3 [c\omega_p^{-1}]$"
            ax[1, 1].set_xlabel(xlabel, fontsize=20)
            ax[1, 1].set_ylabel(ylabel, fontsize=20)
            plt.draw()
            plt.pause(0.01)
            ans = ask_user_input(Message="Enter a new factor to be considered. Use fac > 1 if you want a bigger area to be considered and 0 < fac < 1 if want a smaller area.\nDefault answer [Not necessary]: ")
            try:
                newfac = float(ans)
                sfac *= newfac
                plt.close(fig)
            except ValueError:
                x2min = meanx2 - sfac * stdx2
                x2max = meanx2 + sfac * stdx2
                x3min = meanx3 - sfac * stdx3
                x3max = meanx3 + sfac * stdx3
                break
        plt.close(fig)
        tg = np.where((self.x2 > x2min) & (self.x2 < x2max) & (self.x3 > x3min) & (self.x3 < x3max))
        self.filter_particles(tg)

    def print_params(self):
        bp = DataBeamParameters(
            n0=self.n0,
            x1=self.x1,
            x2=self.x2,
            x3=self.x3,
            ene=self.ene,
            q=self.q,
            p1=self.p1,
            p2=self.p2,
            p3=self.p3,
            dx1=self.dx1,
            dx2=self.dx2,
            dx3=self.dx3,
            alg=self.alg,
        )
        energy = bp.energy()
        charge = bp.charge()
        energy_spread = bp.energy_spread()
        divx = bp.div_rms_x()
        divy = bp.div_rms_y()
        em_tr_x = bp.emit_tr_x()
        em_tr_y = bp.emit_tr_y()
        em_ph_x = bp.emit_ph_n_x()
        em_ph_y = bp.emit_ph_n_y()
        gamma_x = bp.gamma_twiss_x()
        gamma_y = bp.gamma_twiss_y()
        params = (
            "Energy [MeV] = "
            + "{:.2f}".format(energy)
            + "\nEnergy spread [%] = "
            + "{:.2f}".format(energy_spread)
            + "\nCharge [pC] = "
            + "{:.2f}".format(charge)
            + "\nDivergence x [rad] =  "
            + "{:.2e}".format(divx)
            + "\nDivergence y [rad] = "
            + "{:.2e}".format(divy)
            + "\nTrace emittance x [um] =  "
            + "{:.2e}".format(em_tr_x)
            + "\nTrace emittance y [um] = "
            + "{:.2e}".format(em_tr_y)
            + "\nNormalized phase emittance x [um] = "
            + "{:.2f}".format(em_ph_x)
            + "\nNormalized phase emittance y [um] = "
            + "{:.2f}".format(em_ph_y)
            + "\nGamma Twiss x [m^-1] = "
            + "{:.2e}".format(gamma_x)
            + "\nGamma Twiss y [m^-1] = "
            + "{:.2e}".format(gamma_y)
        )
        print(params)
        return

    def longitudinal_position_based_cutoff(self):
        std_plot()
        self.print_params()
        x1min_ran = np.min(self.x1)
        x1max_ran = np.max(self.x1)
        x1range = np.linspace(x1min_ran, x1max_ran, num=3)
        fig, ax = plt.subplots(figsize=(15, 8), ncols=1, nrows=2, constrained_layout=True)
        ax[0].scatter(self.x1, self.ene, s=0.5, c="k")
        ax[0].set_xticks(x1range)
        ax[0].xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax[0].set_xlim([np.min(self.x1), np.max(self.x1)])
        ax[0].set_ylim([np.min(self.ene), np.max(self.ene)])
        xlabel = "$x_1 [c\omega_p^{-1}]$"
        ylabel = "$Energy [m_e c^2]$"
        ax[0].set_xlabel(xlabel, fontsize=20)
        ax[0].set_ylabel(ylabel, fontsize=20)
        plt.draw()
        plt.pause(0.01)
        """
        """
        x1mm = np.min(self.x1)
        x1mp = np.max(self.x1)
        flag = True
        flagmin = True
        flagmax = True
        x1bk = self.x1
        x2bk = self.x2
        x3bk = self.x3
        p1bk = self.p1
        p2bk = self.p2
        p3bk = self.p3
        enebk = self.ene
        qbk = self.q
        tagbk = self.tag
        while flag:
            self.x1 = x1bk
            self.x2 = x2bk
            self.x3 = x3bk
            self.p1 = p1bk
            self.p2 = p2bk
            self.p3 = p3bk
            self.ene = enebk
            self.q = qbk
            self.tag = tagbk
            x1min = ask_user_input(Message="Enter the new lowest value of x1.\nDefault: x1min = [LAST VALUE USED]\nx1min = ")
            try:
                x1min = float(x1min)
                x1mm = x1min
                flagmin = True
            except ValueError:
                x1min = x1mm
                flagmin = False
            x1max = ask_user_input(Message="Enter the new highest value of x1.\nDefault: x1max = [LAST VALUE USED]\nx1max = ")
            try:
                x1max = float(x1max)
                x1mp = x1max
                flagmax = True
            except ValueError:
                x1max = x1mp
                flagmax = False
            if not flagmax and not flagmin:
                tg = np.where((self.x1 > x1min) & (self.x1 < x1max))
                self.filter_particles(tg)
                break
            else:
                tg = np.where((self.x1 > x1min) & (self.x1 < x1max))
                self.filter_particles(tg)
                self.print_params()
                ax[1].cla()
                x1range = np.linspace(x1min, x1max, num=3)
                ax[1].scatter(self.x1, self.ene, s=0.5, c="k")
                ax[1].set_xticks(x1range)
                ax[1].xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
                ax[1].set_xlim([x1min, x1max])
                ax[1].set_ylim([np.min(self.ene), np.max(self.ene)])
                # xlabel='$x_1 [c\omega_p^{-1}]$'
                # ylabel='$x_2 [c\omega_p^{-1}]$'
                ax[1].set_xlabel(xlabel, fontsize=20)
                ax[1].set_ylabel(ylabel, fontsize=20)
                plt.draw()
                plt.pause(0.01)
        plt.close(fig)


"""
"""


class FollowCutBeam(tk.Frame):
    def __init__(self, file=False, n0=False):
        if not file:
            self.file = askfile_path("/Volumes/EXT/Thales/EuPRAXIA/runs", title="Choose the file")
        else:
            self.file = file
        self.alg = osiris_alg_test(self.file)
        if self.alg not in ["q3D", "3D"]:
            print("Invalid algorithm. Aborting...")
            sys.exit()
        if not n0:
            self.n0 = ask_user_input("Enter the plasma density in cm^-3 (eg: 1e19)\nn0 = ")
        else:
            self.n0 = n0
        if self.alg == "3D":
            self.quants = ["ene", "x1", "x2", "x3", "p1", "p2", "p3", "q", "tag"]
            self.attrs, self.data = osiris_open_particle_data(self.file, self.quants)
            self.dx1 = self.attrs["DX"][0]
            self.dx2 = self.attrs["DX"][1]
            self.dx3 = self.attrs["DX"][2]
        elif self.alg == "q3D":
            self.quants = ["ene", "x1", "x3", "x4", "p1", "p2", "p3", "q", "tag"]
            self.attrs, self.data = osiris_open_particle_data(self.file, self.quants)
            self.dx1 = self.attrs["DX"][0]
            self.dx2 = self.attrs["DX"][1]
            self.dx3 = False
        self.open_raw_data()
        self.reset_cut_data()
        self.x1min = np.min(self.x1_fixed)
        self.x1max = np.max(self.x1_fixed)
        self.enemin = np.min(self.ene_fixed)
        self.enemax = np.max(self.ene_fixed)
        self.rmax = np.sqrt(np.max(self.x2_fixed**2 + self.x3_fixed**2))
        """
        """
        self.master = tk.Tk()
        tk.Frame.__init__(self, self.master)
        # self.master.geometry('1000x1000')
        self.pack()
        self.make_widgets()

    def make_widgets(self):
        self.top = self.winfo_toplevel()
        self.top.title("Beam")
        self.PlotFrame = tk.Frame(self.top)  # ,width=1080,height=972)
        fig = self.plot()
        canvas = FigureCanvasTkAgg(fig, master=self.PlotFrame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas, self.PlotFrame)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.PlotFrame.pack(side=tk.LEFT)
        self.PlotFrame.update()
        height_std = self.PlotFrame.winfo_height()
        """
        """
        params = self.print_params()
        self.BeamParamFrame = tk.Frame(self.top, width=500, height=340, relief=tk.SUNKEN)
        self.BeamParamFrame.pack_propagate(False)
        text = tk.Text(self.BeamParamFrame)
        text.configure(font=Font(size=25))
        text.pack(side=tk.LEFT)
        text.insert(1.0, params)
        self.BeamParamFrame.pack(side=tk.LEFT)
        """
        """
        text2 = tk.Text(self.BeamParamFrame)
        text2.configure(font=Font(size=25))
        text2.pack(side=tk.BOTTOM)
        text2.insert(1.0, params)
        self.BeamParamFrame.pack(side=tk.BOTTOM)

    def open_raw_data(self):
        self.ene_fixed = self.data[0][:]
        self.x1_fixed = self.data[1][:]
        self.x2_fixed = self.data[2][:]
        self.x3_fixed = self.data[3][:]
        self.p1_fixed = self.data[4][:]
        self.p2_fixed = self.data[5][:]
        self.p3_fixed = self.data[6][:]
        self.q_fixed = self.data[7][:]
        self.x2_fixed -= np.average(self.x2_fixed, weights=self.q_fixed)
        self.x3_fixed -= np.average(self.x3_fixed, weights=self.q_fixed)
        self.tag_fixed = self.data[8][:]

    def reset_cut_data(self):
        self.ene = self.data[0][:]
        self.x1 = self.data[1][:]
        self.x2 = self.data[2][:]
        self.x3 = self.data[3][:]
        self.p1 = self.data[4][:]
        self.p2 = self.data[5][:]
        self.p3 = self.data[6][:]
        self.q = self.data[7][:]
        self.x2 -= np.average(self.x2, weights=self.q)
        self.x3 -= np.average(self.x3, weights=self.q)
        self.tag = self.data[8][:]

    def save_last(self):
        self.ene_bk = self.ene
        self.x1_bk = self.x1
        self.x2_bk = self.x2
        self.x3_bk = self.x3
        self.p1_bk = self.p1
        self.p2_bk = self.p2
        self.p3_bk = self.p3
        self.q_bk = self.q
        self.tag_bk = self.tag

    def restore_last(self):
        self.ene = self.ene_bk
        self.x1 = self.x1_bk
        self.x2 = self.x2_bk
        self.x3 = self.x3_bk
        self.p1 = self.p1_bk
        self.p2 = self.p2_bk
        self.p3 = self.p3_bk
        self.q = self.q_bk
        self.tag = self.tag_bk

    def filter_particles(self, tg):
        lenold = len(self.x1)
        tg = tg[0]
        self.ene = self.ene[tg]
        self.x1 = self.x1[tg]
        self.x2 = self.x2[tg]
        self.x3 = self.x3[tg]
        self.p1 = self.p1[tg]
        self.p2 = self.p2[tg]
        self.p3 = self.p3[tg]
        self.q = self.q[tg]
        self.tag = self.tag[tg]
        lennew = len(self.x1)
        print(lennew, "of", lenold, "particles remain.")

    def print_params(self):
        bp = DataBeamParameters(
            n0=self.n0,
            x1=self.x1,
            x2=self.x2,
            x3=self.x3,
            ene=self.ene,
            q=self.q,
            p1=self.p1,
            p2=self.p2,
            p3=self.p3,
            dx1=self.dx1,
            dx2=self.dx2,
            dx3=self.dx3,
            alg=self.alg,
        )
        energy = bp.energy()
        charge = bp.charge()
        energy_spread = bp.energy_spread()
        divx = bp.div_rms_x()
        divy = bp.div_rms_y()
        em_tr_x = bp.emit_tr_x()
        em_tr_y = bp.emit_tr_y()
        em_ph_x = bp.emit_ph_n_x()
        em_ph_y = bp.emit_ph_n_y()
        gamma_x = bp.gamma_twiss_x()
        gamma_y = bp.gamma_twiss_y()
        params = (
            "Energy [MeV] = "
            + "{:.2f}".format(energy)
            + "\nEnergy spread [%] = "
            + "{:.2f}".format(energy_spread)
            + "\nCharge [pC] = "
            + "{:.2f}".format(charge)
            + "\nDivergence x [rad] =  "
            + "{:.2e}".format(divx)
            + "\nDivergence y [rad] = "
            + "{:.2e}".format(divy)
            + "\nTrace emittance x [um] =  "
            + "{:.2e}".format(em_tr_x)
            + "\nTrace emittance y [um] = "
            + "{:.2e}".format(em_tr_y)
            + "\nNormalized phase emittance x [um] = "
            + "{:.2f}".format(em_ph_x)
            + "\nNormalized phase emittance y [um] = "
            + "{:.2f}".format(em_ph_y)
            + "\nGamma Twiss x [m^-1] = "
            + "{:.2e}".format(gamma_x)
            + "\nGamma Twiss y [m^-1] = "
            + "{:.2e}".format(gamma_y)
        )
        return params

    def plot(self):
        std_plot()
        fig = Figure(figsize=(15, 13.5))  # ,ncols=2,nrows=2)
        xlabel = "$x_1 [c\omega_p^{-1}]$"
        ylabel = "$Ene [m_ec^2]$"
        ax00 = fig.add_subplot(221)
        ax00.scatter(self.x1_fixed, self.ene_fixed, s=0.5, c="k")
        ax00.set_xticks(np.arange(int(np.floor(self.x1min)), int(np.ceil(self.x1max)), 0.5))
        ax00.set_yticks(np.arange(int(np.floor(self.enemin)), int(np.ceil(self.enemax)), 100))
        ax00.xaxis.grid(True, which="major", color="r")
        ax00.yaxis.grid(True, which="major", color="r")
        ax00.set_xlim([self.x1min, self.x1max])
        ax00.set_ylim([self.enemin, self.enemax])
        ax00.set_xlabel(xlabel, fontsize=20)
        ax00.set_ylabel(ylabel, fontsize=20)
        ax00.set_position([0.10, 0.55, 0.36, 0.4])
        ax00.set_title("Original beam", pad=10, fontsize=30)
        """
        """
        ax01 = fig.add_subplot(222)
        ax01.scatter(self.x1, self.ene, s=0.5, c="k")
        ax01.set_xticks(np.arange(int(np.floor(self.x1min)), int(np.ceil(self.x1max)), 0.5))
        ax01.set_yticks(np.arange(int(np.floor(self.enemin)), int(np.ceil(self.enemax)), 100))
        ax01.xaxis.grid(True, which="major", color="r")
        ax01.yaxis.grid(True, which="major", color="r")
        ax01.set_xlim([self.x1min, self.x1max])
        ax01.set_ylim([self.enemin, self.enemax])
        ax01.set_xlabel(xlabel, fontsize=20)
        ax01.set_ylabel(ylabel, fontsize=20)
        ax01.set_position([0.60, 0.55, 0.36, 0.4])
        ax01.set_title("Cutted beam", pad=10, fontsize=30)
        """
        """
        xlabel = "$x_2 [c\omega_p^{-1}]$"
        ylabel = "$x_3 [c\omega_p^{-1}]$"
        ax10 = fig.add_subplot(223)
        ax10.scatter(self.x2_fixed, self.x3_fixed, s=0.5, c="k")
        ax10.set_xticks(np.arange(int(np.floor(-self.rmax)), int(np.ceil(self.rmax)), 0.5))
        ax10.set_yticks(np.arange(int(np.floor(-self.rmax)), int(np.ceil(self.rmax)), 0.5))
        ax10.xaxis.grid(True, which="major", color="r")
        ax10.yaxis.grid(True, which="major", color="r")
        ax10.set_xlim([-self.rmax, self.rmax])
        ax10.set_ylim([-self.rmax, self.rmax])
        ax10.set_xlabel(xlabel, fontsize=20)
        ax10.set_ylabel(ylabel, fontsize=20)
        ax10.set_position([0.10, 0.07, 0.36, 0.4])
        """
        """
        ax11 = fig.add_subplot(224)
        ax11.scatter(self.x2, self.x3, s=0.5, c="k")
        ax11.set_xticks(np.arange(int(np.floor(-self.rmax)), int(np.ceil(self.rmax)), 0.5))
        ax11.set_yticks(np.arange(int(np.floor(-self.rmax)), int(np.ceil(self.rmax)), 0.5))
        ax11.xaxis.grid(True, which="major", color="r")
        ax11.yaxis.grid(True, which="major", color="r")
        ax11.set_xlim([-self.rmax, self.rmax])
        ax11.set_ylim([-self.rmax, self.rmax])
        ax11.set_xlabel(xlabel, fontsize=20)
        ax11.set_ylabel(ylabel, fontsize=20)
        ax11.set_position([0.60, 0.07, 0.36, 0.4])
        return fig
