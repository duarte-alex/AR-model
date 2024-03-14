#%%
import scipy
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
from utilities.plot import std_plot
from math import erf
from scipy.optimize import newton, root_scalar, fsolve
from plasmapy.dispersion import plasma_dispersion_func as Z
from scipy.special import jvp as dJ
from scipy.integrate import quad as quadsci
import time

std_plot()
# %%

kz = 0.01
kp = 3
w0g = 4.048 - 0.0517j
# w0g = 3.9267881876 - 0.00001j
# w0g = 3.91 - 0.0001j
Tz = 20 / 511e3
Tp = 20 / 511e3
p0 = 0.2
Omega = 4
sqrt2Tzkz = np.sqrt(2 * Tz) * kz
paux = p0 / np.sqrt(2 * Tp)
eta = np.exp(-(paux ** 2)) + np.sqrt(np.pi) * paux * (1 + erf(paux))
norm = 1 / (eta * Tp)

if np.abs(np.imag(w0g)) < 1e-8:
    sigma = 1
elif np.imag(w0g) > 0:
    sigma = 0
else:
    sigma = 2


def fp(p):
    return norm * np.exp(-((p - p0) ** 2) / (2 * Tp))


def dfp(p):
    return -norm / Tp * (p - p0) * np.exp(-((p - p0) ** 2) / (2 * Tp))


def r1(p):
    return np.log10(fp(p) / norm) + 10


p0guess = p0 + np.sqrt(Tp)
l = fsolve(r1, np.array([p0guess]))[0]
l = np.abs(p0 - l)
lim = [np.max([0, p0 - l]), p0 + l]
lower_lim = np.max([lim[0], 0.0])
upper_lim = lim[1]


pol_r, pol_i = np.real(Omega / w0g), np.imag(Omega / w0g)
pol = Omega / w0g

if np.abs(pol_i) < 1e-10:
    pol = np.real(pol)
    pol_0 = np.sqrt(pol ** 2 - 1)
    # print(lower_lim, pol_0, upper_lim)
    if pol_0 == np.nan:
        pol_0 = -100

    def int_pure(p):
        bp = kp * p / Omega
        dd = dJ(1, bp) ** 2
        return p ** 2 * dfp(p) * dd / (np.sqrt(1 + p ** 2) - pol)

    #   print(pol, lower_lim, upper_lim)

    if (pol_0 > lower_lim) and (pol_0 < upper_lim):
        intr1 = quadsci(int_pure, lower_lim, pol_0 - 1e-8, limit=150)[0]
        intr2 = quadsci(int_pure, pol_0 + 1e-8, upper_lim, limit=150)[0]
        intr = intr1 + intr2
    else:
        intr = quadsci(int_pure, lower_lim, upper_lim, limit=150, complex_func=True)[0]

else:
    polre = np.real(pol)

    def int_pure(p):
        bp = kp * p / Omega
        dd = dJ(1, bp) ** 2
        return p ** 2 * dfp(p) * dd / (np.sqrt(1 + p ** 2) - pol)

    intr = quadsci(int_pure, lower_lim, upper_lim, limit=150, complex_func=True)[0]

a = Omega / w0g
af = np.sqrt(a ** 2 - 1)
bp = kp * af / Omega
dd = dJ(1, bp) ** 2
res = a * af ** 2 * dfp(af) * dd / af
qq = 2j * np.pi * res

# print(intr - sigma / 2 * qq)

den = kz * np.sqrt(2 * Tz)
ofac = Omega / den


def int_1(p):
    g = np.sqrt(1 + p ** 2)
    bp = kp * p / Omega
    dd = dJ(1, bp) ** 2
    return p ** 2 / g * dfp(p) * dd


def int_2(p):

    g = np.sqrt(1 + p ** 2)
    xi = (g * w0g - Omega) / den
    bp = kp * p / Omega
    dd = dJ(1, bp) ** 2
    return p ** 2 / g * dfp(p) * dd * Z(xi) * ofac


int1 = quadsci(int_1, lower_lim, upper_lim, limit=150, complex_func=True)[0]
int2 = quadsci(int_2, lower_lim, upper_lim, limit=150, complex_func=True)[0]
# print(int1 - int2)


def integral(w):
    Oo = Omega / w
    fac_r, fac_i = np.real(Oo), np.imag(Oo)
    Oo2 = Omega / np.abs(w) ** 2
    wr, wi = np.real(w), np.imag(w)
    Wii = Oo2 * wi
    ppole = np.sqrt((Omega / w) ** 2 - 1)
    re_pole = np.real(ppole)

    if (np.abs(fac_i) < 1e-8) and (fac_r > 1) and (re_pole > lower_lim) and (re_pole < upper_lim):
        flag = "Half-pole"

        def int_pure(p):
            bp = kp * p / Omega
            dd = dJ(1, bp) ** 2
            return p ** 2 * dfp(p) * dd / (np.sqrt(1 + p ** 2) - fac_r)

        intr1 = quadsci(int_pure, lower_lim, re_pole - 1e-8, limit=150)[0]
        intr2 = quadsci(int_pure, re_pole + 1e-8, upper_lim, limit=150)[0]

        bp = kp * re_pole / Omega
        dd = dJ(1, bp) ** 2
        Res = -1j * np.pi * np.sqrt(1 + re_pole ** 2) / re_pole * (re_pole ** 2 * dfp(re_pole) * dd)
        intr = intr1 + intr2 + Res

        # return intr

    else:

        if (fac_i > 0) and (fac_r > 1) and (re_pole > lower_lim) and (re_pole < upper_lim):
            flag = "Full-pole"
            bp = kp * ppole / Omega
            dd = dJ(1, bp) ** 2
            Res = -2j * np.pi * np.sqrt(1 + ppole ** 2) / ppole * (ppole ** 2 * dfp(ppole) * dd)
        else:
            flag = "Std"
            Res = 0

        def int_re(p):
            bp = kp * p / Omega
            dd = dJ(1, bp) ** 2
            gn = p ** 2 * dfp(p) * dd
            Wrr = np.sqrt(1 + p ** 2) - Oo2 * wr
            den = Wrr ** 2 + Wii ** 2
            return Wrr * gn / den

        def int_im(p):
            bp = kp * p / Omega
            dd = dJ(1, bp) ** 2
            gn = p ** 2 * dfp(p) * dd
            Wrr = np.sqrt(1 + p ** 2) - Oo2 * wr
            den = Wrr ** 2 + Wii ** 2
            return Wii * gn / den

        intr1 = quadsci(int_re, lower_lim, upper_lim, limit=150)[0]
        intr2 = quadsci(int_im, lower_lim, upper_lim, limit=150)[0]
        intr = intr1 - 1j * intr2 + Res
        """

        def int_pure(p):
            bp = kp * p / Omega
            dd = dJ(1, bp) ** 2
            return p ** 2 * dfp(p) * dd / (np.sqrt(1 + p ** 2) - Oo)

        intr0 = quadsci(int_pure, lower_lim, upper_lim, limit=150, complex_func=True)[0]
        return intr0 + Res
        """
    print(intr, Res, ppole)

    return


def disp_kp_Tp(w, kp, n=1):

    nrange = np.arange(-n, n + 1)
    idx0 = int((nrange.shape[0] - 1) / 2)
    resonance = np.zeros(nrange.shape, dtype=np.int8)
    Oo2 = Omega / np.abs(w) ** 2
    wr, wi = np.real(w), np.imag(w)
    Wii_std = Oo2 * wi
    ppole = np.sqrt((Omega / w) ** 2 - 1)

    if np.abs(wi) < 1e-8:
        sigma = 1
        flag = False
    elif wi > 0:
        sigma = 0
        flag = True
    else:
        sigma = 2
        flag = True

    kO = kp / Omega
    test_res = np.real(np.sqrt((nrange * Omega / w) ** 2 - 1))

    for i in range(idx0 + 1, nrange.shape[0]):
        if (test_res[i] < upper_lim) and (test_res[i] > lower_lim):
            resonance[i] = 1

    arg1 = np.where(resonance == 0)[0]
    arg2 = np.where(resonance == 1)[0]
    integrals = np.zeros(nrange.shape, dtype=np.complex128)

    def int_simple(n):
        def intre(p, n):
            Wrr = np.sqrt(1 + p ** 2) - n * Oo2 * wr
            Wii = n * Wii_std
            den = Wrr ** 2 + Wii ** 2
            gg = -norm / Tp * (p - p0) * np.exp(-((p - p0) ** 2) / (2 * Tp)) * dJ(n, kO * p) ** 2 * p ** 2
            return gg * Wrr / den

        def intim(p, n):
            Wrr = np.sqrt(1 + p ** 2) - n * Oo2 * wr
            Wii = n * Wii_std
            den = Wrr ** 2 + Wii ** 2
            gg = -norm / Tp * (p - p0) * np.exp(-((p - p0) ** 2) / (2 * Tp)) * dJ(n, kO * p) ** 2 * p ** 2
            return -gg * Wii / den

        intr = quadsci(intre, lower_lim, upper_lim, args=(n,), limit=150)[0]
        inti = quadsci(intim, lower_lim, upper_lim, args=(n,), limit=150)[0]

        return intr + 1j * inti

    def int_pv(n):
        ppole = np.sqrt((n * Omega / w) ** 2 - 1)
        re_pole = np.real(ppole)
        Ok = n * Omega / w

        def int00(p, n):
            gg = -norm / Tp * (p - p0) * np.exp(-((p - p0) ** 2) / (2 * Tp)) * dJ(n, kO * p) ** 2 * p ** 2
            den = np.sqrt(1 + p ** 2) - Ok
            return gg / den

        intr1 = quadsci(int00, lower_lim, re_pole - 1e-8, args=(n,), limit=150)[0]
        intr2 = quadsci(int00, re_pole + 1e-8, upper_lim, args=(n,), limit=150)[0]
        return intr1 + intr2

    def res(p, n):
        return -norm / Tp * (p - p0) * np.exp(-((p - p0) ** 2) / (2 * Tp)) * dJ(n, kO * p) ** 2 * p * np.sqrt(1 + p ** 2)

    residue = 0
    for j in arg2:
        if flag:
            ppole = np.sqrt((nrange[j] * Omega / w) ** 2 - 1)
            residue = -1j * np.pi * sigma * res(ppole, nrange[j])
            sol = int_simple(nrange[j])
            integrals[j] = sol + residue
        else:
            ppole = np.sqrt((nrange[j] * Omega / w) ** 2 - 1)
            # print(ppole)
            residue = -1j * np.pi * sigma * res(ppole, nrange[j])
            # print(residue)
            sol = int_pv(nrange[j])
            integrals[j] = sol + residue

    for j in arg1:
        sol = int_simple(nrange[j])
        integrals[j] = sol

    print(integrals[-1], ppole, residue)
    sum_ints = np.sum(integrals)
    return
    # return w ** 2 - kp ** 2 - sum_ints


integral(w0g)
disp_kp_Tp(w0g, kp)

# %%
