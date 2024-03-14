# %%
from plasmapy.dispersion import plasma_dispersion_func as Z
from scipy.optimize import newton, root_scalar, fsolve, fminbound, bisect, minimize_scalar, bracket
import numpy as np
import matplotlib.pyplot as plt
import sys
from sympy import lowergamma
from utilities.plot import std_plot
from quadpy import quad
from scipy.special import jv as J
from scipy.special import j0 as J0
from scipy.special import j1 as J1
from scipy.special import jvp as dJ
from scipy.integrate import quad as quadsci
from tqdm import tqdm
from scipy.misc import derivative as D
from math import erf

# import time
std_plot()


# %%
class Weibel:
    def __init__(self, A, Tcold):
        self.A = A
        self.Tcold = Tcold
        self.d = 0.0000316228
        self.kmin = 3 * self.d
        self.kmax = np.sqrt(A / 3) + 3 * self.d

    def growth_rate(self, k):
        if k**2 > self.A:
            return 0.0
        A = self.A
        Tcold = self.Tcold
        kpos = np.sqrt(A - k**2) + 1e-4

        def dis_rel(gamma):
            xi = gamma / (k * np.sqrt(2 * Tcold))
            f = k**2 + gamma**2 - A + (A + 1) * xi * np.imag(Z(xi * 1j))
            return f

        gamma = root_scalar(dis_rel, bracket=(0.0, kpos), method="brentq")
        gamma = gamma.root
        return gamma

    def k_max_growth_rate(self):
        def dgr(k):
            der = (self.growth_rate(k + self.d) - self.growth_rate(k - self.d)) / (2 * self.d)
            return der

        if self.A < 2:
            maxk = root_scalar(
                dgr,
                bracket=(self.kmin, self.kmax),
                method="brentq",
                x0=np.sqrt(self.A / 3),
            )
        else:
            maxk = root_scalar(
                dgr,
                bracket=(self.kmin, self.kmax),
                method="brentq",
                x0=0.5 * np.sqrt(self.A / 3),
            )
        maxk = maxk.root
        return maxk

    def max_growth_rate(self):
        maxk = self.k_max_growth_rate()
        gr = self.growth_rate(maxk)
        return gr


# %%


class Weibel_relativistic_ring:
    def __init__(self, p0, Tz):
        self.p0 = p0
        self.Tz = Tz
        self.g0 = np.sqrt(1 + p0**2)

    def disp(self, w, k):
        g0 = self.g0
        p0 = self.p0
        Tz = self.Tz
        A = (1 + g0**2) / (2 * g0**3)
        B = p0**2 / (2 * g0 * Tz)
        eta = g0 * w / np.sqrt(2 * Tz * k**2)
        Z_eta = Z(eta)
        return w**2 - k**2 - A + B * (1 + eta * Z_eta)

    def sol(self, k, w0g):
        if type(w0g) is float or type(w0g) is complex:
            w0g = [w0g]
        wg = np.array(w0g, dtype=np.complex128)
        len = wg.shape[0]
        roots = np.zeros(wg.shape, dtype=np.complex128)

        for j in range(len):
            try:
                w0 = newton(self.disp, wg[j], tol=1e-12, args=(k,), maxiter=10000)
            except:
                w0 = np.complex128(0)
            if np.abs(self.disp(w0, k)) < 1e-8:
                roots[j] = w0

        return roots

    def max_gr(self):
        Tz = self.Tz
        p0 = self.p0
        gamma0 = np.sqrt(1 + p0**2)
        maxk = 7 / 10 * p0 / np.sqrt(gamma0 * Tz)
        kzt = (maxk) / 2
        w0g = np.array([0.5j * p0 / np.sqrt(2 * g0**3)], dtype=np.complex128)

        def fmin(kz, wg):
            return -np.imag(self.sol(kz, wg))

        kmax = fminbound(fmin, 0, maxk, args=(w0g,))
        gr = self.sol(kmax, w0g)
        return kmax, gr


# %%


class Weibel_approx:
    def __init__(self, Thot, Tcold):
        self.Thot = Thot
        self.Tcold = Tcold

    def growth_rate_0(self, kz):
        Th = self.Thot
        Tz = self.Tcold
        p = np.array([1, 0, -(kz**2 + 1), 0, -Th * kz**2])
        roots = np.roots(p)
        max_imaginary_part = max(roots, key=lambda root: root.imag)
        return max_imaginary_part

    def growth_rate_1(self, kz):
        Th = self.Thot
        Tz = self.Tcold
        p = np.array([1, 0, -(kz**2 + 1), 0, -Th * kz**2, 0, -3 * Th * kz**4 * Tz])
        roots = np.roots(p)
        max_imaginary_part = max(roots, key=lambda root: root.imag)
        print(roots)
        return max_imaginary_part

    def growth_rate_2(self, kz):
        Th = self.Thot
        Tz = self.Tcold
        p = np.array(
            [
                1,
                0,
                -(kz**2 + 1),
                0,
                -Th * kz**2,
                0,
                -3 * Th * kz**4 * Tz,
                0,
                -15 * Th * kz**6 * Tz**2,
            ]
        )
        roots = np.roots(p)
        max_imaginary_part = max(roots, key=lambda root: root.imag)
        return max_imaginary_part


# %%
class Maser:
    def __init__(self, p0, Omega, Tz=None, Tp=None):
        self.p0 = p0
        self.Omega = Omega
        self.Tz = Tz
        self.Tp = Tp
        if Tz == 0.0:
            Tz = None
        if Tp == 0.0:
            Tp = None

        if Tz == None and Tp == None:
            self.Tz_warm = False
            self.Tp_warm = False
        elif Tz != None and Tp == None:
            self.Tz_warm = True
            self.Tp_warm = False
        elif Tp != None and Tz == None:
            self.Tz_warm = False
            self.Tp_warm = True
        else:
            self.Tz_warm = True
            self.Tp_warm = True

        if self.Tp_warm:
            paux = p0 / np.sqrt(2 * Tp)
            self.eta = np.exp(-(paux**2)) + np.sqrt(np.pi) * paux * (1 + erf(paux))

            def fp(p):
                return np.exp(-((p - p0) ** 2) / (2 * Tp))

            def r1(p):
                return np.log10(fp(p)) + 10

            p0guess = p0 + np.sqrt(Tp)
            l = fsolve(r1, np.array([p0guess]))[0]
            l = np.abs(p0 - l)
            lim = [np.max([0, p0 - l]), p0 + l]
            self.lower_lim = np.max([lim[0], 0.0])
            self.upper_lim = lim[1]
            self.g_low_lim = np.sqrt(1 + self.lower_lim**2)
            self.g_upp_lim = np.sqrt(1 + self.upper_lim**2)
            # print(self.g_low_lim, self.g_upp_lim)

    def sol_kz_cold(self, kz, all=True):
        k = kz
        Omega = self.Omega
        p0 = self.p0
        gamma0 = np.sqrt(1 + p0**2)
        c4 = 1
        c3 = -2 * Omega / gamma0
        c2 = p0**2 / (2 * gamma0**3) + Omega**2 / gamma0**2 - 1 / gamma0 - k**2
        c1 = Omega / gamma0**2 + 2 * Omega * k**2 / gamma0
        c0 = -(k**2) * p0**2 / (2 * gamma0**3) - Omega**2 * k**2 / gamma0**2
        coeffs = [c4, c3, c2, c1, c0]
        roots = np.roots(coeffs)
        if all == False:
            id = np.argsort((np.imag(roots)))[-1]
            return roots[id]
        else:
            return roots

    def kz_instability_threshold(self):
        p0 = self.p0
        gamma0 = np.sqrt(1 + p0**2)
        Omega = self.Omega / gamma0
        FAC1 = p0**2 / (2 * gamma0**3)
        FAC2 = 1 / gamma0
        c10 = -16 * FAC1
        c09 = 0
        c08 = -64 * FAC1**2 - 64 * FAC1 * FAC2 + 64 * FAC1 * Omega**2
        c07 = 0
        c06 = -96 * FAC1**3 - 64 * FAC1**2 * FAC2 - 96 * FAC1 * FAC2**2 + 64 * FAC1**2 * Omega**2 + 32 * FAC1 * FAC2 * Omega**2 + 4 * FAC2**2 * Omega**2 - 96 * FAC1 * Omega**4
        c05 = 0
        c04 = -64 * FAC1**4 + 64 * FAC1**3 * FAC2 + 64 * FAC1**2 * FAC2**2 - 64 * FAC1 * FAC2**3 - 64 * FAC1**3 * Omega**2 + 576 * FAC1**2 * FAC2 * Omega**2 + 4 * FAC1 * FAC2**2 * Omega**2 + 12 * FAC2**3 * Omega**2 + 64 * FAC1**2 * Omega**4 + 128 * FAC1 * FAC2 * Omega**4 - 8 * FAC2**2 * Omega**4 + 64 * FAC1 * Omega**6
        c03 = 0
        c02 = (
            -16 * FAC1**5
            + 64 * FAC1**4 * FAC2
            - 96 * FAC1**3 * FAC2**2
            + 64 * FAC1**2 * FAC2**3
            - 16 * FAC1 * FAC2**4
            - 64 * FAC1**4 * Omega**2
            + 32 * FAC1**3 * FAC2 * Omega**2
            - 4 * FAC1**2 * FAC2**2 * Omega**2
            + 24 * FAC1 * FAC2**3 * Omega**2
            + 12 * FAC2**4 * Omega**2
            - 96 * FAC1**3 * Omega**4
            - 128 * FAC1**2 * FAC2 * Omega**4
            - 96 * FAC1 * FAC2**2 * Omega**4
            + 20 * FAC2**3 * Omega**4
            - 64 * FAC1**2 * Omega**6
            - 96 * FAC1 * FAC2 * Omega**6
            + 4 * FAC2**2 * Omega**6
            - 16 * FAC1 * Omega**8
        )
        c01 = 0
        c00 = -4 * FAC1**3 * FAC2**2 * Omega**2 + 12 * FAC1**2 * FAC2**3 * Omega**2 - 12 * FAC1 * FAC2**4 * Omega**2 + 4 * FAC2**5 * Omega**2 - 8 * FAC1**2 * FAC2**2 * Omega**4 - 20 * FAC1 * FAC2**3 * Omega**4 + FAC2**4 * Omega**4 - 4 * FAC1 * FAC2**2 * Omega**6
        coeffs = [c10, c09, c08, c07, c06, c05, c04, c03, c02, c01, c00]
        roots = np.roots(coeffs)
        idx = np.where(np.abs(np.imag(roots)) < 1e-10)
        return np.real(np.max(roots[idx]))

    def kz_thermal_instability_threshold(self, kg):
        Omega = self.Omega
        p0 = self.p0
        gamma0 = np.sqrt(1 + p0**2)
        Tz = self.Tz

        def sol(k):
            w = ((gamma0**2 * k**2 + Omega**2) * p0**2 - np.sqrt((-(gamma0**2) * k**2 + Omega**2) ** 2 * p0**4 - 4 * k**2 * Omega**2 * p0**2 * (-2 * gamma0**2 + p0**2) * Tz)) / (2 * gamma0 * Omega * p0**2)

            return self.disp_kz_Tz(w, k)

        def sol2(k):
            w = ((gamma0**2 * k**2 + Omega**2) * p0**2 + np.sqrt((-(gamma0**2) * k**2 + Omega**2) ** 2 * p0**4 - 4 * k**2 * Omega**2 * p0**2 * (-2 * gamma0**2 + p0**2) * Tz)) / (2 * gamma0 * Omega * p0**2)

            return self.disp_kz_Tz(w, k)

        kg = np.complex128(kg + 0j)
        k = newton(sol, kg, tol=1e-12, maxiter=10000)
        w1 = ((gamma0**2 * k**2 + Omega**2) * p0**2 - np.sqrt((-(gamma0**2) * k**2 + Omega**2) ** 2 * p0**4 - 4 * k**2 * Omega**2 * p0**2 * (-2 * gamma0**2 + p0**2) * Tz)) / (2 * gamma0 * Omega * p0**2)
        k2 = newton(sol2, kg, tol=1e-12, maxiter=10000)
        w2 = ((gamma0**2 * k2**2 + Omega**2) * p0**2 + np.sqrt((-(gamma0**2) * k2**2 + Omega**2) ** 2 * p0**4 - 4 * k2**2 * Omega**2 * p0**2 * (-2 * gamma0**2 + p0**2) * Tz)) / (2 * gamma0 * Omega * p0**2)
        return k, w1, k2, w2

    def sol_kp_cold_one_harm(self, kp, n, all=True):
        k = kp
        Omega = self.Omega
        p0 = self.p0
        gamma0 = np.sqrt(1 + p0**2)
        b0 = k * p0 / Omega
        if k == 0 and (n == 1 or n == -1):
            roots = self.sol_kz_cold(0, all=all)
            try:
                pol = [1.0 for r in roots]
            except:
                pol = 1.0
            return pol, roots
        elif k == 0:
            return np.array([0.0]), np.array([0.0])
        else:
            J_n = J(n, b0)
            dJ_n = dJ(n, b0, 1)
            d2J_n = dJ(n, b0, 2)
            dJ_0 = dJ(0, b0, 1)
            d2J_0 = dJ(0, b0, 2)
            An = n**2 * p0**2 / (gamma0**3 * b0**2) * J_n**2
            Bn = -2 * n**2 / (gamma0 * b0) * J_n * dJ_n
            Cn = p0**2 / gamma0**3 * dJ_n**2
            C0 = p0**2 / gamma0**3 * dJ_0**2
            Dn = -2 / gamma0 * dJ_n * (b0 * d2J_n + dJ_n)
            D0 = -2 / gamma0 * dJ_0 * (b0 * d2J_0 + dJ_0)
            En = n * p0**2 / (gamma0**3 * b0) * J_n * dJ_n
            Fn = -n / gamma0 * (1 / b0 * J_n * dJ_n + dJ_n**2 + J_n * d2J_n)

            c0 = (Bn * (C0 + D0 - k**2) * n**3 * Omega**3) / gamma0**3
            c1 = (gamma0**2 * (-(Fn**2) + Bn * (3 * C0 + 3 * D0 + Dn - 3 * k**2) + An * (C0 + D0 - k**2)) * n**2 * Omega**2 + (C0 + D0 - k**2) * n**4 * Omega**4) / gamma0**4
            c2 = (gamma0**2 * (-2 * Fn * (En + Fn) + Bn * (3 * C0 + Cn + 3 * D0 + 2 * Dn - 3 * k**2) + An * (2 * C0 + 2 * D0 + Dn - 2 * k**2)) * n * Omega + (Bn + 4 * C0 + 4 * D0 + Dn - 4 * k**2) * n**3 * Omega**3) / gamma0**3
            c3 = (
                An * C0
                + Bn * C0
                + An * Cn
                + Bn * Cn
                + An * D0
                + +Bn * D0
                + An * Dn
                + Bn * Dn
                - En**2
                - 2 * En * Fn
                - Fn**2
                - An * k**2
                - Bn * k**2
                + (An * n**2 * Omega**2) / gamma0**2
                + (3 * Bn * n**2 * Omega**2) / gamma0**2
                + (6 * C0 * n**2 * Omega**2) / gamma0**2
                + (Cn * n**2 * Omega**2) / gamma0**2
                + (6 * D0 * n**2 * Omega**2) / gamma0**2
                + (3 * Dn * n**2 * Omega**2) / gamma0**2
                - (6 * k**2 * n**2 * Omega**2) / gamma0**2
                + (n**4 * Omega**4) / gamma0**4
            )
            c4 = (2 * An * n * Omega) / gamma0 + (3 * Bn * n * Omega) / gamma0 + (2 * Cn * n * Omega) / gamma0 + (4 * C0 * n * Omega) / gamma0 + (3 * Dn * n * Omega) / gamma0 + (4 * D0 * n * Omega) / gamma0 - (4 * k**2 * n * Omega) / gamma0 + (4 * n**3 * Omega**3) / gamma0**3
            c5 = An + Bn + Cn + Dn + C0 + D0 - k**2 + (6 * n**2 * Omega**2) / gamma0**2
            c6 = (4 * n * Omega) / gamma0
            c7 = 1
            arr = np.array([c7, c6, c5, c4, c3, c2, c1, c0])
            roots = np.roots(arr)
            ind = np.where(roots != 0)
            roots = roots[ind]
            pol = [np.abs((r**2 * (r + n * Omega / gamma0) ** 2 + r**2 * An + r * Bn * (r + n * Omega / gamma0)) / (r**2 * En + r * Fn * (r + n * Omega / gamma0))) for r in roots]
            if all == False:
                id = np.argsort((np.imag(roots)))[-1]
                return pol[id], roots[id]
            else:
                return pol, roots

    def sol_kp_cold_one_harm_cyclotron_norm(self, kp, n, all=True):
        k = kp
        omegap = 1 / self.Omega
        p0 = self.p0
        gamma0 = np.sqrt(1 + p0**2)
        Omega = 1
        b0 = k * p0
        if k == 0:
            roots = self.sol_kz_cold(0, all=all) * omegap
            try:
                pol = [1.0 for r in roots]
            except:
                pol = 1.0
            return pol, roots
        else:
            J_n = J(n, b0)
            dJ_n = dJ(n, b0, 1)
            d2J_n = dJ(n, b0, 2)
            dJ_0 = dJ(0, b0, 1)
            d2J_0 = dJ(0, b0, 2)
            An = omegap**2 * n**2 * p0**2 / (gamma0**3 * b0**2) * J_n**2
            Bn = -2 * omegap**2 * n**2 / (gamma0 * b0) * J_n * dJ_n
            Cn = omegap**2 * p0**2 / gamma0**3 * dJ_n**2
            Dn = -2 * omegap**2 / gamma0 * dJ_n * (b0 * d2J_n + dJ_n)
            En = n * omegap**2 * p0**2 / (gamma0**3 * b0) * J_n * dJ_n
            Fn = -n * omegap**2 / gamma0 * (1 / b0 * J_n * dJ_n + dJ_n**2 + J_n * d2J_n)
            C0 = p0**2 * omegap**2 / gamma0**3 * dJ_0**2
            D0 = -2 * omegap**2 / gamma0 * dJ_0 * (b0 * d2J_0 + dJ_0)
            # c0 = -((Bn * k**2 * n**3) / gamma0**3)
            # c1 = (Bn * Dn * n**2) / gamma0**2 - (Fn**2 * n**2) / gamma0**2 - (An * k**2 * n**2) / gamma0**2 - (3 * Bn * k**2 * n**2) / gamma0**2 - (k**2 * n**4) / gamma0**4
            # c2 = (Bn * Cn * n) / gamma0 + (An * Dn * n) / gamma0 + (2 * Bn * Dn * n) / gamma0 - (2 * En * Fn * n) / gamma0 - (2 * Fn**2 * n) / gamma0 - (2 * An * k**2 * n) / gamma0 - (3 * Bn * k**2 * n) / gamma0 + (Bn * n**3) / gamma0**3 + (Dn * n**3) / gamma0**3 - (4 * k**2 * n**3) / gamma0**3
            # c3 = An * Cn + Bn * Cn + An * Dn + Bn * Dn - En**2 - 2 * En * Fn - Fn**2 - An * k**2 - Bn * k**2 + (An * n**2) / gamma0**2 + (3 * Bn * n**2) / gamma0**2 + (Cn * n**2) / gamma0**2 + (3 * Dn * n**2) / gamma0**2 - (6 * k**2 * n**2) / gamma0**2 + (n**4) / gamma0**4
            # c4 = (2 * An * n) / gamma0 + (3 * Bn * n) / gamma0 + (2 * Cn * n) / gamma0 + (3 * Dn * n) / gamma0 - (4 * k**2 * n) / gamma0 + (4 * n**3) / gamma0**3
            # c5 = An + Bn + Cn + Dn - k**2 + (6 * n**2) / gamma0**2
            # c6 = (4 * n) / gamma0
            # c7 = 1

            c0 = (Bn * (C0 + D0 - k**2) * n**3 * Omega**3) / gamma0**3
            c1 = (gamma0**2 * (-(Fn**2) + Bn * (3 * C0 + 3 * D0 + Dn - 3 * k**2) + An * (C0 + D0 - k**2)) * n**2 * Omega**2 + (C0 + D0 - k**2) * n**4 * Omega**4) / gamma0**4
            c2 = (gamma0**2 * (-2 * Fn * (En + Fn) + Bn * (3 * C0 + Cn + 3 * D0 + 2 * Dn - 3 * k**2) + An * (2 * C0 + 2 * D0 + Dn - 2 * k**2)) * n * Omega + (Bn + 4 * C0 + 4 * D0 + Dn - 4 * k**2) * n**3 * Omega**3) / gamma0**3
            c3 = (
                An * C0
                + Bn * C0
                + An * Cn
                + Bn * Cn
                + An * D0
                + +Bn * D0
                + An * Dn
                + Bn * Dn
                - En**2
                - 2 * En * Fn
                - Fn**2
                - An * k**2
                - Bn * k**2
                + (An * n**2 * Omega**2) / gamma0**2
                + (3 * Bn * n**2 * Omega**2) / gamma0**2
                + (6 * C0 * n**2 * Omega**2) / gamma0**2
                + (Cn * n**2 * Omega**2) / gamma0**2
                + (6 * D0 * n**2 * Omega**2) / gamma0**2
                + (3 * Dn * n**2 * Omega**2) / gamma0**2
                - (6 * k**2 * n**2 * Omega**2) / gamma0**2
                + (n**4 * Omega**4) / gamma0**4
            )
            c4 = (2 * An * n * Omega) / gamma0 + (3 * Bn * n * Omega) / gamma0 + (2 * Cn * n * Omega) / gamma0 + (4 * C0 * n * Omega) / gamma0 + (3 * Dn * n * Omega) / gamma0 + (4 * D0 * n * Omega) / gamma0 - (4 * k**2 * n * Omega) / gamma0 + (4 * n**3 * Omega**3) / gamma0**3
            c5 = An + Bn + Cn + Dn + C0 + D0 - k**2 + (6 * n**2 * Omega**2) / gamma0**2
            c6 = (4 * n * Omega) / gamma0
            c7 = 1
            arr = np.array([c7, c6, c5, c4, c3, c2, c1, c0])
            roots = np.roots(arr)
            ind = np.where(roots != 0)
            roots = roots[ind]
            pol = [np.abs((r**2 * (r + n / gamma0) ** 2 + r**2 * An + r * Bn * (r + n / gamma0)) / (r**2 * En + r * Fn * (r + n / gamma0))) for r in roots]
            if all == False:
                id = np.argsort((np.imag(roots)))[-1]
                return pol[id], roots[id]
            else:
                return pol, roots

    def disp_kp_cold(self, w, kp, n, symmetric=True):
        if symmetric == True:
            n = np.arange(-n, n + 1)
        else:
            n = np.array(n)
        k = kp
        Omega = self.Omega
        p0 = self.p0
        gamma0 = np.sqrt(1 + p0**2)
        Omrel = Omega / gamma0
        b0 = k * p0 / Omega
        lenn = n.shape[0]
        An = np.zeros((lenn,))
        Bn = np.zeros((lenn,))
        Cn = np.zeros((lenn,))
        Dn = np.zeros((lenn,))
        En = np.zeros((lenn,))
        Fn = np.zeros((lenn,))
        if k == 0:
            for i in range(lenn):
                J_n = J(n[i], b0)
                dJ_n = dJ(n[i], b0, 1)
                d2J_n = dJ(n[i], b0, 2)
                Cn[i] = p0**2 / gamma0**3 * dJ_n**2
                Dn[i] = -2 / gamma0 * dJ_n * (b0 * d2J_n + dJ_n)
                if n[i] in [-1, 1]:
                    An[i] = n[i] ** 2 * p0**2 / gamma0**3 * 1 / 4
                    Bn[i] = -2 * n[i] ** 2 / gamma0 * 1 / 4
                    En[i] = n[i] * p0**2 / gamma0**3 * 1 / 4
                    Fn[i] = -n[i] / gamma0 * (1 / 2)
        else:
            for i in range(lenn):
                J_n = J(n[i], b0)
                dJ_n = dJ(n[i], b0, 1)
                d2J_n = dJ(n[i], b0, 2)
                Cn[i] = p0**2 / gamma0**3 * dJ_n**2
                Dn[i] = -2 / gamma0 * dJ_n * (b0 * d2J_n + dJ_n)
                An[i] = n[i] ** 2 * p0**2 / gamma0**3 * J_n**2 / b0**2
                Bn[i] = -2 * n[i] ** 2 / gamma0 * (J_n * dJ_n / b0)
                En[i] = n[i] * p0**2 / gamma0**3 * (J_n * dJ_n / b0)
                Fn[i] = -n[i] / gamma0 * ((J_n * dJ_n / b0) + dJ_n**2 + J_n * d2J_n)
        if w != 0:
            Dxx = w**2 + np.sum(An[:] * w**2 / (w + n[:] * Omrel) ** 2 + Bn[:] * w / (w + n[:] * Omrel))
            Dyy = w**2 - k**2 + np.sum(Cn[:] * w**2 / (w + n[:] * Omrel) ** 2 + Dn[:] * w / (w + n[:] * Omrel))
            Dxy = np.sum(En[:] * w**2 / (w + n[:] * Omrel) ** 2 + Fn[:] * w / (w + n[:] * Omrel))
        else:
            Dxx = 0
            Dyy = -(k**2)
            Dxy = 0
        return Dxx * Dyy - Dxy * Dxy

    def pol_kp_cold(self, w, kp, n, symmetric=True):
        if symmetric == True:
            n = np.arange(-n, n + 1)
        else:
            n = np.array(n)
        k = kp
        Omega = self.Omega
        p0 = self.p0
        gamma0 = np.sqrt(1 + p0**2)
        Omrel = Omega / gamma0
        b0 = k * p0 / Omega
        lenn = n.shape[0]
        An = np.zeros((lenn,))
        Bn = np.zeros((lenn,))
        En = np.zeros((lenn,))
        Fn = np.zeros((lenn,))
        if k == 0:
            for i in range(lenn):
                J_n = J(n[i], b0)
                dJ_n = dJ(n[i], b0, 1)
                d2J_n = dJ(n[i], b0, 2)
                if n[i] in [-1, 1]:
                    An[i] = n[i] ** 2 * p0**2 / gamma0**3 * 1 / 4
                    Bn[i] = -2 * n[i] ** 2 / gamma0 * 1 / 4
                    En[i] = n[i] * p0**2 / gamma0**3 * 1 / 4
                    Fn[i] = -n[i] / gamma0 * (1 / 2)
        else:
            for i in range(lenn):
                J_n = J(n[i], b0)
                dJ_n = dJ(n[i], b0, 1)
                d2J_n = dJ(n[i], b0, 2)
                An[i] = n[i] ** 2 * p0**2 / gamma0**3 * J_n**2 / b0**2
                Bn[i] = -2 * n[i] ** 2 / gamma0 * (J_n * dJ_n / b0)
                En[i] = n[i] * p0**2 / gamma0**3 * (J_n * dJ_n / b0)
                Fn[i] = -n[i] / gamma0 * ((J_n * dJ_n / b0) + dJ_n**2 + J_n * d2J_n)
        Dxx = w**2 + np.sum(An[:] * w**2 / (w + n[:] * Omrel) ** 2 + Bn[:] * w / (w + n[:] * Omrel))
        Dxy = np.sum(En[:] * w**2 / (w + n[:] * Omrel) ** 2 + Fn[:] * w / (w + n[:] * Omrel))

        return np.abs(Dxx / Dxy)

    def coef_kp_cold(self, w, kp, n, symmetric=True):
        if symmetric == True:
            n = np.arange(-n, n + 1)
        else:
            n = np.array(n)
        k = kp
        Omega = self.Omega
        p0 = self.p0
        gamma0 = np.sqrt(1 + p0**2)
        b0 = k * p0 / Omega
        Omrel = Omega / gamma0
        lenn = n.shape[0]
        An = np.zeros((lenn,))
        Bn = np.zeros((lenn,))
        Cn = np.zeros((lenn,))
        Dn = np.zeros((lenn,))
        En = np.zeros((lenn,))
        Fn = np.zeros((lenn,))
        Ann = np.zeros((lenn,), dtype=np.complex128)
        Bnn = np.zeros((lenn,), dtype=np.complex128)
        Cnn = np.zeros((lenn,), dtype=np.complex128)
        Dnn = np.zeros((lenn,), dtype=np.complex128)
        Enn = np.zeros((lenn,), dtype=np.complex128)
        Fnn = np.zeros((lenn,), dtype=np.complex128)
        if k == 0:
            for i in range(lenn):
                J_n = J(n[i], b0)
                dJ_n = dJ(n[i], b0, 1)
                d2J_n = dJ(n[i], b0, 2)
                Cn[i] = p0**2 / gamma0**3 * dJ_n**2
                Dn[i] = -2 / gamma0 * dJ_n * (b0 * d2J_n + dJ_n)
                if n[i] in [-1, 1]:
                    An[i] = n[i] ** 2 * p0**2 / gamma0**3 * 1 / 4
                    Bn[i] = -2 * n[i] ** 2 / gamma0 * 1 / 4
                    En[i] = n[i] * p0**2 / gamma0**3 * 1 / 4
                    Fn[i] = -n[i] / gamma0 * (1 / 2)
        else:
            for i in range(lenn):
                J_n = J(n[i], b0)
                dJ_n = dJ(n[i], b0, 1)
                d2J_n = dJ(n[i], b0, 2)
                Cn[i] = p0**2 / gamma0**3 * dJ_n**2
                Dn[i] = -2 / gamma0 * dJ_n * (b0 * d2J_n + dJ_n)
                An[i] = n[i] ** 2 * p0**2 / gamma0**3 * J_n**2 / b0**2
                Bn[i] = -2 * n[i] ** 2 / gamma0 * (J_n * dJ_n / b0)
                En[i] = n[i] * p0**2 / gamma0**3 * (J_n * dJ_n / b0)
                Fn[i] = -n[i] / gamma0 * ((J_n * dJ_n / b0) + dJ_n**2 + J_n * d2J_n)
        if w != 0:
            Ann[:] = An[:] * w**2 / (w + n[:] * Omrel) ** 2
            Cnn[:] = Cn[:] * w**2 / (w + n[:] * Omrel) ** 2
            Enn[:] = En[:] * w**2 / (w + n[:] * Omrel) ** 2
            Bnn[:] = Bn[:] * w / (w + n[:] * Omrel)
            Dnn[:] = Dn[:] * w / (w + n[:] * Omrel)
            Fnn[:] = Fn[:] * w / (w + n[:] * Omrel)
        # return An, Bn, Cn, Dn, En, Fn
        return Ann, Bnn, Cnn, Dnn, Enn, Fnn

    def sol_kp_cold(self, wg, kp, n=3, symmetric=True, coef=False):
        try:
            w0 = newton(
                self.disp_kp_cold,
                np.complex128(wg),
                tol=1e-12,
                args=(
                    kp,
                    n,
                    symmetric,
                ),
                maxiter=10000,
            )
        except:
            w0 = 0.0
        if np.abs(self.disp_kp_cold(w0, kp, n, symmetric)) < 1e-8:
            coeff = self.coef_kp_cold(w0, kp, n, symmetric)
            if np.imag(w0) > 1e-10:
                pol = self.pol_kp_cold(w0, kp, n, symmetric)
            else:
                pol = 1
            if coef:
                return coeff, pol, w0
            else:
                return pol, w0
        else:
            if coef:
                return 0.0, 0.0, 0.0
            else:
                return 0.0, 0.0

    def disp_kp_cold_Nr(self, w, kp, Nr, n, symmetric=True):
        if symmetric == True:
            n = np.arange(-n, n + 1)
        else:
            n = np.array(n)
        k = kp
        Omega = self.Omega
        p0 = self.p0
        gamma0 = np.sqrt(1 + p0**2)
        Omrel = Omega / gamma0
        b0 = k * p0 / Omega
        lenn = n.shape[0]
        Nrfac = (Nr - 1) / (2 * Nr)
        An = np.zeros((lenn,))
        Bn = np.zeros((lenn,))
        Cn = np.zeros((lenn,))
        Dn = np.zeros((lenn,))
        En = np.zeros((lenn,))
        Fn = np.zeros((lenn,))
        if k == 0:
            for i in range(lenn):
                J_n = J(n[i], b0)
                dJ_n = dJ(n[i], b0, 1)
                d2J_n = dJ(n[i], b0, 2)
                Cn[i] = p0**2 / gamma0**3 * dJ_n**2
                Dn[i] = -2 / gamma0 * dJ_n * (b0 * d2J_n + dJ_n)
                if n[i] in [-1, 1]:
                    An[i] = n[i] ** 2 * p0**2 / gamma0**3 * 1 / 4
                    Bn[i] = -2 * n[i] ** 2 / gamma0 * 1 / 4
                    En[i] = n[i] * p0**2 / gamma0**3 * 1 / 4
                    Fn[i] = -n[i] / gamma0 * (1 / 2)
        else:
            for i in range(lenn):
                J_n = J(n[i], b0)
                dJ_n = dJ(n[i], b0, 1)
                d2J_n = dJ(n[i], b0, 2)
                Cn[i] = p0**2 / gamma0**3 * dJ_n**2
                Dn[i] = -2 / gamma0 * dJ_n * (b0 * d2J_n + dJ_n)
                An[i] = n[i] ** 2 * p0**2 / gamma0**3 * J_n**2 / b0**2
                Bn[i] = -2 * n[i] ** 2 / gamma0 * (J_n * dJ_n / b0)
                En[i] = n[i] * p0**2 / gamma0**3 * (J_n * dJ_n / b0)
                Fn[i] = -n[i] / gamma0 * ((J_n * dJ_n / b0) + dJ_n**2 + J_n * d2J_n)
        if w != 0:
            Dxx = w**2 - Nrfac * (w / (w + Omega) + w / (w - Omega)) + (1 / Nr) * np.sum(An[:] * w**2 / (w + n[:] * Omrel) ** 2 + Bn[:] * w / (w + n[:] * Omrel))
            Dyy = w**2 - k**2 - Nrfac * (w / (w + Omega) + w / (w - Omega)) + (1 / Nr) * np.sum(Cn[:] * w**2 / (w + n[:] * Omrel) ** 2 + Dn[:] * w / (w + n[:] * Omrel))
            Dxy = -Nrfac * (w / (w + Omega) - w / (w - Omega)) + (1 / Nr) * np.sum(En[:] * w**2 / (w + n[:] * Omrel) ** 2 + Fn[:] * w / (w + n[:] * Omrel))
        else:
            Dxx = 0
            Dyy = -(k**2)
            Dxy = 0
        return Dxx * Dyy - Dxy * Dxy

    def pol_kp_cold_Nr(self, w, kp, Nr, n, symmetric=True):
        if symmetric == True:
            n = np.arange(-n, n + 1)
        else:
            n = np.array(n)
        k = kp
        Omega = self.Omega
        p0 = self.p0
        Nrfac = (Nr - 1) / (2 * Nr)
        gamma0 = np.sqrt(1 + p0**2)
        Omrel = Omega / gamma0
        b0 = k * p0 / Omega
        lenn = n.shape[0]
        An = np.zeros((lenn,))
        Bn = np.zeros((lenn,))
        En = np.zeros((lenn,))
        Fn = np.zeros((lenn,))
        if k == 0:
            for i in range(lenn):
                J_n = J(n[i], b0)
                dJ_n = dJ(n[i], b0, 1)
                d2J_n = dJ(n[i], b0, 2)
                if n[i] in [-1, 1]:
                    An[i] = n[i] ** 2 * p0**2 / gamma0**3 * 1 / 4
                    Bn[i] = -2 * n[i] ** 2 / gamma0 * 1 / 4
                    En[i] = n[i] * p0**2 / gamma0**3 * 1 / 4
                    Fn[i] = -n[i] / gamma0 * (1 / 2)
        else:
            for i in range(lenn):
                J_n = J(n[i], b0)
                dJ_n = dJ(n[i], b0, 1)
                d2J_n = dJ(n[i], b0, 2)
                An[i] = n[i] ** 2 * p0**2 / gamma0**3 * J_n**2 / b0**2
                Bn[i] = -2 * n[i] ** 2 / gamma0 * (J_n * dJ_n / b0)
                En[i] = n[i] * p0**2 / gamma0**3 * (J_n * dJ_n / b0)
                Fn[i] = -n[i] / gamma0 * ((J_n * dJ_n / b0) + dJ_n**2 + J_n * d2J_n)
        Dxx = w**2 - Nrfac * (w / (w + Omega) + w / (w - Omega)) + (1 / Nr) * np.sum(An[:] * w**2 / (w + n[:] * Omrel) ** 2 + Bn[:] * w / (w + n[:] * Omrel))
        Dxy = -Nrfac * (w / (w + Omega) - w / (w - Omega)) + (1 / Nr) * np.sum(En[:] * w**2 / (w + n[:] * Omrel) ** 2 + Fn[:] * w / (w + n[:] * Omrel))

        return np.abs(Dxx / Dxy)

    def sol_kp_cold_Nr(self, wg, kp, Nr, n=3, symmetric=True, coef=False):
        try:
            w0 = newton(
                self.disp_kp_cold_Nr,
                np.complex128(wg),
                tol=1e-12,
                args=(
                    kp,
                    Nr,
                    n,
                    symmetric,
                ),
                maxiter=10000,
            )
        except:
            w0 = 0.0
        if np.abs(self.disp_kp_cold_Nr(w0, kp, Nr, n, symmetric)) < 1e-8:
            coeff = self.coef_kp_cold(w0, kp, n, symmetric)
            if np.imag(w0) > 1e-10:
                pol = self.pol_kp_cold_Nr(w0, kp, Nr, n, symmetric)
            else:
                pol = 1
            if coef:
                return coeff, pol, w0
            else:
                return pol, w0
        else:
            if coef:
                return 0.0, 0.0, 0.0
            else:
                return 0.0, 0.0

    def disp_kp_cold_cyc(self, w, kp, n, symmetric=True):
        if symmetric == True:
            n = np.arange(-n, n + 1)
        else:
            n = np.array(n)
        k = kp
        omegap = 1 / self.Omega
        p0 = self.p0
        gamma0 = np.sqrt(1 + p0**2)
        b0 = k * p0
        lenn = n.shape[0]
        An = np.zeros((lenn,))
        Bn = np.zeros((lenn,))
        Cn = np.zeros((lenn,))
        Dn = np.zeros((lenn,))
        En = np.zeros((lenn,))
        Fn = np.zeros((lenn,))
        if k == 0:
            for i in range(lenn):
                J_n = J(n[i], b0)
                dJ_n = dJ(n[i], b0, 1)
                d2J_n = dJ(n[i], b0, 2)
                Cn[i] = omegap**2 * p0**2 / gamma0**3 * dJ_n**2
                Dn[i] = -2 * omegap**2 / gamma0 * dJ_n * (b0 * d2J_n + dJ_n)
                if n[i] in [-1, 1]:
                    An[i] = omegap**2 * n[i] ** 2 * p0**2 / gamma0**3 * 1 / 4
                    Bn[i] = -2 * omegap**2 * n[i] ** 2 / gamma0 * 1 / 4
                    En[i] = omegap**2 * n[i] * p0**2 / gamma0**3 * 1 / 4
                    Fn[i] = -n[i] * omegap**2 / gamma0 * (1 / 2)
        else:
            for i in range(lenn):
                J_n = J(n[i], b0)
                dJ_n = dJ(n[i], b0, 1)
                d2J_n = dJ(n[i], b0, 2)
                Cn[i] = omegap**2 * p0**2 / gamma0**3 * dJ_n**2
                Dn[i] = -2 * omegap**2 / gamma0 * dJ_n * (b0 * d2J_n + dJ_n)
                An[i] = omegap**2 * n[i] ** 2 * p0**2 / gamma0**3 * J_n**2 / b0**2
                Bn[i] = -2 * omegap**2 * n[i] ** 2 / gamma0 * (J_n * dJ_n / b0)
                En[i] = omegap**2 * n[i] * p0**2 / gamma0**3 * (J_n * dJ_n / b0)
                Fn[i] = -n[i] * omegap**2 / gamma0 * ((J_n * dJ_n / b0) + dJ_n**2 + J_n * d2J_n)
        if w != 0:
            Dxx = w**2 + np.sum(An[:] * w**2 / (w + n[:] / gamma0) ** 2 + Bn[:] * w / (w + n[:] / gamma0))
            Dyy = w**2 - k**2 + np.sum(Cn[:] * w**2 / (w + n[:] / gamma0) ** 2 + Dn[:] * w / (w + n[:] / gamma0))
            Dxy = np.sum(En[:] * w**2 / (w + n[:] / gamma0) ** 2 + Fn[:] * w / (w + n[:] / gamma0))
        else:
            Dxx = 0
            Dyy = -(k**2)
            Dxy = 0
        return Dxx * Dyy - Dxy * Dxy

    def pol_kp_cold_cyc(self, w, kp, n, symmetric=True):
        if symmetric == True:
            n = np.arange(-n, n + 1)
        else:
            n = np.array(n)
        k = kp
        omegap = 1 / self.Omega
        p0 = self.p0
        gamma0 = np.sqrt(1 + p0**2)
        b0 = k * p0
        lenn = n.shape[0]
        An = np.zeros((lenn,))
        Bn = np.zeros((lenn,))
        En = np.zeros((lenn,))
        Fn = np.zeros((lenn,))
        if k == 0:
            for i in range(lenn):
                J_n = J(n[i], b0)
                dJ_n = dJ(n[i], b0, 1)
                d2J_n = dJ(n[i], b0, 2)
                if n[i] in [-1, 1]:
                    An[i] = omegap**2 * n[i] ** 2 * p0**2 / gamma0**3 * 1 / 4
                    Bn[i] = -2 * omegap**2 * n[i] ** 2 / gamma0 * 1 / 4
                    En[i] = n[i] * omegap**2 * p0**2 / gamma0**3 * 1 / 4
                    Fn[i] = -n[i] * omegap**2 / gamma0 * (1 / 2)
        else:
            for i in range(lenn):
                J_n = J(n[i], b0)
                dJ_n = dJ(n[i], b0, 1)
                d2J_n = dJ(n[i], b0, 2)
                An[i] = omegap**2 * n[i] ** 2 * p0**2 / gamma0**3 * J_n**2 / b0**2
                Bn[i] = -2 * omegap**2 * n[i] ** 2 / gamma0 * (J_n * dJ_n / b0)
                En[i] = n[i] * omegap**2 * p0**2 / gamma0**3 * (J_n * dJ_n / b0)
                Fn[i] = -n[i] * omegap**2 / gamma0 * ((J_n * dJ_n / b0) + dJ_n**2 + J_n * d2J_n)
        Dxx = w**2 + np.sum(An[:] * w**2 / (w + n[:] / gamma0) ** 2 + Bn[:] * w / (w + n[:] / gamma0))
        Dxy = np.sum(En[:] * w**2 / (w + n[:] / gamma0) ** 2 + Fn[:] * w / (w + n[:] / gamma0))

        return np.abs(Dxx / Dxy)

    def coef_kp_cold_cyc(self, w, kp, n, symmetric=True):
        if symmetric == True:
            n = np.arange(-n, n + 1)
        else:
            n = np.array(n)
        k = kp
        omegap = 1 / self.Omega
        p0 = self.p0
        gamma0 = np.sqrt(1 + p0**2)
        b0 = k * p0
        lenn = n.shape[0]
        An = np.zeros((lenn,))
        Bn = np.zeros((lenn,))
        Cn = np.zeros((lenn,))
        Dn = np.zeros((lenn,))
        En = np.zeros((lenn,))
        Fn = np.zeros((lenn,))
        Ann = np.zeros((lenn,), dtype=np.complex128)
        Bnn = np.zeros((lenn,), dtype=np.complex128)
        Cnn = np.zeros((lenn,), dtype=np.complex128)
        Dnn = np.zeros((lenn,), dtype=np.complex128)
        Enn = np.zeros((lenn,), dtype=np.complex128)
        Fnn = np.zeros((lenn,), dtype=np.complex128)
        if k == 0:
            for i in range(lenn):
                J_n = J(n[i], b0)
                dJ_n = dJ(n[i], b0, 1)
                d2J_n = dJ(n[i], b0, 2)
                Cn[i] = p0**2 / gamma0**3 * dJ_n**2
                Dn[i] = -2 / gamma0 * dJ_n * (b0 * d2J_n + dJ_n)
                if n[i] in [-1, 1]:
                    An[i] = n[i] ** 2 * p0**2 / gamma0**3 * 1 / 4
                    Bn[i] = -2 * n[i] ** 2 / gamma0 * 1 / 4
                    En[i] = n[i] * p0**2 / gamma0**3 * 1 / 4
                    Fn[i] = -n[i] / gamma0 * (1 / 2)
        else:
            for i in range(lenn):
                J_n = J(n[i], b0)
                dJ_n = dJ(n[i], b0, 1)
                d2J_n = dJ(n[i], b0, 2)
                Cn[i] = omegap**2 * p0**2 / gamma0**3 * dJ_n**2
                Dn[i] = -2 * omegap**2 / gamma0 * dJ_n * (b0 * d2J_n + dJ_n)
                An[i] = n[i] ** 2 * omegap**2 * p0**2 / gamma0**3 * J_n**2 / b0**2
                Bn[i] = -2 * n[i] ** 2 * omegap**2 / gamma0 * (J_n * dJ_n / b0)
                En[i] = n[i] * p0**2 * omegap**2 / gamma0**3 * (J_n * dJ_n / b0)
                Fn[i] = -n[i] * omegap**2 / gamma0 * ((J_n * dJ_n / b0) + dJ_n**2 + J_n * d2J_n)
        if w != 0:
            Ann[:] = An[:] * w**2 / (w + n[:] / gamma0) ** 2
            Cnn[:] = Cn[:] * w**2 / (w + n[:] / gamma0) ** 2
            Enn[:] = En[:] * w**2 / (w + n[:] / gamma0) ** 2
            Bnn[:] = Bn[:] * w / (w + n[:] / gamma0)
            Dnn[:] = Dn[:] * w / (w + n[:] / gamma0)
            Fnn[:] = Fn[:] * w / (w + n[:] / gamma0)
        # return An, Bn, Cn, Dn, En, Fn
        return Ann, Bnn, Cnn, Dnn, Enn, Fnn

    def sol_kp_cold_cyc(self, wg, kp, n=3, symmetric=True, coef=False):
        try:
            w0 = newton(
                self.disp_kp_cold_cyc,
                np.complex128(wg),
                tol=1e-12,
                args=(
                    kp,
                    n,
                    symmetric,
                ),
                maxiter=10000,
            )
        except:
            w0 = 0.0
        if np.abs(self.disp_kp_cold_cyc(w0, kp, n, symmetric)) < 1e-8:
            coeff = self.coef_kp_cold_cyc(w0, kp, n, symmetric)
            if np.imag(w0) > 1e-10:
                pol = self.pol_kp_cold_cyc(w0, kp, n, symmetric)
            else:
                pol = 1
            if coef:
                return coeff, pol, w0
            else:
                return pol, w0
        else:
            if coef:
                return 0.0, 0.0, 0.0
            else:
                return 0.0, 0.0

    def disp_kp_Tp(self, w, kp, n=1, symmetric=True):
        if symmetric == True:
            nrange = np.arange(-n, n + 1)
        else:
            # "-n" To be consistent with the other functions that had w+nOmega/g argument
            nrange = np.sort(-np.array([n]))
        idx0 = int((nrange.shape[0] - 1) / 2)
        resonance = np.zeros(nrange.shape, dtype=np.int8)
        Tp = self.Tp
        p0 = self.p0
        Omega = self.Omega
        kO = kp / Omega
        normxx = -1 / (self.eta * Tp**2 * kO**2)
        normxy = -1 / (self.eta * Tp**2 * kO)
        normyy = -1 / (self.eta * Tp**2)
        lower_lim = self.lower_lim
        upper_lim = self.upper_lim
        Oo2 = Omega / np.abs(w) ** 2
        wr, wi = np.real(w), np.imag(w)
        Wii_std = Oo2 * wi

        if np.abs(wi) < 1e-8:
            sigma = 1
            flag = False
        elif wi > 0:
            sigma = 0
            flag = True
        else:
            sigma = 2
            flag = True

        test_res = np.real(nrange[:] * Omega / w)
        id = np.where(test_res > 1)
        test_res = -np.ones(test_res.shape)
        test_res[id] = np.real(np.sqrt((nrange[id] * Omega / w) ** 2 - 1))
        for i in id[0]:
            if (test_res[i] < upper_lim) and (test_res[i] > lower_lim):
                resonance[i] = 1

        arg1 = np.where(resonance == 0)[0]
        arg2 = np.where(resonance == 1)[0]

        def gxxn(p, n):
            return normxx * n**2 * (p - p0) * np.exp(-((p - p0) ** 2) / (2 * Tp)) * J(n, kO * p) ** 2

        def gyyn(p, n):
            return normyy * (p - p0) * np.exp(-((p - p0) ** 2) / (2 * Tp)) * dJ(n, kO * p) ** 2 * p**2

        def gxyn(p, n):
            return normxy * n * (p - p0) * np.exp(-((p - p0) ** 2) / (2 * Tp)) * dJ(n, kO * p) * J(n, kO * p) * p

        def resxxn(p, n):
            return gxxn(p, n) * np.sqrt(1 + p**2) / p**2

        def resyyn(p, n):
            return gyyn(p, n) * np.sqrt(1 + p**2) / p**2

        def resxyn(p, n):
            return gxyn(p, n) * np.sqrt(1 + p**2) / p**2

        def int_simple(func, n):
            def intre(p, n):
                Wrr = np.sqrt(1 + p**2) - n * Oo2 * wr
                Wii = n * Wii_std
                den = Wrr**2 + Wii**2
                gg = func(p, n)
                return gg * Wrr / den

            def intim(p, n):
                Wrr = np.sqrt(1 + p**2) - n * Oo2 * wr
                Wii = n * Wii_std
                den = Wrr**2 + Wii**2
                gg = func(p, n)
                return -gg * Wii / den

            intr = quadsci(intre, lower_lim, upper_lim, args=(n,), limit=150)[0]
            inti = quadsci(intim, lower_lim, upper_lim, args=(n,), limit=150)[0]

            return intr + 1j * inti

        def int_pv(func, n):
            ppole = np.sqrt((n * Omega / w) ** 2 - 1)
            re_pole = np.real(ppole)
            Ok = n * Omega / w

            def int00(p, n):
                gg = func(p, n)
                den = np.sqrt(1 + p**2) - Ok
                return gg / den

            intr1 = quadsci(int00, lower_lim, re_pole - 1e-8, args=(n,), limit=150)[0]
            intr2 = quadsci(int00, re_pole + 1e-8, upper_lim, args=(n,), limit=150)[0]
            return intr1 + intr2

        def integrals_calc(func_int, func_pol):
            integrals = np.zeros(nrange.shape, dtype=np.complex128)
            for j in arg2:
                if flag:
                    ppole = np.sqrt((nrange[j] * Omega / w) ** 2 - 1)
                    residue = -1j * np.pi * sigma * func_pol(ppole, nrange[j])
                    sol = int_simple(func_int, nrange[j])
                    integrals[j] = sol + residue
                else:
                    ppole = np.sqrt((nrange[j] * Omega / w) ** 2 - 1)
                    residue = -1j * np.pi * sigma * func_pol(ppole, nrange[j])
                    sol = int_pv(func_int, nrange[j])
                    integrals[j] = sol + residue

            for j in arg1:
                sol = int_simple(func_int, nrange[j])
                integrals[j] = sol

            return np.sum(integrals)

        exx = integrals_calc(gxxn, resxxn)
        eyy = integrals_calc(gyyn, resyyn)
        exy = integrals_calc(gxyn, resxyn)
        Dxx = w**2 + exx
        Dyy = w**2 - kp**2 + eyy
        disp = Dxx * Dyy - exy**2
        return disp

    def pol_kp_Tp(self, w, kp, n=1, symmetric=True):
        if symmetric == True:
            nrange = np.arange(-n, n + 1)
        else:
            # "-n" To be consistent with the other functions that had w+nOmega/g argument
            nrange = np.sort(-np.array([n]))
        idx0 = int((nrange.shape[0] - 1) / 2)
        resonance = np.zeros(nrange.shape, dtype=np.int8)
        Tp = self.Tp
        p0 = self.p0
        Omega = self.Omega
        kO = kp / Omega
        normxx = -1 / (self.eta * Tp**2 * kO**2)
        normxy = -1 / (self.eta * Tp**2 * kO)
        lower_lim = self.lower_lim
        upper_lim = self.upper_lim
        Oo2 = Omega / np.abs(w) ** 2
        wr, wi = np.real(w), np.imag(w)
        Wii_std = Oo2 * wi

        if np.abs(wi) < 1e-8:
            sigma = 1
            flag = False
        elif wi > 0:
            sigma = 0
            flag = True
        else:
            sigma = 2
            flag = True

        test_res = np.real(nrange[:] * Omega / w)
        id = np.where(test_res > 1)
        test_res = -np.ones(test_res.shape)
        test_res[id] = np.real(np.sqrt((nrange[id] * Omega / w) ** 2 - 1))
        for i in id[0]:
            if (test_res[i] < upper_lim) and (test_res[i] > lower_lim):
                resonance[i] = 1

        arg1 = np.where(resonance == 0)[0]
        arg2 = np.where(resonance == 1)[0]

        def gxxn(p, n):
            return normxx * n**2 * (p - p0) * np.exp(-((p - p0) ** 2) / (2 * Tp)) * J(n, kO * p) ** 2

        def gxyn(p, n):
            return normxy * n * (p - p0) * np.exp(-((p - p0) ** 2) / (2 * Tp)) * dJ(n, kO * p) * J(n, kO * p) * p

        def resxxn(p, n):
            return gxxn(p, n) * np.sqrt(1 + p**2) / p**2

        def resxyn(p, n):
            return gxyn(p, n) * np.sqrt(1 + p**2) / p**2

        def int_simple(func, n):
            def intre(p, n):
                Wrr = np.sqrt(1 + p**2) - n * Oo2 * wr
                Wii = n * Wii_std
                den = Wrr**2 + Wii**2
                gg = func(p, n)
                return gg * Wrr / den

            def intim(p, n):
                Wrr = np.sqrt(1 + p**2) - n * Oo2 * wr
                Wii = n * Wii_std
                den = Wrr**2 + Wii**2
                gg = func(p, n)
                return -gg * Wii / den

            intr = quadsci(intre, lower_lim, upper_lim, args=(n,), limit=150)[0]
            inti = quadsci(intim, lower_lim, upper_lim, args=(n,), limit=150)[0]

            return intr + 1j * inti

        def int_pv(func, n):
            ppole = np.sqrt((n * Omega / w) ** 2 - 1)
            re_pole = np.real(ppole)
            Ok = n * Omega / w

            def int00(p, n):
                gg = func(p, n)
                den = np.sqrt(1 + p**2) - Ok
                return gg / den

            intr1 = quadsci(int00, lower_lim, re_pole - 1e-8, args=(n,), limit=150)[0]
            intr2 = quadsci(int00, re_pole + 1e-8, upper_lim, args=(n,), limit=150)[0]
            return intr1 + intr2

        def integrals_calc(func_int, func_pol):
            integrals = np.zeros(nrange.shape, dtype=np.complex128)
            for j in arg2:
                if flag:
                    ppole = np.sqrt((nrange[j] * Omega / w) ** 2 - 1)
                    residue = -1j * np.pi * sigma * func_pol(ppole, nrange[j])
                    sol = int_simple(func_int, nrange[j])
                    integrals[j] = sol + residue
                else:
                    ppole = np.sqrt((nrange[j] * Omega / w) ** 2 - 1)
                    residue = -1j * np.pi * sigma * func_pol(ppole, nrange[j])
                    sol = int_pv(func_int, nrange[j])
                    integrals[j] = sol + residue

            for j in arg1:
                sol = int_simple(func_int, nrange[j])
                integrals[j] = sol

            return np.sum(integrals)

        exx = integrals_calc(gxxn, resxxn)
        exy = integrals_calc(gxyn, resxyn)
        Dxx = w**2 + exx
        pol = np.abs(Dxx / exy)
        return pol

    def sol_kp_Tp(self, w0g, kp, n=3, symmetric=True):
        """
        w0g = np.complex128(w0g)
        w0 = newton(self.disp_kp_Tp, w0g, args=(k, n, symmetric), tol=1e-12, maxiter=10000)
        if np.abs(self.disp_kp_Tp(w0, k, n)) < 1e-8:
            return w0
        else:
            return 0
        """
        try:
            w0 = newton(
                self.disp_kp_Tp,
                np.complex128(w0g),
                tol=1e-12,
                args=(
                    kp,
                    n,
                    symmetric,
                ),
                maxiter=10000,
            )
        except:
            w0 = 0.0
        if np.abs(self.disp_kp_Tp(w0, kp, n, symmetric)) < 1e-8:
            if np.imag(w0) > 1e-10:
                pol = self.pol_kp_Tp(w0, kp, n, symmetric)
            else:
                pol = 1
            return pol, w0
        else:
            return 0.0, 0.0

    def disp_0_Tp(self, w):
        Tp = self.Tp
        p0 = self.p0
        Omega = self.Omega
        lower_lim = self.lower_lim
        upper_lim = self.upper_lim
        Oo2 = Omega / np.abs(w) ** 2
        wr, wi = np.real(w), np.imag(w)
        Wii_std = Oo2 * wi
        norm = 1 / (2 * self.eta * Tp**2)

        if np.abs(wi) < 1e-8:
            sigma = 1
            flag = False
        elif wi > 0:
            sigma = 0
            flag = True
        else:
            sigma = 2
            flag = True

        test_res = np.real(Omega / w)
        if (test_res > 1) and (test_res < upper_lim) and (test_res > lower_lim):
            resonance = 1
        else:
            resonance = 0

        arg1 = np.where(resonance == 0)[0]
        arg2 = np.where(resonance == 1)[0]

        def g(p):
            return norm * p**2 * (p - p0) * np.exp(-((p - p0) ** 2) / (2 * Tp))

        def resg(p):
            return g(p) * np.sqrt(1 + p**2) / p**2

        def int_simple(func):
            def intre(p):
                Wrr = np.sqrt(1 + p**2) - Oo2 * wr
                Wii = Wii_std
                den = Wrr**2 + Wii**2
                gg = func(p)
                return gg * Wrr / den

            def intim(p):
                Wrr = np.sqrt(1 + p**2) - Oo2 * wr
                Wii = Wii_std
                den = Wrr**2 + Wii**2
                gg = func(p)
                return -gg * Wii / den

            intr = quadsci(intre, lower_lim, upper_lim, limit=150)[0]
            inti = quadsci(intim, lower_lim, upper_lim, limit=150)[0]

            return intr + 1j * inti

        def int_pv(func):
            ppole = np.sqrt((Omega / w) ** 2 - 1)
            re_pole = np.real(ppole)
            Ok = Omega / w

            def int00(p):
                gg = func(p)
                den = np.sqrt(1 + p**2) - Ok
                return gg / den

            intr1 = quadsci(int00, lower_lim, re_pole - 1e-8, limit=150)[0]
            intr2 = quadsci(int00, re_pole + 1e-8, upper_lim, limit=150)[0]
            return intr1 + intr2

        def integrals_calc(func_int, func_pol):
            integrals = 0 + 0j
            if resonance == 1:
                if flag:
                    ppole = np.sqrt((Omega / w) ** 2 - 1)
                    residue = -1j * np.pi * sigma * func_pol(ppole)
                    sol = int_simple(func_int)
                    integrals = sol + residue
                else:
                    ppole = np.sqrt((Omega / w) ** 2 - 1)
                    residue = -1j * np.pi * sigma * func_pol(ppole)
                    sol = int_pv(func_int)
                    integrals = sol + residue

            else:
                sol = int_simple(func_int)
                integrals = sol

            return integrals

        exx = integrals_calc(g, resg)
        disp = w**2 - exx
        return disp

    def sol_0_Tp(self, w0g):
        try:
            w0 = newton(
                self.disp_0_Tp,
                np.complex128(w0g),
                tol=1e-12,
                maxiter=10000,
            )
        except:
            w0 = 0.0
        return w0

    def disp_kz_Tz(self, w, kz):
        if not self.Tz:
            print("Missing value of Tz. Aborting...")
            sys.exit()

        k = kz
        Omega = self.Omega
        p0 = self.p0
        gamma0 = np.sqrt(1 + p0**2)
        Tz = self.Tz
        xi = gamma0 / (k * np.sqrt(2 * Tz)) * (w - Omega / gamma0)
        Zxi = Z(xi)

        X_1 = p0**2 / (Tz * gamma0) - Omega * w * p0**2 / (Tz * k**2 * gamma0**2)
        X_2 = 1 + xi * Zxi
        Y_1 = p0**2 / gamma0**3 - 2 / gamma0
        Y_2 = 1 - Omega / (k * np.sqrt(2 * Tz)) * Zxi

        return w**2 - k**2 + (X_1 * X_2 + Y_1 * Y_2) / 2

    def sol_kz_Tz(self, w0g, kz):
        k = kz
        if type(w0g) is float or type(w0g) is complex:
            w0g = [w0g]
        wg = np.array(w0g, dtype=np.complex128)
        len = wg.shape[0]
        roots = np.zeros(wg.shape, dtype=np.complex128)

        for j in range(len):
            try:
                w0 = newton(self.disp_kz_Tz, wg[j], tol=1e-12, args=(k,), maxiter=10000)
            except:
                w0 = np.complex128(0)
            if np.abs(self.disp_kz_Tz(w0, k)) < 1e-8:
                roots[j] = w0

        return roots

    def max_gr_kz_Tz(self):
        Tz = self.Tz
        Omega = self.Omega
        p0 = self.p0
        gamma0 = np.sqrt(1 + p0**2)
        maxk = 7 / 10 * p0 / np.sqrt(gamma0 * Tz)
        mink = self.kz_instability_threshold()
        if mink > maxk:
            # print("There is no unstable solution probably. Try to check it manually.")
            return 0, 0
        kzt = (maxk + mink) / 2
        w0g = np.array([self.sol_kz_cold(kzt, all=False)], dtype=np.complex128)

        def fmin(kz, wg):
            return -np.imag(self.sol_kz_Tz(wg, kz))

        kmax = fminbound(fmin, mink, maxk, args=(w0g,))
        gr = self.sol_kz_Tz(w0g, kmax)
        return kmax, gr

    def max_gr_kp_1st_harm(self):
        Omega = self.Omega
        p0 = self.p0
        gamma0 = np.sqrt(1 + p0**2)
        maxk = Omega / gamma0
        mink = 0

        def fmin(k):
            # pol, val = self.sol_kp_cold_one_harm(k, -1)
            # id = np.where(np.array(pol) > 1)
            # val = -np.max(np.imag(val[id]))
            # return val
            pol, val = self.sol_kp_cold_one_harm(k, -1, all=False)
            return -np.imag(val)

        # result = minimize_scalar(fmin, bounds=(mink, maxk), method="bounded")
        result = minimize_scalar(fmin, bracket=(mink, maxk / 20), method="brent")
        # result = bracket(fmin, xa=mink, xb=maxk / 20)
        return result.x, -result.fun

    def disp_kz_TzTp(self, w, kz):
        if not self.Tz or not self.Tp:
            print("Missing value of T. Aborting...")
            sys.exit()

        epsabs = 1e-12
        epsrel = 1e-12
        k = kz
        Omega = self.Omega
        p0 = self.p0
        Tz = self.Tz
        Tp = self.Tp
        c0 = 1 / (2 * Tp)

        norm = 1 / (Tp * np.exp(-(p0**2) / (2 * Tp)) + np.sqrt(np.pi * Tp / 2) * p0 * (1 + erf(p0 / np.sqrt(2 * Tp))))

        def fp(p):
            return norm * np.exp(-c0 * (p - p0) ** 2)

        def r1(p):
            return np.log10(fp(p)) + 10

        p0guess = p0 + np.sqrt(Tp)
        l = fsolve(r1, np.array([p0guess]))[0]
        l = np.abs(p0 - l)
        lim = [np.max([0, p0 - l]), p0 + l]
        lower_lim = np.max([lim[0], 0.0])
        upper_lim = lim[1]

        A1 = 1 / Tz
        A2 = Omega * w / (Tz * k**2)
        A3 = w / (np.sqrt(2 * Tz) * k)
        A4 = Omega / (np.sqrt(2 * Tz) * k)

        def integrand1(p):
            gamma = np.sqrt(1 + p**2)
            xi = gamma * A3 - A4
            return (A1 * p**3 / gamma - A2 * p**3 / gamma**2) * (1 + xi * Z(xi)) * fp(p)

        integration_1 = quad(integrand1, lower_lim, upper_lim, epsabs=epsabs, epsrel=epsrel)[0]

        def integrand2(p):
            gamma = np.sqrt(1 + p**2)
            xi = gamma * A3 - A4
            return (p**3 / gamma**3 - 2 * p / gamma) * (1 - A4 * Z(xi)) * fp(p)

        integration_2 = quad(integrand2, lower_lim, upper_lim, epsabs=epsabs, epsrel=epsrel)[0]

        return w**2 - k**2 + (integration_1 + integration_2) / 2

        """
        def I1(w):
            def integrand(p):
                gamma = np.sqrt(1 + p ** 2)
                xi = np.sqrt(1 + p ** 2) * One_sqrt2Tk * w - Omega_sqrt2Tk
                return p ** 2 / gamma * dfp(p) * (1 + xi * Z(xi))

            integration = quad(integrand, lower_lim, upper_lim, epsabs=epsabs, epsrel=epsrel)[0]
            return integration

        def I2(w):
            def integrand(p):
                gamma = np.sqrt(1 + p ** 2)
                xi = np.sqrt(1 + p ** 2) * One_sqrt2Tk * w - Omega_sqrt2Tk
                return p ** 3 / gamma * fp(p) * (1 + xi * Z(xi))

            integration = quad(integrand, lower_lim, upper_lim, epsabs=epsabs, epsrel=epsrel)[0]
            return integration

        def I3(w):
            def integrand(p):
                xi = np.sqrt(1 + p ** 2) * One_sqrt2Tk * w - Omega_sqrt2Tk
                return p ** 2 * dfp(p) * Z(xi)

            integration = quad(integrand, lower_lim, upper_lim, epsabs=epsabs, epsrel=epsrel)[0]
            return integration

        def Theta(w):
            return I1(w) + I2(w) / Tz - w * I3(w) * One_sqrt2Tk

        return w ** 2 - k ** 2 + Theta(w) / 2
        """

    def sol_kz_TzTp(self, w0g, kz):
        k = kz
        if type(w0g) is float or type(w0g) is complex or type(w0g) is int:
            w0g = [w0g]
        wg = np.array(w0g, dtype=np.complex128)
        len = wg.shape[0]
        roots = np.zeros(wg.shape, dtype=np.complex128)
        for j in range(len):
            try:
                w0 = newton(self.disp_kz_TzTp, wg[j], tol=1e-12, args=(k,), maxiter=10000)
            except:
                w0 = np.complex128(0)
            if np.abs(self.disp_kz_TzTp(w0, k)) < 1e-7:
                roots[j] = w0

        return roots

    def disp_kzkp_Tz(self, kz, kp, w0g):
        if not self.Tz:
            print("Missing value of Tz. Aborting...")
            sys.exit()
        if type(w0g) is float or type(w0g) is complex:
            w0g = [w0g]
        wg = np.array(w0g, dtype=np.complex128)
        len = wg.shape[0]
        roots = np.zeros(wg.shape, dtype=np.complex128)

        Omega = self.Omega
        p0 = self.p0
        Tz = self.Tz
        One_sqrt2Tk = 1 / (np.sqrt(2 * Tz) * kz)
        Omega_sqrt2Tk = Omega / (np.sqrt(2 * Tz) * kz)
        gamma0 = np.sqrt(1 + p0**2)
        Gamma_sqrt2Tk = gamma0 / (np.sqrt(2 * Tz) * kz)

        b0 = kp * p0 / Omega
        J1b = J1(b0)
        J1p = dJ(1, b0)
        J1p2 = dJ(1, b0, 2)

        Ixx1 = 2 * J1b * J1p * Omega_sqrt2Tk / (p0 * kp)
        Ixx2 = 2 * Omega_sqrt2Tk**2 * J1b**2 / (gamma0 * kp**2)
        Ixx3 = Omega**2 * J1b**2 / (kp**2 * gamma0**3)
        Ixx4 = 2 * Omega * J1b * J1p / (kp * p0 * gamma0)
        Ixx5 = Omega**2 * J1b**2 / (np.sqrt(2 * Tz) * kz * kp**2 * gamma0**2)
        Ixx6 = Omega**2 * J1b**2 / (Tz * kp**2 * gamma0)

        Iyy1 = np.sqrt(2 / Tz) * J1p**2 / kz
        Iyy2 = np.sqrt(2 / Tz) * J1p * J1p2 * kp * p0 / (kz * Omega)
        Iyy3 = J1p**2 * p0**2 / (Tz * kz**2 * gamma0)
        Iyy4 = p0**2 * J1p**2 / gamma0**3
        Iyy5 = 2 * J1p**2 / gamma0
        Iyy6 = 2 * J1p * J1p2 * kp * p0 / (Omega * gamma0)
        Iyy7 = J1p**2 * p0**2 / (np.sqrt(2 * Tz) * kz * gamma0**2)
        Iyy8 = p0**2 * J1p**2 / (Tz * gamma0)

        Ixy1 = J1b * J1p * Omega_sqrt2Tk / (kp * p0)
        Ixy2 = J1p**2 * One_sqrt2Tk
        Ixy3 = J1b * J1p2 * One_sqrt2Tk
        Ixy4 = J1b * J1p * Omega * p0 / (Tz * kz**2 * kp * gamma0)
        Ixy5 = Omega * J1b * J1p / (kp * p0 * gamma0)
        Ixy6 = Omega * p0 * J1b * J1p / (kp * gamma0**3)
        Ixy7 = J1p**2 / gamma0
        Ixy8 = J1b * J1p2 / gamma0
        Ixy9 = J1b * J1p * Omega_sqrt2Tk * p0 / (kp * gamma0**2)
        Ixy0 = Omega * p0 * J1b * J1p / (kp * gamma0 * Tz)

        def disp(w):
            xi = Gamma_sqrt2Tk * w - Omega_sqrt2Tk
            Zxi = Z(xi)
            Zp1 = 1 + xi * Zxi
            Zp2 = Zxi - 2 * xi - 2 * xi**2 * Zxi
            I1 = Ixx1 * w * Zxi - Ixx2 * w**2 * Zp1
            I2 = (Ixx3 - Ixx4) * Zp1 - Ixx5 * w * Zp2
            I3 = Ixx6 * Zp1
            I4 = (Iyy1 + Iyy2) * w * Zxi - Iyy3 * w**2 * Zp1
            I5 = (Iyy4 - Iyy5 - Iyy6) * Zp1 - Iyy7 * w * Zp2
            I6 = Iyy8 * Zp1
            I7 = (Ixy1 + Ixy2 + Ixy3) * w * Zxi - Ixy4 * w**2 * Zp1
            I8 = (Ixy5 - Ixy6 + Ixy7 + Ixy8) * Zp1 + Ixy9 * w * Zp2
            I9 = Ixy0 * Zp1

            Dxx = w**2 - kz**2 + I1 + I2 + I3
            Dyy = w**2 - kp**2 - kz**2 + I4 + I5 + I6
            DxyDyx = (I7 - I8 + I9) ** 2
            return Dxx * Dyy - DxyDyx

        for j in tqdm(range(len)):
            try:
                w0 = newton(disp, wg[j], tol=1e-20, maxiter=10000)
            except:
                w0 = np.complex128(0)
            if np.abs(disp(w0)) < 1e-6:
                roots[j] = w0
                if np.imag(w0) > 1e-6:
                    break
        return roots

    def disp_kzkp_Tz_harmonics(self, kz, kp, n, w0g, tol=1e-20, iter=40000, pol="circ", print_disp=False):
        if not self.Tz:
            print("Missing value of Tz. Aborting...")
            sys.exit()
        if type(w0g) is float or type(w0g) is complex:
            w0g = [w0g]
        if pol != "circ" and pol != "Ex" and pol != "Ey":
            pol = "circ"
            print("** Warning: invalid polarization. Defaulting to circular. **")
        wg = np.array(w0g, dtype=np.complex128)
        len = wg.shape[0]
        roots = np.zeros(wg.shape, dtype=np.complex128)

        n = np.array(n)
        n = np.sort(n)
        J0 = np.zeros(n.shape, dtype=np.float128)
        J1 = np.zeros(n.shape, dtype=np.float128)
        J2 = np.zeros(n.shape, dtype=np.float128)
        Omega = self.Omega
        p0 = self.p0
        Tz = self.Tz
        gamma = np.sqrt(1 + p0**2)

        b = kp * p0 / Omega

        J0[:] = J(n[:], b)
        J1[:] = dJ(n[:], b)
        J2[:] = dJ(n[:], b, 2)

        Kxx11 = np.sqrt(2 / Tz) * Omega / (p0 * kz * kp) * (n[:] ** 2 * J0[:] * J1[:])
        Kxx12 = -(Omega**2) / (gamma * Tz * kz**2 * kp**2) * (n[:] ** 2 * J0[:] * J0[:])
        Kxx21 = Omega**2 / (kp**2 * gamma**3) * (n[:] ** 2 * J0[:] * J0[:])
        Kxx22 = -2 * Omega / (gamma * p0 * kp) * (n[:] ** 2 * J0[:] * J1[:])
        Kxx23 = -(Omega**2) / (gamma**2 * np.sqrt(2 * Tz) * kz * kp**2) * (n[:] ** 2 * J0[:] * J0[:])
        Kxx31 = Omega**2 / (gamma * kp**2 * Tz) * (n[:] ** 2 * J0[:] * J0[:])

        Kyy41 = np.sqrt(2 / Tz) / kz * (J1[:] * J1[:])
        Kyy42 = np.sqrt(2 / Tz) / kz * b * (J1[:] * J2[:])
        Kyy43 = -(p0**2) / (gamma * Tz * kz**2) * (J1[:] * J1[:])
        Kyy51 = p0**2 / gamma**3 * (J1[:] * J1[:])
        Kyy52 = -2 / gamma * (J1[:] * J1[:])
        Kyy53 = -2 / gamma * b * (J1[:] * J2[:])
        Kyy54 = -(p0**2) / (np.sqrt(2 * Tz) * kz * gamma**2) * (J1[:] * J1[:])
        Kyy61 = p0**2 / (gamma * Tz) * (J1[:] * J1[:])

        Kxy71 = 1 / (b * np.sqrt(2 * Tz) * kz) * (n[:] * J0[:] * J1[:])
        Kxy72 = 1 / (np.sqrt(2 * Tz) * kz) * (n[:] * J1[:] * J1[:])
        Kxy73 = 1 / (np.sqrt(2 * Tz) * kz) * (n[:] * J0[:] * J2[:])
        Kxy74 = -Omega * p0 / (gamma * kp * Tz * kz**2) * (n[:] * J0[:] * J1[:])
        Kxy81 = Omega * p0 / (gamma**3 * kp) * (n[:] * J0[:] * J1[:])
        Kxy82 = -1 / (b * gamma) * (n[:] * J0[:] * J1[:])
        Kxy83 = -1 / gamma * (n[:] * J1[:] * J1[:])
        Kxy84 = -1 / gamma * (n[:] * J0[:] * J2[:])
        Kxy85 = -Omega * p0 / (np.sqrt(2 * Tz) * kz * kp * gamma**2) * (n[:] * J0[:] * J1[:])
        Kxy91 = Omega * p0 / (kp * gamma * Tz) * (n[:] * J0[:] * J1[:])

        def disp(w):
            xn = np.zeros(n.shape, dtype=np.complex128)
            zn = np.zeros(n.shape, dtype=np.complex128)
            xnzn_1 = np.zeros(n.shape, dtype=np.complex128)
            zxdiff = np.zeros(n.shape, dtype=np.complex128)
            xn[:] = (gamma * w + n[:] * Omega) / (np.sqrt(2 * Tz) * kz)
            zn[:] = Z(xn[:])
            xnzn_1[:] = 1 + xn[:] * zn[:]
            zxdiff[:] = zn[:] - 2 * xn[:] - 2 * xn[:] * xn[:] * zn[:]
            K1 = np.sum(w * Kxx11 * zn + w**2 * Kxx12 * xnzn_1)
            K2 = np.sum((Kxx21 + Kxx22) * xnzn_1 + w * Kxx23 * zxdiff)
            K3 = np.sum(Kxx31 * xnzn_1)
            K4 = np.sum(w * (Kyy41 + Kyy42) * zn + w**2 * Kyy43 * xnzn_1)
            K5 = np.sum((Kyy51 + Kyy52 + Kyy53) * xnzn_1 + w * Kyy54 * zxdiff)
            K6 = np.sum(Kyy61 * xnzn_1)
            K7 = np.sum(w * (Kxy71 + Kxy72 + Kxy73) * zn + w**2 * Kxy74 * xnzn_1)
            K8 = np.sum((Kxy81 + Kxy82 + Kxy83 + Kxy84) * xnzn_1 + w * Kxy85 * zxdiff)
            K9 = np.sum(Kxy91 * xnzn_1)

            Dxx = w**2 - kz**2 + K1 + K2 + K3
            Dyy = w**2 - kp**2 - kz**2 + K4 + K5 + K6
            DxyDyx = (K7 + K8 + K9) ** 2
            self.Dxx = Dxx
            self.Dyy = Dyy
            self.DxyDyx = np.sqrt(DxyDyx)
            if pol == "circ":
                return Dxx * Dyy - DxyDyx
            elif pol == "Ex":
                return Dxx
            elif pol == "Ey":
                return Dyy

        for j in range(len):
            try:
                w0 = newton(disp, wg[j], tol=tol, maxiter=iter)
                if np.abs(disp(w0)) < 1e-6:
                    roots[j] = w0
                    if print_disp:
                        print(disp(w0))
                    if np.imag(w0) > 1e-6:
                        break
                else:
                    print(w0, np.abs(disp(w0)))
            except:
                return
        return roots

    def disp_kzkp_TzTp(self, kz, kp, w0g):
        if not self.Tz or not self.Tp:
            print("Missing value of T. Aborting...")
            sys.exit()

        if type(w0g) is float or type(w0g) is complex:
            w0g = [w0g]
        wg = np.array(w0g, dtype=np.complex128)
        len = wg.shape[0]
        roots = np.zeros(wg.shape, dtype=np.complex128)

        epsabs = 1e-12
        epsrel = 1e-12
        Omega = self.Omega
        p0 = self.p0
        Tz = self.Tz
        Tp = self.Tp

        c0 = 1 / (2 * Tp)
        One_sqrt2Tk = 1 / (np.sqrt(2 * Tz) * kz)
        Omega_sqrt2Tk = Omega / (np.sqrt(2 * Tz) * kz)
        Omega2_kp2 = Omega**2 / kp**2

        def ufperp(p):
            return np.exp(-c0 * (p - p0) ** 2)

        def upfperp(p):
            return p * ufperp(p)

        norm = 1 / (quad(upfperp, 0, np.inf, epsabs=epsabs, epsrel=epsrel)[0])
        normdf = 2 * norm * c0

        def fp(p):
            return norm * np.exp(-c0 * (p - p0) ** 2)

        def dfp(p):
            return -normdf * (p - p0) * np.exp(-c0 * (p - p0) ** 2)

        def r1(p):
            return np.log10(fp(p)) + 10

        p0guess = p0 + np.sqrt(Tp)
        l = fsolve(r1, np.array([p0guess]))[0]
        l = np.abs(p0 - l)
        lim = [np.max([0, p0 - l]), p0 + l]
        lower_lim = np.max([lim[0], 0.0])
        upper_lim = lim[1]

        def Ixx_p0_g0_J1_J1_df_Zm(w):
            def integrand(p):
                jarg = p * kp / Omega
                xi = np.sqrt(1 + p**2) * One_sqrt2Tk * w - Omega_sqrt2Tk
                return J1(jarg) ** 2 * dfp(p) * Z(xi)

            integration = quad(integrand, lower_lim, upper_lim, epsabs=epsabs, epsrel=epsrel)[0]
            return integration

        def Ixx_p0_g1_J1_J1_df_1Zm(w):
            def integrand(p):
                gamma = np.sqrt(1 + p**2)
                jarg = p * kp / Omega
                xi = gamma * One_sqrt2Tk * w - Omega_sqrt2Tk
                return 1 / gamma * J1(jarg) ** 2 * dfp(p) * (1 + xi * Z(xi))

            integration = quad(integrand, lower_lim, upper_lim, epsabs=epsabs, epsrel=epsrel)[0]
            return integration

        def Ixx_p1_g1_J1_J1_f0_1Zm(w):
            def integrand(p):
                gamma = np.sqrt(1 + p**2)
                jarg = p * kp / Omega
                xi = gamma * One_sqrt2Tk * w - Omega_sqrt2Tk
                return p / gamma * J1(jarg) ** 2 * fp(p) * (1 + xi * Z(xi))

            integration = quad(integrand, lower_lim, upper_lim, epsabs=epsabs, epsrel=epsrel)[0]
            return integration

        """
        def Iyy_p2_g0_J1_J1_df_Z0(w):
            def integrand(p):
                gamma = np.sqrt(1 + p ** 2)
                jarg = p * kp / Omega
                xi = gamma * One_sqrt2Tk * w
                return p ** 2 * J1(jarg) ** 2 * dfp(p) * Z(xi)

            integration = quad(integrand, lower_lim, upper_lim, epsabs=epsabs, epsrel=epsrel)[0]
            return integration

        def Iyy_p2_g1_J1_J1_df_1Z0(w):
            def integrand(p):
                gamma = np.sqrt(1 + p ** 2)
                jarg = p * kp / Omega
                xi = gamma * One_sqrt2Tk * w
                return p ** 2 / gamma * J1(jarg) ** 2 * dfp(p) * (1 + xi * Z(xi))

            integration = quad(integrand, lower_lim, upper_lim, epsabs=epsabs, epsrel=epsrel)[0]
            return integration

        def Iyy_p3_g1_J1_J1_f0_1Z0(w):
            def integrand(p):
                gamma = np.sqrt(1 + p ** 2)
                jarg = p * kp / Omega
                xi = gamma * One_sqrt2Tk * w
                return p ** 3 / gamma * J1(jarg) ** 2 * fp(p) * (1 + xi * Z(xi))

            integration = quad(integrand, lower_lim, upper_lim, epsabs=epsabs, epsrel=epsrel)[0]
            return integration
        """

        def Iyy_p2_g0_Jp_Jp_df_Zm(w):
            def integrand(p):
                gamma = np.sqrt(1 + p**2)
                jarg = p * kp / Omega
                xi = gamma * One_sqrt2Tk * w - Omega_sqrt2Tk
                return p**2 * dJ(1, jarg) ** 2 * dfp(p) * Z(xi)

            integration = quad(integrand, lower_lim, upper_lim, epsabs=epsabs, epsrel=epsrel)[0]
            return integration

        def Iyy_p2_g1_Jp_Jp_df_1Zm(w):
            def integrand(p):
                gamma = np.sqrt(1 + p**2)
                jarg = p * kp / Omega
                xi = gamma * One_sqrt2Tk * w - Omega_sqrt2Tk
                return p**2 / gamma * dJ(1, jarg) ** 2 * dfp(p) * (1 + xi * Z(xi))

            integration = quad(integrand, lower_lim, upper_lim, epsabs=epsabs, epsrel=epsrel)[0]
            return integration

        def Iyy_p3_g1_Jp_Jp_f0_1Zm(w):
            def integrand(p):
                gamma = np.sqrt(1 + p**2)
                jarg = p * kp / Omega

                xi = gamma * One_sqrt2Tk * w - Omega_sqrt2Tk
                return p**3 / gamma * dJ(1, jarg) ** 2 * fp(p) * (1 + xi * Z(xi))

            integration = quad(integrand, lower_lim, upper_lim, epsabs=epsabs, epsrel=epsrel)[0]
            return integration

        def Ixy_p1_g0_J1_Jp_df_Zm(w):
            def integrand(p):
                gamma = np.sqrt(1 + p**2)
                jarg = p * kp / Omega
                xi = gamma * One_sqrt2Tk * w - Omega_sqrt2Tk
                return p * J1(jarg) * dJ(1, jarg) * dfp(p) * Z(xi)

            integration = quad(integrand, lower_lim, upper_lim, epsabs=epsabs, epsrel=epsrel)[0]
            return integration

        def Ixy_p1_g1_J1_Jp_df_1Zm(w):
            def integrand(p):
                gamma = np.sqrt(1 + p**2)
                jarg = p * kp / Omega
                xi = gamma * One_sqrt2Tk * w - Omega_sqrt2Tk
                return p / gamma * J1(jarg) * dJ(1, jarg) * dfp(p) * (1 + xi * Z(xi))

            integration = quad(integrand, lower_lim, upper_lim, epsabs=epsabs, epsrel=epsrel)[0]
            return integration

        def Ixy_p2_g1_J1_Jp_f0_1Zm(w):
            def integrand(p):
                gamma = np.sqrt(1 + p**2)
                jarg = p * kp / Omega
                xi = gamma * One_sqrt2Tk * w - Omega_sqrt2Tk
                return p**2 / gamma * J1(jarg) * dJ(1, jarg) * fp(p) * (1 + xi * Z(xi))

            integration = quad(integrand, lower_lim, upper_lim, epsabs=epsabs, epsrel=epsrel)[0]
            return integration

        def disp(w):
            I1 = One_sqrt2Tk * w * Ixx_p0_g0_J1_J1_df_Zm(w)
            I2 = Ixx_p0_g1_J1_J1_df_1Zm(w)
            I3 = Ixx_p1_g1_J1_J1_f0_1Zm(w) / Tz
            I4 = One_sqrt2Tk * w * (Iyy_p2_g0_Jp_Jp_df_Zm(w))  # + Iyy_p2_g0_J1_J1_df_Z0(w)
            I5 = Iyy_p2_g1_Jp_Jp_df_1Zm(w)  # + Iyy_p2_g1_J1_J1_df_1Z0(w)
            I6 = (Iyy_p3_g1_Jp_Jp_f0_1Zm(w)) / Tz  # + Iyy_p3_g1_J1_J1_f0_1Z0(w)
            I7 = -One_sqrt2Tk * w * Ixy_p1_g0_J1_Jp_df_Zm(w)
            I8 = -Ixy_p1_g1_J1_Jp_df_1Zm(w)
            I9 = -Ixy_p2_g1_J1_Jp_f0_1Zm(w) / Tz
            """
            print(w)
            print(I1)
            print(I2)
            print(I3)
            print(I4)
            print(I5)
            print(I6)
            print(I7)
            print(I8)
            print(I9)
            sys.exit()
            """
            Dxx = w**2 - kz**2 - Omega2_kp2 * (I1 - I2 - I3)
            Dyy = w**2 - kz**2 - kp**2 - (I4 - I5 - I6)
            DxyDyx = Omega2_kp2 * (I7 - I8 - I9) ** 2
            return Dxx * Dyy - DxyDyx

        for j in range(len):
            try:
                w0 = newton(disp, wg[j], tol=1e-12, maxiter=10000)
            # except ValueError or RuntimeWarning or RuntimeError or TypeError:
            except:
                w0 = np.complex128(0)
            if np.abs(disp(w0)) < 1e-7:
                roots[j] = w0
        return roots

    def disp_Xmodes_kzkp_TzTp(self, w, kp, kz, n=1):
        Tz = self.Tz
        Tp = self.Tp
        p0 = self.p0
        Omega = self.Omega
        sqrt2Tzkz = np.sqrt(2 * Tz) * kz
        norm = 1 / (self.eta * Tp)

        def fp(p):
            return norm * np.exp(-((p - p0) ** 2) / (2 * Tp))

        def dfp(p):
            return -norm / Tp * (p - p0) * np.exp(-((p - p0) ** 2) / (2 * Tp))

        nrange = np.arange(-n, n + 1)
        integrals = np.zeros(nrange.shape, dtype=np.complex128)

        def int_01(n):
            def intre(p, n):
                g = np.sqrt(p**2 + 1)
                bp = kp * p / Omega
                dd = dJ(n, bp) ** 2
                xi = (g * w - n * Omega) / sqrt2Tzkz
                return np.real(p**2 / g * dd * dfp(p))

            def intim(p, n):
                g = np.sqrt(p**2 + 1)
                bp = kp * p / Omega
                dd = dJ(n, bp) ** 2
                xi = (g * w - n * Omega) / sqrt2Tzkz
                return np.imag(p**2 / g * dd * dfp(p))

            intr = quadsci(intre, self.lower_lim, self.upper_lim, args=(n,), limit=150)[0]
            inti = quadsci(intim, self.lower_lim, self.upper_lim, args=(n,), limit=150)[0]

            return intr + 1j * inti

        def int_02(n):
            def intre(p, n):
                g = np.sqrt(p**2 + 1)
                bp = kp * p / Omega
                dd = dJ(n, bp) ** 2
                xi = (g * w - n * Omega) / sqrt2Tzkz
                Zxi = Z(xi)
                fac = (-n * Omega) / sqrt2Tzkz
                return np.real(fac * p**2 / g * dd * dfp(p) * Zxi)

            def intim(p, n):
                g = np.sqrt(p**2 + 1)
                bp = kp * p / Omega
                dd = dJ(n, bp) ** 2
                xi = (g * w - n * Omega) / sqrt2Tzkz
                Zxi = Z(xi)
                fac = (-n * Omega) / sqrt2Tzkz
                return np.imag(fac * p**2 / g * dd * dfp(p) * Zxi)

            intr = quadsci(intre, self.lower_lim, self.upper_lim, args=(n,), limit=150)[0]
            inti = quadsci(intim, self.lower_lim, self.upper_lim, args=(n,), limit=150)[0]

            return intr + 1j * inti

        def int_03(n):
            def intre(p, n):
                g = np.sqrt(p**2 + 1)
                bp = kp * p / Omega
                dd = dJ(n, bp) ** 2
                xi = (g * w - n * Omega) / sqrt2Tzkz
                Zxi = Z(xi)
                fac = 1 / Tz
                return np.real(fac * p**3 / g * dd * fp(p) * (1 + xi * Zxi))

            def intim(p, n):
                g = np.sqrt(p**2 + 1)
                bp = kp * p / Omega
                dd = dJ(n, bp) ** 2
                xi = (g * w - n * Omega) / sqrt2Tzkz
                Zxi = Z(xi)
                fac = 1 / Tz
                return np.real(fac * p**3 / g * dd * fp(p) * (1 + xi * Zxi))

            intr = quadsci(intre, self.lower_lim, self.upper_lim, args=(n,), limit=150)[0]
            inti = quadsci(intim, self.lower_lim, self.upper_lim, args=(n,), limit=150)[0]

            return intr + 1j * inti

        for i in range(nrange.shape[0]):
            n = nrange[i]
            I1 = int_01(n)
            I2 = int_02(n)
            I3 = int_03(n)
            integrals[i] = I1 + I2 + I3

        return w**2 - kp**2 - kz**2 + np.sum(integrals)

    def sol_Xmodes_kzkp_TzTp(self, w0g, kp, kz, n=1):
        w0g = np.complex128(w0g)
        w0 = newton(self.disp_Xmodes_kzkp_TzTp, w0g, args=(kp, kz, n), tol=1e-12, maxiter=10000)
        if np.abs(self.disp_Xmodes_kzkp_TzTp(w0, kp, kz, n)) < 1e-8:
            return w0
        else:
            return 0


# %%


class MaserMultipleColdRings:
    def __init__(self, p0, Omega):
        self.p0 = p0
        self.Omega = Omega
        self.NR = len(self.p0)

    def sol_kp_cold_one_harm(self, kp, n, all=True):
        k = kp
        Omega = self.Omega
        p0 = self.p0
        gamma0 = np.sqrt(1 + p0**2)
        b0 = k * p0 / Omega
        J_n = J(n, b0)
        dJ_n = dJ(n, b0, 1)
        d2J_n = dJ(n, b0, 2)
        An = n**2 * p0**2 / (gamma0**3 * b0**2) * J_n**2
        Bn = -2 * n**2 / (gamma0 * b0) * J_n * dJ_n
        Cn = p0**2 / gamma0**3 * dJ_n**2
        Dn = -2 / gamma0 * dJ_n * (b0 * d2J_n + dJ_n)
        En = n * p0**2 / (gamma0**3 * b0) * J_n * dJ_n
        Fn = -n / gamma0 * (1 / b0 * J_n * dJ_n + dJ_n**2 + J_n * d2J_n)

        c0 = -((Bn * k**2 * n**3 * Omega**3) / gamma0**3)
        c1 = (Bn * Dn * n**2 * Omega**2) / gamma0**2 - (Fn**2 * n**2 * Omega**2) / gamma0**2 - (An * k**2 * n**2 * Omega**2) / gamma0**2 - (3 * Bn * k**2 * n**2 * Omega**2) / gamma0**2 - (k**2 * n**4 * Omega**4) / gamma0**4
        c2 = (Bn * Cn * n * Omega) / gamma0 + (An * Dn * n * Omega) / gamma0 + (2 * Bn * Dn * n * Omega) / gamma0 - (2 * En * Fn * n * Omega) / gamma0 - (2 * Fn**2 * n * Omega) / gamma0 - (2 * An * k**2 * n * Omega) / gamma0 - (3 * Bn * k**2 * n * Omega) / gamma0 + (Bn * n**3 * Omega**3) / gamma0**3 + (Dn * n**3 * Omega**3) / gamma0**3 - (4 * k**2 * n**3 * Omega**3) / gamma0**3
        c3 = An * Cn + Bn * Cn + An * Dn + Bn * Dn - En**2 - 2 * En * Fn - Fn**2 - An * k**2 - Bn * k**2 + (An * n**2 * Omega**2) / gamma0**2 + (3 * Bn * n**2 * Omega**2) / gamma0**2 + (Cn * n**2 * Omega**2) / gamma0**2 + (3 * Dn * n**2 * Omega**2) / gamma0**2 - (6 * k**2 * n**2 * Omega**2) / gamma0**2 + (n**4 * Omega**4) / gamma0**4
        c4 = (2 * An * n * Omega) / gamma0 + (3 * Bn * n * Omega) / gamma0 + (2 * Cn * n * Omega) / gamma0 + (3 * Dn * n * Omega) / gamma0 - (4 * k**2 * n * Omega) / gamma0 + (4 * n**3 * Omega**3) / gamma0**3
        c5 = An + Bn + Cn + Dn - k**2 + (6 * n**2 * Omega**2) / gamma0**2
        c6 = (4 * n * Omega) / gamma0
        c7 = 1
        arr = np.array([c7, c6, c5, c4, c3, c2, c1, c0])
        roots = np.roots(arr)
        pol = [np.abs((r**2 * (r + n * Omega / gamma0) ** 2 + r**2 * An + r * Bn * (r + n * Omega / gamma0)) / (r**2 * En + r * Fn * (r + n * Omega / gamma0))) for r in roots]
        if all == False:
            id = np.argsort((np.imag(roots)))[-1]
            return pol[id], roots[id]
        else:
            return pol, roots

    def disp_kp_cold(self, w, kp, n):
        k = kp
        Omega = self.Omega
        p0 = self.p0
        NR = self.NR
        gamma_R = [np.sqrt(1 + i**2) for i in p0]
        Omrel_R = [Omega / np.sqrt(1 + i**2) for i in p0]
        b_R = [k / Omega * i for i in p0]
        print(p0)
        print(NR)
        print(gamma_R)
        print(Omrel_R)
        print(b_R)

        An = np.zeros((NR,))
        Bn = np.zeros((NR,))
        Cn = np.zeros((NR,))
        Dn = np.zeros((NR,))
        En = np.zeros((NR,))
        Fn = np.zeros((NR,))
        """
        for i in range(lenn):
            J_n = J(n[i], b0)
            dJ_n = dJ(n[i], b0, 1)
            d2J_n = dJ(n[i], b0, 2)

            An[i] = n[i] ** 2 * p0**2 / (gamma0**3 * b0**2) * J_n**2
            Bn[i] = -2 * n[i] ** 2 / (gamma0 * b0) * J_n * dJ_n
            Cn[i] = p0**2 / gamma0**3 * dJ_n**2
            Dn[i] = -2 / gamma0 * dJ_n * (b0 * d2J_n + dJ_n)
            En[i] = n[i] * p0**2 / (gamma0**3 * b0) * J_n * dJ_n
            Fn[i] = -n[i] / gamma0 * (1 / b0 * J_n * dJ_n + dJ_n**2 + J_n * d2J_n)

        Dxx = w**2 + np.sum(An[:] * w**2 / (w + n[:] * Omrel) ** 2 + Bn[:] * w / (w + n[:] * Omrel))
        Dyy = w**2 - k**2 + np.sum(Cn[:] * w**2 / (w + n[:] * Omrel) ** 2 + Dn[:] * w / (w + n[:] * Omrel))
        Dxy = np.sum(En[:] * w**2 / (w + n[:] * Omrel) ** 2 + Fn[:] * w / (w + n[:] * Omrel))
        return Dxx * Dyy - Dxy * Dxy
        """

    def pol_kp_cold(self, w, kp, n, symmetric=True):
        if symmetric == True:
            n = np.arange(-n, n + 1)
        else:
            n = np.array(n)
        k = kp
        Omega = self.Omega
        p0 = self.p0
        gamma0 = np.sqrt(1 + p0**2)
        Omrel = Omega / gamma0
        b0 = k * p0 / Omega
        lenn = n.shape[0]
        An = np.zeros((lenn,))
        Bn = np.zeros((lenn,))
        En = np.zeros((lenn,))
        Fn = np.zeros((lenn,))
        for i in range(lenn):
            J_n = J(n[i], b0)
            dJ_n = dJ(n[i], b0, 1)
            d2J_n = dJ(n[i], b0, 2)

            An[i] = n[i] ** 2 * p0**2 / (gamma0**3 * b0**2) * J_n**2
            Bn[i] = -2 * n[i] ** 2 / (gamma0 * b0) * J_n * dJ_n
            En[i] = n[i] * p0**2 / (gamma0**3 * b0) * J_n * dJ_n
            Fn[i] = -n[i] / gamma0 * (1 / b0 * J_n * dJ_n + dJ_n**2 + J_n * d2J_n)

        Dxx = w**2 + np.sum(An[:] * w**2 / (w + n[:] * Omrel) ** 2 + Bn[:] * w / (w + n[:] * Omrel))
        Dxy = np.sum(En[:] * w**2 / (w + n[:] * Omrel) ** 2 + Fn[:] * w / (w + n[:] * Omrel))

        return np.abs(Dxx / Dxy)

    def sol_kp_cold(self, wg, kp, n=3, symmetric=True):
        try:
            w0 = newton(
                self.disp_kp_cold,
                np.complex128(wg),
                tol=1e-12,
                args=(
                    kp,
                    n,
                    symmetric,
                ),
                maxiter=10000,
            )
        except:
            w0 = 0.0
        if np.abs(self.disp_kp_cold(w0, kp, n, symmetric)) < 1e-8:
            if np.imag(w0) > 1e-10:
                pol = self.pol_kp_cold(w0, kp, n, symmetric)
            else:
                pol = 1
            return pol, w0
        else:
            return 0.0, 0.0


# Maser model


class Maser_Model:
    def __init__(self, p0, Omega, Tz=None):
        self.p0 = p0
        self.Omega = Omega
        self.Tz = Tz

    def kz_smin(self):
        p0 = self.p0
        Omega = self.Omega
        g0 = np.sqrt(1 + p0**2)
        return np.sqrt(Omega**2 / g0**2 - (g0**2 + 1) / (4 * g0**3) + np.sqrt(8 * Omega**2 / g0**2 + g0**3 + g0 - (1 + g0**2) / (g0**3)) / (4 * p0 * g0 ** (3 / 2)))

    def kz_smax(self):
        p0 = self.p0
        g0 = np.sqrt(1 + p0**2)
        Tz = self.Tz
        return p0 / np.sqrt(2 * g0 * Tz)

    def gr_smax(self):
        p0 = self.p0
        Omega = self.Omega
        g0 = np.sqrt(1 + p0**2)
        Tz = self.Tz
        O_supp = 0.5 * np.sqrt(g0 * (g0**2 / p0**2 + 2 * p0**2 / Tz) - np.sqrt(g0**6 / p0**4 + 4 / Tz))
        return p0 / np.sqrt(2 * g0**3) * (1 - Omega / O_supp)

    def Omega_funs(self):
        p0 = self.p0
        g0 = np.sqrt(1 + p0**2)
        return 1 / (2 * p0) * np.sqrt(2 * p0**2 / g0 + g0**3 - 12 * p0**2 * g0 + (g0**2 + 4 * p0**2) ** (3 / 2))

    def kz_fmax(self):
        p0 = self.p0
        Omega = self.Omega
        g0 = np.sqrt(1 + p0**2)
        return np.sqrt(Omega**2 / g0**2 - (g0**2 + 1) / (4 * g0**3) - np.sqrt(8 * Omega**2 / g0**2 + g0**3 + g0 - (1 + g0**2) / (g0**3)) / (4 * p0 * g0 ** (3 / 2)))

    def gr_fmax(self):
        p0 = self.p0
        Omega = self.Omega
        g0 = np.sqrt(1 + p0**2)
        O_th = (1 + g0**2) / (2 * np.sqrt(2 * g0) * p0)
        alp = np.sqrt(2) * p0 / g0 ** (5 / 2)
        return alp / np.sqrt(2) * np.sqrt(Omega**2 - O_th**2) / np.sqrt(alp * O_th + Omega**2 / g0**2 + np.sqrt((Omega**2 / g0**2 + alp * Omega) ** 2 - 2 * alp * Omega**2 * (Omega - O_th) / (g0**2)))


# %%


class ES_Ring:
    def __init__(self, v0, Tp):
        self.v0 = v0
        self.Tp = Tp
        norm = 1 / (2 * np.pi * Tp * np.exp(-(v0**2) / (2 * Tp)) + np.sqrt(2 * np.pi**3 * Tp) * v0 * (1 + erf(v0 / np.sqrt(2 * Tp))))
        self.normT = norm / Tp

        def fp(v):
            return norm * np.exp(-((v - v0) ** 2) / (2 * Tp))

        def r1(p):
            return np.log10(fp(p) / norm) + 10

        p0guess = v0 + np.sqrt(Tp)
        l = fsolve(r1, np.array([p0guess]))[0]
        l = np.abs(v0 - l)
        lim = [np.max([0, v0 - l]), v0 + l]
        self.lower_lim = lim[0]
        self.upper_lim = lim[1]

    def disp(self, w, k):
        v0 = self.v0
        Tp = self.Tp

        def dfv(v):
            return -self.normT * (v - v0) * np.exp(-((v - v0) ** 2) / (2 * Tp))

        def intr_plus(theta, w, k):
            v_ph = w / (k * np.cos(theta))
            a = np.real(v_ph)
            b = np.imag(v_ph)

            def intre(v):
                return v * (v - a) * dfv(v) / ((v - a) ** 2 + b**2)

            def intim(v):
                return b * v * dfv(v) / ((v - a) ** 2 + b**2)

            intr = quadsci(intre, self.lower_lim, self.upper_lim, limit=150)[0]
            inti = quadsci(intim, self.lower_lim, self.upper_lim, limit=150)[0]

            return intr + 1j * inti

        def intr_mins(theta, w, k):
            costh = np.cos(theta)
            v_ph = w / (k * costh)
            a = np.real(v_ph)
            b = np.imag(v_ph)

            def intre(v):
                return v * (v - a) * dfv(v) / ((v - a) ** 2 + b**2)

            def intim(v):
                return b * v * dfv(v) / ((v - a) ** 2 + b**2)

            intr = quadsci(intre, self.lower_lim, self.upper_lim, limit=150)[0]
            inti = quadsci(intim, self.lower_lim, self.upper_lim, limit=150)[0]

            pole = 0
            if costh > 0:
                pole = 2j * np.pi * v_ph * dfv(v_ph)

            return intr + 1j * inti + pole

        def intr_zero(theta, w, k):
            costh = np.cos(theta)
            v_ph = np.real(w / (k * costh))

            def integrand(v):
                return v * dfv(v)

            int0 = quadsci(
                integrand,
                self.lower_lim,
                self.upper_lim,
                weight="cauchy",
                wvar=v_ph,
                limit=150,
            )[0]

            pole = 0
            if costh > 0:
                pole = 1j * np.pi * v_ph * dfv(v_ph)

            return int0 + pole

        def intth(w, k):
            b = np.imag(w)

            if np.abs(b) < 1e-6:

                def integrand(th):
                    return intr_zero(th, w, k)

            elif b > 0:

                def integrand(th):
                    return intr_plus(th, w, k)

            else:

                def integrand(th):
                    return intr_mins(th, w, k)

            I1 = quadsci(integrand, 0, np.pi / 2 - 1e-8, complex_func=True)[0]
            I2 = quadsci(integrand, np.pi / 2 + 1e-8, 3 * np.pi / 2 - 1e-8, complex_func=True)[0]
            I3 = quadsci(integrand, 3 * np.pi / 2 + 1e-8, 2 * np.pi, complex_func=True)[0]
            return I1 + I2 + I3

        return k**2 - (intth(w, k))

    def sol(self, w0g, k):
        w0g = np.complex128(w0g)
        w0 = newton(self.disp, w0g, args=(k,), tol=1e-12)
        if np.abs(self.disp(w0, k)) < 1e-8:
            return w0
        else:
            return 0


# %%


class EM_Ring:
    def __init__(self, v0, Tp, Tz):
        self.v0 = v0
        self.Tp = Tp
        self.Tz = Tz
        norm = 1 / (2 * np.pi * Tp * np.exp(-(v0**2) / (2 * Tp)) + np.sqrt(2 * np.pi**3 * Tp) * v0 * (1 + erf(v0 / np.sqrt(2 * Tp))))
        self.normT = norm / Tp

        def fp(v):
            return norm * np.exp(-((v - v0) ** 2) / (2 * Tp))

        def r1(p):
            return np.log10(fp(p) / norm) + 10

        def int0(v):
            return v**3 * fp(v)

        II = quadsci(int0, 0, np.inf)[0]

        self.Teff = np.pi * II

        p0guess = v0 + np.sqrt(Tp)
        l = fsolve(r1, np.array([p0guess]))[0]
        l = np.abs(v0 - l)
        lim = [np.max([0, v0 - l]), v0 + l]
        self.lower_lim = lim[0]
        self.upper_lim = lim[1]

    def disp_perp_to_ring(self, w, k):
        Tz = self.Tz
        Tratio = self.Teff / self.Tz
        xi = w / (np.sqrt(2 * Tz) * k)
        Zxi = Z(xi)

        return w**2 - k**2 - 1 + Tratio * (1 + xi * Zxi)

    def sol_perp_to_ring(self, w0g, k):
        w0g = np.complex128(w0g)
        w0 = newton(self.disp_perp_to_ring, w0g, args=(k,), tol=1e-12)
        if np.abs(self.disp_perp_to_ring(w0, k)) < 1e-8:
            return w0
        else:
            return 0

    def disp_prll_to_ring(self, w, k):
        v0 = self.v0
        Tp = self.Tp
        Tz = self.Tz

        def dfv(v):
            return -self.normT * (v - v0) * np.exp(-((v - v0) ** 2) / (2 * Tp))

        def intr_plus(theta, w, k):
            v_ph = w / (k * np.cos(theta))
            a = np.real(v_ph)
            b = np.imag(v_ph)

            def intre(v):
                return v * (v - a) * dfv(v) / ((v - a) ** 2 + b**2)

            def intim(v):
                return b * v * dfv(v) / ((v - a) ** 2 + b**2)

            intr = quadsci(intre, self.lower_lim, self.upper_lim, limit=150)[0]
            inti = quadsci(intim, self.lower_lim, self.upper_lim, limit=150)[0]

            return intr + 1j * inti

        def intr_mins(theta, w, k):
            costh = np.cos(theta)
            v_ph = w / (k * costh)
            a = np.real(v_ph)
            b = np.imag(v_ph)

            def intre(v):
                return v * (v - a) * dfv(v) / ((v - a) ** 2 + b**2)

            def intim(v):
                return b * v * dfv(v) / ((v - a) ** 2 + b**2)

            intr = quadsci(intre, self.lower_lim, self.upper_lim, limit=150)[0]
            inti = quadsci(intim, self.lower_lim, self.upper_lim, limit=150)[0]

            pole = 0
            if costh > 0:
                pole = 2j * np.pi * v_ph * dfv(v_ph)

            return intr + 1j * inti + pole

        def intr_zero(theta, w, k):
            costh = np.cos(theta)
            v_ph = np.real(w / (k * costh))

            def integrand(v):
                return v * dfv(v)

            int0 = quadsci(
                integrand,
                self.lower_lim,
                self.upper_lim,
                weight="cauchy",
                wvar=v_ph,
                limit=150,
            )[0]

            pole = 0
            if costh > 0:
                pole = 1j * np.pi * v_ph * dfv(v_ph)

            return int0 + pole

        def intth(w, k):
            b = np.imag(w)

            if np.abs(b) < 1e-6:

                def integrand(th):
                    return intr_zero(th, w, k)

            elif b > 0:

                def integrand(th):
                    return intr_plus(th, w, k)

            else:

                def integrand(th):
                    return intr_mins(th, w, k)

            I1 = quadsci(integrand, 0, np.pi / 2 - 1e-8, complex_func=True)[0]
            I2 = quadsci(integrand, np.pi / 2 + 1e-8, 3 * np.pi / 2 - 1e-8, complex_func=True)[0]
            I3 = quadsci(integrand, 3 * np.pi / 2 + 1e-8, 2 * np.pi, complex_func=True)[0]
            return I1 + I2 + I3

        return w**2 - k**2 - 1 - Tz * intth(w, k)

    def sol_prll_to_ring(self, w0g, k):
        w0g = np.complex128(w0g)
        w0 = newton(self.disp_prll_to_ring, w0g, args=(k,), tol=1e-12)
        if np.abs(self.disp_prll_to_ring(w0, k)) < 1e-8:
            return w0
        else:
            return 0


# %%
