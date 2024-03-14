import scipy.constants as cons
import numpy as np
import sys

###
e = cons.e
m_e = cons.m_e
epsilon_0 = cons.epsilon_0
rest_ene_electron = cons.physical_constants["electron mass energy equivalent in MeV"][0]
c = cons.c
###


class unit_conversion:
    def __init__(
        self, unit_of_measurement, quant, unitin, unit, symbols=None, conversion_index=None, conversion_value=None,
    ):
        unit_of_measurement = unit_of_measurement.lower()
        available_unit_meas = [
            ["length", "len", "l"],
            ["time", "t"],
            ["efield", "ef", "electric field", "electricfield"],
            ["bfield", "bf", "magnetic field", "magneticfield"],
            ["energy", "ene"],
            ["freq", "frequency"],
            ["charge", "q"],
            ["current", "I"],
            ["power", "P"],
            ["laserintensity", "I0", "intensity"],
            ["custom"],
        ]
        # @auto-fold here
        if not any(unit_of_measurement in sl for sl in available_unit_meas):
            print(
                "Unit of measurament not identified. Allowed values are:", [i[0] for i in available_unit_meas],
            )
            print("Aborting...")
            sys.exit()
        unit_of_measurement = self.unit_fix(unit_of_measurement, available_unit_meas)
        self.unit_of_measurement = unit_of_measurement
        self.quant = quant
        self.unit = unit
        self.unitin = unitin
        self.symbols = symbols
        self.conversion_index = conversion_index
        self.conversion_value = conversion_value
        exec("self." + unit_of_measurement + "()")

    # @auto-fold here
    def unit_fix(self, unit_of_measurement, available_unit_meas):
        for sl in available_unit_meas:
            for tl in sl:
                if tl == unit_of_measurement:
                    unit_of_measurement = sl[0]
                    return sl[0]

    # @auto-fold here
    def metric_prefix(self, symbol):
        mp_sym = [
            "Y",
            "Z",
            "E",
            "P",
            "T",
            "G",
            "M",
            "k",
            "h",
            "da",
            "",
            "d",
            "c",
            "m",
            "u",
            "n",
            "p",
            "f",
            "a",
            "z",
            "y",
        ]
        mp_val = [
            24,
            21,
            18,
            15,
            12,
            9,
            6,
            3,
            2,
            1,
            0,
            -1,
            -2,
            -3,
            -6,
            -9,
            -12,
            -15,
            -18,
            -21,
            -24,
        ]
        try:
            sm = mp_sym.index(symbol)
        except ValueError:
            print(symbol + " unit prefix not identified. Aborting...")
            sys.exit()
        return mp_val[sm]

    # @auto-fold here
    def prefix(self, unit, symbols, sign):
        unit_conv_val = None
        for sym in symbols:
            lensym = len(sym)
            if unit[-lensym:] == sym:
                unit_conv_val = True
                self.unit_conv_idx = symbols.index(sym)
                unit_prefix = unit[:-lensym]
        if not unit_conv_val:
            print("Unit not identified. Aborting...")
            sys.exit()
        unit_prefix_val = self.metric_prefix(unit_prefix)
        if sign == "+":
            factor = 10 ** (unit_prefix_val)
        elif sign == "-":
            factor = 10 ** (-unit_prefix_val)
        else:
            print("Valid values of direction are + and -. Aborting...")
            sys.exit()
        return factor

    # @auto-fold here
    def separate(self, unit):
        sym_slash = unit.find("/")
        sym_up = unit[:sym_slash]
        sym_down = unit[sym_slash + 1 :]
        return sym_up, sym_down

    # @auto-fold here
    def unit_exponent(self, unit):
        sym_exp = unit.find("^")
        if sym_exp == -1:
            return unit, 1
        else:
            un = unit[:sym_exp]
            exp = unit[sym_exp + 1 :]
            return un, exp

    # @auto-fold here
    def match_units(self, sym_up, sym_down):
        match_unit = False
        for j in range(len(self.symbols)):
            if self.symbols[j] == sym_up[-len(self.symbols[j]) :] and self.symbols_[j] == sym_down[-len(self.symbols_[j]) :]:
                match_unit = True
                return
        if not match_unit:
            print(
                self.unit, "it is not a unit of", self.unit_of_measurement + ".", "Aborting...",
            )
            sys.exit()

    # @auto-fold here
    def conv_1(self):
        fac_in = self.prefix(self.unitin, self.symbols, "+")
        fac_in *= self.conversion_value[self.conversion_index[self.unit_conv_idx]]
        fac_out = self.prefix(self.unit, self.symbols, "-")
        fac_out /= self.conversion_value[self.conversion_index[self.unit_conv_idx]]
        self.quant *= fac_in * fac_out

    # @auto-fold here
    def conv_2(self):
        sym_up, sym_down = self.separate(self.unitin)
        self.match_units(sym_up, sym_down)
        _, exp_up = self.unit_exponent(sym_up)
        _, exp_down = self.unit_exponent(sym_down)
        exp_up = float(exp_up)
        exp_down = float(exp_down)
        fac_in_1 = self.prefix(sym_up, self.symbols, "+")
        fac_in_2 = self.prefix(sym_down, self.symbols_, "-")
        fac_in = (fac_in_1) ** exp_up * (fac_in_2) ** exp_down * self.conversion_value[self.conversion_index[self.unit_conv_idx]]
        sym_up, sym_down = self.separate(self.unit)
        self.match_units(sym_up, sym_down)
        _, exp_up = self.unit_exponent(sym_up)
        _, exp_down = self.unit_exponent(sym_down)
        exp_up = float(exp_up)
        exp_down = float(exp_down)
        fac_out_1 = self.prefix(sym_up, self.symbols, "-")
        fac_out_2 = self.prefix(sym_down, self.symbols_, "+")
        fac_out = (fac_out_1) ** exp_up * (fac_out_2) ** exp_down / self.conversion_value[self.conversion_index[self.unit_conv_idx]]
        self.quant *= fac_in * fac_out

    # @auto-fold here
    def conv_val(self):
        return self.quant

    # @auto-fold here
    def conv(self, decimal=5):
        return format(self.quant, "." + str(decimal) + "e")

    # @auto-fold here
    def custom(self):
        if self.symbols == None or self.conversion_index == None or self.conversion_value == None:
            print("Custom option was selected, but no symbols, conversion index and/or conversion value were provided. Aborting...")
            sys.exit()
        elif not len(self.symbols) == len(self.conversion_index) or not max(self.conversion_index) == len(self.conversion_value) - 1:
            print("Error matching the symbols, conversion index and/or conversion value. Aborting...")
            sys.exit()
        self.conv_1()

    # @auto-fold here
    def time(self):
        self.symbols = [
            "s",
            "sec",
            "second",
            "seconds",
            "min",
            "minute",
            "minutes",
            "h",
            "hour",
            "hours",
        ]
        self.conversion_index = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
        self.conversion_value = [1, 60, 3600]
        self.conv_1()

    # @auto-fold here
    def length(self):
        self.symbols = [
            "m",
            "meter",
            "meters",
            "metre",
            "metres",
            "ft",
            "foot",
            "feet",
            "in",
            "inches",
            "inch",
            "yd",
            "yards",
            "yard",
            "mi",
            "miles",
            "mile",
            "nmi",
            "Angstrom",
        ]
        self.conversion_index = [
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            2,
            2,
            2,
            3,
            3,
            3,
            4,
            4,
            4,
            5,
            6,
        ]
        self.conversion_value = [1, 0.3048, 0.0254, 0.9144, 1609.344, 1852, 1e-10]
        self.conv_1()

    # @auto-fold here
    def efield(self):
        self.symbols = ["V", "N"]
        self.symbols_ = ["m", "C"]
        self.conversion_index = [0, 0]
        self.conversion_value = [1]
        self.conv_2()

    # @auto-fold here
    def bfield(self):
        self.symbols = ["T", "G"]
        self.conversion_index = [0, 1]
        self.conversion_value = [1, 1e-4]
        self.conv_1()

    # @auto-fold here
    def energy(self):
        self.symbols = [
            "J",
            "joule",
            "joules",
            "eV",
            "electron volt",
            "erg",
            "ergs",
            "cal",
            "calorie",
            "calories",
            "BTU",
            "W.h",
        ]
        self.conversion_index = [0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 5]
        self.conversion_value = [1, e, 1e-7, 4.184, 1054.3503, 3600]
        self.conv_1()

    # @auto-fold here
    def freq(self):
        self.symbols = ["Hz"]
        self.conversion_index = [0]
        self.conversion_value = [1]
        self.conv_1()

    # @auto-fold here
    def charge(self):
        self.symbols = [
            "C",
            "coulomb",
            "coulombs",
            "e",
            "elementary charge",
            "esu",
            "electrostatic unit of charge",
            "Fr",
            "franklin",
            "statcoulomb",
            "statC",
        ]
        self.conversion_index = [0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2]
        self.conversion_value = [1, e, 3.3356409519815206e-10]
        self.conv_1()

    # @auto-fold here
    def current(self):
        self.symbols = ["A", "ampere", "amperes", "statampere", "statA"]
        self.conversion_index = [0, 0, 0, 1, 1]
        self.conversion_value = [1, 1 / (10 * c)]
        self.conv_1()

    # @auto-fold here
    def power(self):
        self.symbols = ["W"]
        self.conversion_index = [0]
        self.conversion_value = [1]
        self.conv_1()

    # @auto-fold here
    def laserintensity(self):
        self.symbols = ["W"]
        self.symbols_ = ["m^2"]
        self.conversion_index = [0]
        self.conversion_value = [1]
        self.conv_2()


def unit_conv(
    unit_of_measurement, quant, unitin, unit, symbols=None, conversion_index=None, conversion_value=None,
):
    return unit_conversion(unit_of_measurement, quant, unitin, unit, symbols=None, conversion_index=None, conversion_value=None,).conv_val()


class plasma_parameters:
    def __init__(self, *args, **kwargs):
        self.check_kwargs(args, kwargs)

    def wp_inv(self, n0=None, unit="s"):
        n0 = self.check_n0(n0)
        wp = np.float128(e * np.sqrt(n0 / (m_e * epsilon_0)))
        wp_inv = 1 / wp
        wp_inv = unit_conv("time", wp_inv, "s", unit)
        return wp_inv

    def wp(self, n0=None, unit="THz"):
        wp_inv = self.wp_inv(n0, unit="s")
        wp = 1 / wp_inv
        wp = unit_conv("freq", wp, "Hz", unit)
        return wp

    def cwp(self, n0=None, unit="um"):
        wp_inv = self.wp_inv(n0, unit="s")
        cwp = c * wp_inv
        cwp = unit_conv("length", cwp, "m", unit)
        return cwp

    def efield(self, n0=None, unit="GV/m"):
        wp_inv = self.wp_inv(n0, unit="s")
        efield = m_e * c / (wp_inv * e)
        efield = unit_conv("efield", efield, "N/C", unit)
        return efield

    def bfield(self, n0=None, unit="MG"):
        wp_inv = self.wp_inv(n0, unit="s")
        bfield = m_e / (wp_inv * e)
        bfield = unit_conv("bfield", bfield, "T", unit)
        return bfield

    def charge(self, n0=None, unit="pC"):
        if not n0 == None:
            cwp = self.cwp(n0, unit="cm")
            charge = e * cwp ** 3 * n0
        else:
            cwp = self.cwp(n0, unit="m")
            charge = e * cwp ** 3 * self.n0
        charge = unit_conv("charge", charge, "C", unit)
        return charge

    """

    I don't know if this is right

    def current(self,n0=None,unit='kA'):
        charge=self.charge(n0,unit='C')
        wp_inv=self.wp_inv(n0,unit='s')
        current=charge/wp_inv
        current=unit_conv('current',current,'A',unit)
        return current
    """

    def electron_rest_energy(self, unit="MeV"):
        energy = m_e * c ** 2
        energy = unit_conv("energy", energy, "J", unit)
        return energy

    def rest_energy(self, mass_ratio=False, mass=False, unit="GeV"):
        if not mass_ratio and not mass:
            print("Either mass_ratio (particle to electron mass ratio) or mass (in kg) must be defined. Aborting...")
            sys.exit()
        elif mass_ratio:
            mass = mass_ratio * m_e
        energy = mass * c ** 2
        energy = unit_conv("energy", energy, "J", unit)
        return energy

    def check_n0(self, n0):
        if self.if_n0 and not n0 == None:
            print("Warning. Double definition of n0. Using", n0)
            n0 = np.float128(n0)
            n0 *= 1e6
            return n0
        elif self.if_n0:
            return self.n0
        elif n0 == None:
            print("Value of n0 not defined. Aborting...")
            sys.exit()
        else:
            n0 = np.float128(n0)
            n0 *= 1e6
            return n0

    def check_kwargs(self, args, kwargs):
        self.if_n0 = False
        if args:
            self.n0 = np.float128(args[0])
            self.n0 *= 1e6
            self.if_n0 = True
        if "n0" in kwargs:
            self.n0 = np.float128(kwargs["n0"])
            self.n0 *= 1e6
            self.if_n0 = True
        if "warning" in kwargs and kwargs["warning"] == True:
            print("Warning: n0 should be given in cm^-3.")


class laser_parameters:
    def __init__(self, Wavelength, Waist=[False, False], Fnumber=False, Energy=[False, False], Duration=[False, False], a_0=False, Intensity=[False, False], n0=False):

        # Plasma parameters, if necessary

        if n0:
            self.n0 = True
            self.pp = plasma_parameters(n0)
        else:
            self.n0 = False

        # Wavelength input

        lamb_val = Wavelength[0]
        lamb_uni = Wavelength[1]
        self.lamb_um = unit_conv("length", lamb_val, lamb_uni, "um")

        # Dealing with the beam waist

        if (bool(Waist[0]) == True) & (bool(Fnumber) == True):
            print("Give either the beam waist or the Fnumber, not both")
            sys.exit()
        elif (bool(Waist[0]) == False) & (bool(Fnumber) == True):
            self.FiniteWaist = True
            self.Fnumber = Fnumber
            self.w_0 = self.w_0__from__Fnum()
        elif (bool(Waist[0]) == True) & (bool(Fnumber) == False):
            self.FiniteWaist = True
            w0_val = Waist[0]
            w0_uni = Waist[1]
            self.w_0 = unit_conv("length", w0_val, w0_uni, "um")
        else:
            self.FiniteWaist = False

        # Dealing with the intensity

        self.conv_fac_a0_I0 = 8.549297074506445 * 10 ** (-10)
        if (bool(a_0) == True) & (bool(Intensity[0]) == True):
            print("Give either a_0 or the peak Intensity, not both")
        elif (bool(a_0) == False) & (bool(Intensity[0]) == True):
            self.IntensityGiven = True
            I0_val = Intensity[0]
            I0_uni = Intensity[1]
            self.I_0 = unit_conv("laserintensity", I0_val, I0_uni, "W/cm^2")
            self.a_0 = self.a_0__from__I_0()
        elif (bool(a_0) == True) & (bool(Intensity[0]) == False):
            self.IntensityGiven = True
            self.a_0 = a_0
            self.I_0 = self.I_0__from__a_0()
        else:
            self.IntensityGiven = False

        # Dealing with the energy

        if bool(Energy[0]) == True:
            self.EnergyGiven = True
            ene_val = Energy[0]
            ene_uni = Energy[1]
            self.ener = unit_conv("energy", ene_val, ene_uni, "J")
        else:
            self.EnergyGiven = False
            self.ener = False

        # Dealing with the pulse duration

        if bool(Duration[0]) == True:
            self.DurationGiven = True
            dur_val = Duration[0]
            dur_uni = Duration[1]
            self.dur = unit_conv("time", dur_val, dur_uni, "s")
        else:
            self.DurationGiven = False
            self.dur = False

        if self.DurationGiven & self.EnergyGiven & self.FiniteWaist & self.IntensityGiven:
            print("Too much information given... Aborting.")
            sys.exit()

        if self.DurationGiven & self.EnergyGiven & self.FiniteWaist:
            w0_cm = unit_conv("length", self.w_0, "um", "cm")
            self.I_0 = 2 * self.ener / (np.pi * self.dur * w0_cm ** 2)
            self.a_0 = self.a_0__from__I_0()
            self.IntensityGiven = True

        if self.DurationGiven & self.IntensityGiven & self.FiniteWaist:
            w0_cm = unit_conv("length", self.w_0, "um", "cm")
            self.ener = self.I_0 * self.dur * np.pi * w0_cm ** 2 / 2
            self.EnergyGiven = True

        if self.DurationGiven & self.IntensityGiven & self.EnergyGiven:
            w0_cm = np.sqrt(2 * self.ener / (np.pi * self.dur * self.I_0))
            self.w_0 = unit_conv("length", w0_cm, "cm", "um")
            self.FiniteWaist = True

        if self.FiniteWaist & self.IntensityGiven & self.EnergyGiven:
            w0_cm = unit_conv("length", self.w_0, "um", "cm")
            self.dur = 2 * self.ener / (self.I_0 * np.pi * w0_cm ** 2)
            self.DurationGiven = True

        if self.DurationGiven & self.EnergyGiven:
            self.PowerGiven = True
            p = self.ener / self.dur
            self.power = unit_conv("power", p, "W", "TW")
        elif self.FiniteWaist & self.IntensityGiven:
            self.PowerGiven = True
            w0_cm = unit_conv("length", self.w_0, "um", "cm")
            p = self.I_0 * np.pi * w0_cm ** 2 / 2
            self.power = unit_conv("power", p, "W", "TW")
        else:
            self.PowerGiven = False
            self.power = False

    def w_0__from__Fnum(self):
        return 2 / np.pi * self.lamb_um * self.Fnumber

    def a_0__from__I_0(self):
        return self.conv_fac_a0_I0 * self.lamb_um * np.sqrt(self.I_0)

    def I_0__from__a_0(self):
        return (self.a_0 / (self.conv_fac_a0_I0 * self.lamb_um)) ** 2

    def lamb(self, unit=False):
        if type(unit) == str:
            l = unit_conv("length", self.lamb_um, "um", unit)
        elif self.n0:
            cwp = self.pp.cwp(unit="um")
            l = self.lamb_um / cwp
            unit = "c/omega_p"
        else:
            unit = "um"
            l = unit_conv("length", self.lamb_um, "um", unit)
        return l, unit

    def ome(self, unit=False):
        lamb_m = unit_conv("length", self.lamb_um, "um", "m")
        if type(unit) == str:
            l = unit_conv("frequency", 2 * np.pi * c / lamb_m, "Hz", unit)
        elif self.n0:
            wp_inv = self.pp.wp_inv(unit="s")
            l = 2 * np.pi * c * wp_inv / lamb_m
            unit = "omega_p"
        else:
            unit = "THz"
            l = unit_conv("frequency", 2 * np.pi * c / lamb_m, "Hz", unit)
        return l, unit

    def I0(self, unit="W/cm^2"):
        l = unit_conv("laserintensity", self.I_0, "W/cm^2", unit)
        return l, unit

    def w0(self, unit=False):
        if type(unit) == str:
            l = unit_conv("length", self.w_0, "um", unit)
        elif self.n0:
            cwp = self.pp.cwp(unit="um")
            l = self.w_0 / cwp
            unit = "c/omega_p"
        else:
            unit = "um"
            l = unit_conv("length", self.w_0, "um", unit)
        return l, unit

    def pow(self, unit="TW"):
        l = unit_conv("power", self.power, "TW", unit)
        return l, unit

    def tau(self, unit=False):
        if type(unit) == str:
            l = unit_conv("time", self.dur, "s", unit)
        elif self.n0:
            l = unit_conv("time", self.dur, "s", "fs")
            wp_inv = self.pp.wp_inv(unit="fs")
            l = l / wp_inv
            unit = "1/omega_p"
        else:
            unit = "fs"
            l = unit_conv("time", self.dur, "s", unit)
        return l, unit

    def ene(self, unit="J"):
        l = unit_conv("energy", self.ener, "J", unit)
        return l, unit

    def a0(self):
        return self.a_0


# test=plasma_parameters(n0=1.e17)
# test.charge(unit='nC')

##pp.cwp(unit="um") = c/wp Value in um
##if you have the waist = 10um, for example
##then W_0 [c/wp] = 10um / pp.cwp(unit="um")

lp = laser_parameters([1315, "nm"])
