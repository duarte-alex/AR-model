from utilities.find import find_string_match
from utilities.path import ospathup
from utilities.ask import askfolderexists_create
import numpy as np
from utilities.osiris.save import osiris_save_grid


def osiris_total_energy_1species(filefld, filepar):
    iter, time, b1, b2, b3, e1, e2, e3 = np.loadtxt(filefld, skiprows=2, unpack=True)
    iter, time, npar, kin = np.loadtxt(filepar, skiprows=2, unpack=True)
    if "/HIST/" in filefld:
        sta, end = find_string_match("HIST/", filefld)
        askfolderexists_create(filefld[:sta] + "MS/")
        folderout = filefld[:sta] + "MS/ENE/"
    else:
        folderout = ospathup(filefld) + "ENE/"
    askfolderexists_create(folderout)
    data = b1 + b2 + b3 + e1 + e2 + e3 + kin
    filename = "total_ene"
    comps = "total"
    long_name = "\epsilon_{total}"
    attrsdata = {"UNITS": "m_ec^2", "LONG_NAME": long_name}
    axattr = {"UNITS": b"1/\\omega_p", "NAME": "t", "LONG_NAME": "Time"}
    osiris_save_grid(
        folder=folderout,
        filename=filename,
        data=data,
        dataset_name=long_name,
        data_attrs=attrsdata,
        axis1=time,
        ax1attrs=axattr,
    )


filepar = "/Volumes/EXT/Thales/NLPW/Simulations/Weibel/Flattened/T_01.00keV__klD_0.3__du_2.0/HIST/par01_ene"
filefld = (
    "/Volumes/EXT/Thales/NLPW/Simulations/Weibel/Flattened/T_01.00keV__klD_0.3__du_2.0/HIST/fld_ene"
)

osiris_total_energy_1species(filefld, filepar)
