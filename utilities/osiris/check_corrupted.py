from utilities.ask import check_file_or_folder
from utilities.osiris.open import osiris_open_grid_data, osiris_open_particle_data
import os
from tqdm import tqdm
import multiprocessing as mp

mp.set_start_method("fork")

folder = "/Volumes/Drobo/Fireball/Simulations/Beam/quasi3D/"
# folder = "/Volumes/Drobo/Fireball/Simulations/Beam/quasi3D/HollowChannel/Radius/02.00/MS/RAW/pbeam/"
folder = "/Volumes/Drobo/Maser/Simulations/RingGeneration/all/10000nm/"
res = check_file_or_folder(folder)

if res == "folder":
    rawlist = []
    gridlist = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".h5"):
                if "RAW-" not in file:
                    gridlist.append(os.path.join(root, file))
                else:
                    rawlist.append(os.path.join(root, file))

raw_corrupt = []
for f in tqdm(rawlist):
    try:
        test = osiris_open_particle_data(f, quants=["p1"])
    except OSError or AttributeError:
        raw_corrupt.append(f)


print(raw_corrupt)

grid_corrupt = []
for f in tqdm(gridlist):
    try:
        test = osiris_open_grid_data(f)
    except OSError or AttributeError:
        grid_corrupt.append(f)
        print(f)


print(grid_corrupt)
