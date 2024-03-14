from utilities.osiris.open import osiris_open_particle_data
import numpy as np

quants=['x1','x2','p1','p2','p3','ene','q']
file='/Volumes/EXT/Thales/Weibel/runs/FiniteSpotSize/2D_spotsize_100.00_w0_10.0_a0_0.20_T_0.0010_20000x20000/MS/RAW/electrons/RAW-electrons-000030.h5'

def filter(quant,ind):
    quant=quant[:]
    quant=quant[ind]
    return quant

attrs,data=osiris_open_particle_data(file,quants)

x1=data[0][:]
ind=np.where((x1<147.5)&(x1>137.5))
x1=x1[ind]
x2=filter(data[1],ind)
p1=filter(data[2],ind)
p2=filter(data[3],ind)
p3=filter(data[4],ind)
ene=filter(data[5],ind)
q=filter(data[6],ind)
ind=np.where((x2<50.0)&(x2>-50.0))
x1=x1[ind]
x2=x2[ind]
p1=p1[ind]
p2=p2[ind]
p3=p3[ind]
ene=ene[ind]
q=q[ind]

