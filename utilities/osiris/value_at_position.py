from utilities.osiris.open import filetags, osiris_open_grid_data
from utilities.osiris.save import osiris_save_grid_1d
from utilities.find import find_nearest
from utilities.ask import askfolderexists_create, askexists_skip
import numpy as np
import os
import progressbar
import sys

class value2D:
    def __init__(self,folder,x_1,x_2):
        self.folder=folder
        self.filename,self.tags=filetags(folder)
        self.folderout=os.path.join(folder,'VALUE')
        askfolderexists_create(self.folderout)
        file_0=folder+self.filename+self.tags[0]+'.h5'
        attrs,axis,data=osiris_open_grid_data(file_0)
        self.attrs=attrs
        self.unitsdata=(data.attrs['UNITS'][0]).decode('utf-8')
        self.longname=(data.attrs['LONG_NAME'][0]).decode('utf-8')
        ax1=axis[0]
        ax2=axis[1]
        x1=np.linspace(ax1[0],ax1[1],data.shape[-1],endpoint=False);dx1=x1[1]-x1[0];x1+=dx1/2
        x2=np.linspace(ax2[0],ax2[1],data.shape[ 0],endpoint=False);dx2=x2[1]-x2[0];x2+=dx2/2
        self.indx1=find_nearest(x1,x_1)
        self.indx2=find_nearest(x2,x_2)
        self.x1str='{0:08.3f}'.format(x_1)
        self.x2str='{0:08.3f}'.format(x_2)
        self.x1str_='{0:0.2f}'.format(x_1)
        self.x2str_='{0:0.2f}'.format(x_2)
        lentags=len(self.tags)
        self.time=np.zeros(lentags)
        self.value=np.zeros(lentags)
        self.mainloop()
        self.value=np.abs(self.value)
        self.save()
    def mainloop(self):
        flag=-1
        pbar=progressbar.ProgressBar()
        for t in pbar(self.tags):
            flag+=1
            file=self.folder+self.filename+t+'.h5'
            attrs,axis,data=osiris_open_grid_data(file)
            self.time[flag]=attrs['TIME']
            data=data[:]
            self.value[flag]=data[self.indx2,self.indx1]
    def save(self):
        folderout=self.folderout
        fileout='x1_'+self.x1str+'x2_'+self.x2str
        name=(self.attrs['NAME'][0]).decode('utf-8')+'@x1='+self.x1str+',x2='+self.x2str
        dataset=name
        units=self.unitsdata
        longname='|'+self.longname+'_{,@x1='+self.x1str_+',x2='+self.x2str_+'}|'
        axisunits='1/\omega_p'
        axisname='t'
        axislong='Time'
        if askexists_skip(folderout+fileout+'.h5'):
            os.remove(folderout+fileout+'.h5')
        osiris_save_grid_1d(folderout,fileout,self.value,self.time,name,dataset,units,longname,axisunits,axisname,axislong)

folder='/Volumes/EXT/Thales/Weibel/runs/FiniteSpotSize/2D_spotsize_050.00_w0_10.0_a0_0.20_T_0.0010_10000x20000/MS/UDIST/electrons/T22-savg/'
o=value2D(folder,x_1=69.6,x_2=1.6)

