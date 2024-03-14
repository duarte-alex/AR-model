from utilities.osiris.open import filetags, osiris_open_particle_data, osiris_open_grid_data
from utilities.find import find_string_match, find_nearest
from utilities.ask import askfolderexists_create, askfolder_path, askfile_path
from utilities.osiris.save import osiris_save_grid_1d, osiris_save_grid_copy_attrs_and_axis_2d
import progressbar
import numpy as np
import sys
import os
from joblib import Parallel, delayed
from astropy.convolution import convolve, Box2DKernel
"""
"""
class Temperature:
    """
    Temperature definition used

    m_ec^2 T_{ij} = \int f(\mathbf{p}) v_i p_j d^3p / \int f(\mathbf{p}) d^3p

    """
    def __init__(self,folder):
        self.folder=folder
        self.filename,self.tags,self.time=filetags(folder,time=True,start='RAW')
        self.lentags=len(self.tags)
    def T_calc(self,ij,file):
        ij=str(ij)
        i=ij[0]
        j=ij[1]
        quants=['ene','q','p'+i,'p'+j]
        attrs,data=osiris_open_particle_data(file,quants)
        ene=(data[0][:])+1
        q=data[1][:]
        vi=(data[2][:])/ene
        pj=(data[3][:])
        vipj=vi*pj
        q,vipj=self.check_range(self.xrange,'x1',vipj,q,file)
        q,vipj=self.check_range(self.yrange,'x2',vipj,q,file)
        temp=np.average(vipj,weights=q)
        return temp
    def time_vs_T(self,ij,xrange=False,yrange=False):
        self.xrange=xrange
        self.yrange=yrange
        time=self.time
        temperature=np.zeros(self.lentags)
        flag=-1
        pbar=progressbar.ProgressBar()
        for t in pbar(self.tags):
            flag+=1
            file=self.folder+self.filename+t+'.h5'
            temperature[flag]=self.T_calc(ij,file)
        self.save_time_vs_T(ij,time,temperature)
    def save_time_vs_T(self,ij,time,temperature):
        fileout='T'+ij
        if '/MS/' in self.folder:
            sta,end=find_string_match('/MS/',self.folder)
            folderout=self.folder[:end]+'TEMP/TIME/'+fileout+'/'
        else:
            folderout=self.folder+'TEMP/TIME/'+fileout+'/'
        askfolderexists_create(folderout)
        name='T'+ij
        dataset=name
        units='m_ec^2'
        longname='T_{'+ij+'}'
        axisunits='1/\omega_p'
        axisname='t'
        axislong='Time'
        osiris_save_grid_1d(folderout,fileout,temperature,time,name,dataset,units,longname,axisunits,axisname,axislong)
    def T_at_time(self,t,xrange=False,yrange=False):
        self.xrange=xrange
        self.yrange=yrange
        t_ind=self.find_t(t)
        tag=self.tags[t_ind]
        file=self.folder+self.filename+tag+'.h5'
        t11=self.T_calc(11,file)
        t22=self.T_calc(22,file)
        t12=self.T_calc(12,file)
        t21=t12
        tij=np.array([[t11,t12],[t21,t22]])
        eigvalues,eigvectors=np.linalg.eig(tij)
        anisotropy=np.max(eigvalues)/np.min(eigvalues)-1
        pr_axis_angle_1=np.arctan(eigvectors[1,0]/eigvectors[0,0])*180/np.pi
        pr_axis_angle_2=np.arctan(eigvectors[1,1]/eigvectors[0,1])*180/np.pi
        self.t_cold=np.min(eigvalues)
        self.t_hot=np.max(eigvalues)
        self.anisotropy=anisotropy
        return anisotropy, pr_axis_angle_1, pr_axis_angle_2
    def find_t(self,t):
        return find_nearest(np.array(self.time),t)
    def Weibel_growth_rate(self,t=False):
        if t:
            self.T_at_time(t)
        return np.sqrt((self.anisotropy+1)*self.t_cold)
    def check_range(self,x,var_string,vp,q,file):
        if x:
            xmin=x[ 0]
            xmax=x[-1]
            quants=[var_string]
            attrs,data=osiris_open_particle_data(file,quants)
            data=data[0][:]
            ind=np.where((data>xmin)&(data<xmax))
            return q[ind], vp[ind]
        else:
            return q, vp
"""
"""
class AnisotropyCalc:
    def __init__(self,file11,file12,file22):
        self.file11=file11
        self.file12=file12
        self.file22=file22
        self.main()
    def anis_calc(self,t11,t12,t22):
        t_aux=np.sqrt((t11-t22)**2+4*t12**2)
        t_hot=0.5*(t11+t22+t_aux)
        t_cold=0.5*(t11+t22-t_aux)
        anis=2*t_aux/(1e-5+t11+t22-t_aux)
        #if np.abs(t12)<1e-6:
        #    if t11>t22:
        #        hot_dir=0
        #    else:
        #        hot_dir=np.pi/2
        #else:
        #    hot_dir=np.arctan(2*t12/(t11-t22+t_aux))
        #hot_dir=np.arctan(np.sqrt((t22-t11+t_aux+1e-10)/(t11-t22+t_aux+1e-10)))
        hot_dir=np.arctan(2*t12/(t11-t22+t_aux+1e-6))
        anis_x=anis*np.cos(hot_dir)
        anis_y=anis*np.sin(hot_dir)
        return anis,anis_x,anis_y,hot_dir,t_hot,t_cold
    def main(self):
        a11,ax11,data11=osiris_open_grid_data(self.file11)
        a12,ax12,data12=osiris_open_grid_data(self.file12)
        a22,ax22,data22=osiris_open_grid_data(self.file22)
        if not data11.shape==data12.shape or not data11.shape==data22.shape:
            print('The shape os the datasets is not equal. Aborting...')
            sys.exit()
        t11=data11[:]
        t12=data12[:]
        t22=data22[:]
        self.anisotropy,self.anisotropy_x,self.anisotropy_y,self.hot_dir,self.t_hot,self.t_cold=self.anis_calc(t11,t12,t22)
        self.ax=ax11
        self.attrs=a11
def Anisotropy(file11,file12,file22):
    a=AnisotropyCalc(file11,file12,file22)
    return a.attrs,a.ax,a.anisotropy,a.anisotropy_x,a.anisotropy_y,a.hot_dir,a.t_hot,a.t_cold
"""
"""
class TemperatureGrid:
    def __init__(self,region_str,bin=3,file=False):
        if file:
            self.file=file
        else:
            self.file=askfile_path(initialdir='/Volumes/EXT/Thales/Weibel/runs',title='Choose a file')
        self.quants=['x1','x2','p1','p2']
        self.region=region_str
        self.filter_init()
        minx1=np.floor(np.min(self.x1))
        maxx1=np.ceil(np.max(self.x1))
        minx2=np.floor(np.min(self.x2))
        maxx2=np.ceil(np.max(self.x2))
        self.x1range=np.arange(minx1,maxx1+bin*1e-10,bin)
        self.x2range=np.arange(minx2,maxx2+bin*1e-10,bin)
        i=0
        self.mainloop(i)
        #r=Parallel(n_jobs=3)(delayed(self.mainloop)(i) for i in range(len(self.x1range)-1))
    def filter_init(self):
        attrs,data=osiris_open_particle_data(self.file,['x1','x2'])
        x1=data[0][:]
        x2=data[1][:]
        ind=eval('np.where'+self.region)
        data=self.filter(ind)
        self.x1=data[0]
        self.x2=data[1]
        self.p1=data[2]
        self.p2=data[3]
    def filter(self,ind):
        attrs,data=osiris_open_particle_data(self.file,self.quants)
        for i in range(len(data)):
            d=data[i][:]
            d=d[ind]
            data[i]=d
        return data
    def mainloop(self,i):
        anis=0
        flag=0
        for j in range(len(self.x2range)-1):
            ind=np.where((self.x1>self.x1range[i])&(self.x1<self.x1range[i+1])&(self.x2>self.x2range[j])&(self.x2<self.x2range[j+1]))
            p1=self.p1[ind]
            p2=self.p2[ind]
            meanp1=np.mean(p1)
            meanp2=np.mean(p2)
            p1-=meanp1
            p2-=meanp2
            t11=np.mean(p1**2)
            t22=np.mean(p2**2)
            t12=np.mean(p1*p2)
            t_aux=np.sqrt((t11-t22)**2+4*t12**2)
            anis_t=2*t_aux/(1e-5+t11+t22-t_aux)
            anis+=anis_t
            flag+=1
        print(anis/flag)
"""
"""
#region_str='((x1<242.0)&(x1>236.0)&(x2<20.0)&(x2>-20.0))'
#temp=TemperatureGrid(region_str,file='/Volumes/EXT/Thales/Weibel/runs/FiniteSpotSize/2D_spotsize_100.00_w0_10.0_a0_0.20_T_0.0010_20000x20000/MS/RAW/electrons/RAW-electrons-000030.h5')