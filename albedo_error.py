"""
read albedo_read.py --rttov albedo
read modis_albedo.py -- modis albedo
RESULT ANALYSIS AND ERROR CALCULATE
author:ida
date:2021-09-18
"""

from albedo import albedo_read
import modis_albedo
import numpy as np
from math import sqrt

def rmse(error):
    # local = np.where(error!=np.nan)

    squarederror = (error*error)
    rmse = sqrt(np.nansum(squarederror) / (len(error[~np.isnan(error)])))
    mae = (np.nansum(abs(error)))/ len(error[~np.isnan(error)])
    mean = (np.nansum(error))/ len(error[~np.isnan(error)])
    max = np.nanmax(error)
    min = np.nanmin(error)
    return rmse,mean,mae,max,min


modisfile = r'D:\work\data\ertm\modis\albedo\MCD43C1.A2016065.006.2016196185546.hdf'
h8file = r'D:\work\data\ertm\NC_H08_20160305_0700_R21_FLDK.02401_02401.nc'
landfile = r'D:\work\data\ertm\modis\MOD11C2.A2016065.006.2016242200752.hdf'
path = r'D:\work\data\ertm\rttov\all030507_2.dat'
rtbrdf,rth8albedo,rtmodisa = albedo_read.rttota(path)
h8ab,modisa,wmaf,wh8albeho,land = modis_albedo.modisa(modisfile, h8file, landfile)
rmsem = np.zeros([7],dtype=np.float)
meanm = np.zeros([7],dtype=np.float)
maxm = np.zeros([7],dtype=np.float)
maem = np.zeros([7],dtype=np.float)
minm = np.zeros([7],dtype=np.float)

rmseh = np.zeros([6],dtype=np.float)
meanh = np.zeros([6],dtype=np.float)
maxh = np.zeros([6],dtype=np.float)
maeh = np.zeros([6],dtype=np.float)
minh = np.zeros([6],dtype=np.float)
for i in range(6):
    h8albedo = rth8albedo[:,:,i]-h8ab[i,:,:]
    rmseh[i],meanh[i],maeh[i],maxh[i],minh[i] = rmse(h8albedo)

for j in range(7):
    malbeodo = rtmodisa[:,:,j]-modisa[j,:,:]
    rmsem[j], meanm[j], maem[j], maxm[j], minm[j] = rmse(malbeodo)

print(rmseh,maeh,rmsem,maem)


