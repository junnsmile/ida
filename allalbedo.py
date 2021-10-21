"""
read albedo_read.py --rttov albedo
read modis_albedo.py -- modis albedo
merge land (black/white sky) albedo and ocean albedo
author:ida
date:2021-09-24
"""
import numpy as np
import netCDF4 as nc
from albedo import albedo_read, modis_albedo_case
import pandas as pd

def savencprofile(lon,lat,landper,allwhite,allblack,name):
    """
    函数说明：保存nc文件
    """
    new = nc.Dataset(name,'w',format='NETCDF4')
    # 创建维度
    new.createDimension('lon', len(lon))
    new.createDimension('lat', len(lat))
    #创建变量
    new.createVariable("lon", 'f', ("lon"))
    new.createVariable("lat", 'f', ("lat"))

    new.createVariable("walbedo1",'f',("lat","lon"))
    new.createVariable("walbedo2", 'f', ("lat", "lon"))
    new.createVariable("walbedo3", 'f', ("lat", "lon"))
    new.createVariable("walbedo4", 'f', ("lat", "lon"))
    new.createVariable("walbedo5", 'f', ("lat", "lon"))
    new.createVariable("walbedo6", 'f', ("lat", "lon"))

    new.createVariable("balbedo1", 'f', ("lat", "lon"))
    new.createVariable("balbedo2", 'f', ("lat", "lon"))
    new.createVariable("balbedo3", 'f', ("lat", "lon"))
    new.createVariable("balbedo4", 'f', ("lat", "lon"))
    new.createVariable("balbedo5", 'f', ("lat", "lon"))
    new.createVariable("balbedo6", 'f', ("lat", "lon"))

    new.createVariable("landper", 'f', ("lat", "lon"))

    new.variables['lat'][:] = lat
    new.variables['lon'][:] = lon
    new.variables['landper'][:] = landper
    new.variables['walbedo1'][:] = allwhite[0,:,:]
    new.variables['walbedo2'][:] = allwhite[1,:,:]
    new.variables['walbedo3'][:] = allwhite[2,:,:]
    new.variables['walbedo4'][:] = allwhite[3,:,:]
    new.variables['walbedo5'][:] = allwhite[4,:,:]
    new.variables['walbedo6'][:] = allwhite[5,:,:]
    new.variables['balbedo1'][:] = allblack[0,:,:]
    new.variables['balbedo2'][:] = allblack[1,:,:]
    new.variables['balbedo3'][:] = allblack[2,:,:]
    new.variables['balbedo4'][:] = allblack[3,:,:]
    new.variables['balbedo5'][:] = allblack[4,:,:]
    new.variables['balbedo6'][:] = allblack[5,:,:]

    new.close()


modisfile = r'D:\work\data\ertm\modis\albedo\MCD43C1.A2015276.006.2016195105448.hdf'
h8file = r'D:\work\data\ertm\NC_H08_20151003_0600_R21_FLDK.02401_02401.nc'
landfile = r'D:\work\data\ertm\modis\MOD11C2.A2016065.006.2016242200752.hdf'
path = r'D:\work\data\ertm\rttov\all030507_2.dat'
# rtbrdf,rth8albedo,rtmodisa = albedo_read.rttota(path)
h8ab,modisa,wmaf,wh8albeho,land,lon,lat = modis_albedo_case.modisa(modisfile, h8file, landfile)

allwhite = np.zeros([6,401,501],dtype=np.float)
allblack = np.zeros([6,401,501],dtype=np.float)
landper = np.zeros([401,501])
landper[np.where(land>=50)] = 1
landper[np.where(land<50)] = 0

allwhite[0,:,:][np.where(land<50)]= 0.01352
allwhite[1,:,:][np.where(land<50)]= 0.00995
allwhite[2,:,:][np.where(land<50)]= 0.00663
allwhite[3,:,:][np.where(land<50)]= 0.00629
allwhite[4,:,:][np.where(land<50)]= 0.00593
allwhite[5,:,:][np.where(land<50)]= 0.00509

allblack[0,:,:][np.where(land<50)]= 0.01352
allblack[1,:,:][np.where(land<50)]= 0.00995
allblack[2,:,:][np.where(land<50)]= 0.00663
allblack[3,:,:][np.where(land<50)]= 0.00629
allblack[4,:,:][np.where(land<50)]= 0.00593
allblack[5,:,:][np.where(land<50)]= 0.00509

for i in range(6):
    allwhite[i,:,:][np.where(land>=50)]= wh8albeho[i,:,:][np.where(land>=50)]
    tem = pd.DataFrame(allwhite[i,:,:])
    tem0 = tem.interpolate(method='linear', limit_direction='forward', axis=1)
    allwhite[i,:,:] = np.array(tem0.interpolate(method='linear', limit_direction='forward', axis=0))
for i in range(6):
    allblack[i,:,:][np.where(land>=50)] = h8ab[i,:,:][np.where(land>=50)]
    tem = pd.DataFrame(allblack[i,:,:])
    tem0 = tem.interpolate(method='linear', limit_direction='forward', axis=1)
    allblack[i,:,:] = np.array(tem0.interpolate(method='linear', limit_direction='forward', axis=0))
savencprofile(lon,lat,landper,allwhite,allblack,r'D:\work\data\ertm\modis\albedo\2015100306_albedo.nc')
print('finish')