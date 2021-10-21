"""
ERA5_U V wind  interpolation
time read
author:ida
date:2021-09-22
"""

import netCDF4 as nc
import numpy as np
from scipy.interpolate import griddata

def interpola(o3,output,points,x,y,t):
    """
    函数说明：profile插值
    """
    data = np.zeros([len(x), len(y)], dtype=np.float64)
    # for t in range(np.shape(o3)[0]):

    data[:,:] = griddata(points, o3[t,:,:].flatten(), (output[0], output[1]), method='nearest')

    print(data.shape)
    #print(data.max())
    #print(data.min())
    return data

def savencprofile(u,lon,lat,name):
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
    new.createVariable("u10", 'f', ("lat", "lon"))


    new.variables['lat'][:] = lat
    new.variables['lon'][:] = lon
    new.variables['u10'][:] = u

    new.close()

xshape = 2401
yshape = 2401
ds =nc.Dataset(r'D:\work\data\ertm\uvw\era5_20160305.nc')
u10 =ds.variables['u10'][:].data
lon = ds.variables['longitude'][:].data
lat = ds.variables['latitude'][:].data
# 6001每个格网经纬度
ysize = (lon[-1] - lon[0]) / xshape
xsize = (lat[-1] - lat[0]) / yshape
# 需要矩形经纬度分布
y = np.arange(lon[128], lon[176], ysize)
x = np.arange(lat[80], lat[128], xsize)
output = np.meshgrid(x, y)
# 现存网格经纬度
latt, lonn = np.meshgrid(np.array(lat[80:128]), np.array(lon[128:176]))
# 现存网格点个数
points = np.array((latt.flatten(), lonn.flatten())).T
u10p = interpola(u10[ :,80:128, 128:176],output,points,x,y,7)
savencprofile(u10p, y, x, r'D:\work\data\ertm\uvw\2016030507.nc')