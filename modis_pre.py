"""
MODIS DATA ANALYSIS

author:ida
date:2021-07-21
"""

import netCDF4 as nc
from pyhdf.SD import SD
import numpy as np
from scipy.interpolate import griddata
import skimage.transform

def savencprofile(lon,lat,ts,name):
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
    new.createVariable("ts", 'f', ("lat", "lon"))

    new.variables['lat'][:] = lat
    new.variables['lon'][:] = lon
    new.variables['ts'][:] = ts

    new.close()


path = r'D:\work\data\ertm\modis'
file_d = '\\TERRA_MODIS.20160711_20160718.L3m.8D.SST.sst.4km.nc'
file_n = '\\TERRA_MODIS.20160711_20160718.L3m.8D.SST.sst.4km.nc'
file2 = '\\MOD11C2.A2016193.006.2016243201734.hdf'


data_d = nc.Dataset(path+file_d)
print(data_d.variables.keys())
data_n = nc.Dataset(path+file_n)
print(data_n.variables.keys())

LST = SD(path+file2)
ds_dict = LST.datasets()
for idx, sds in enumerate(ds_dict.keys()):
    print(idx, sds)

lst_DAY = LST.select('LST_Day_CMG')
daytime = LST.select('Day_view_time')
lst_NIGHT = LST.select('LST_Night_CMG')
nighttime = LST.select('Night_view_time')
land = LST.select('Percent_land_in_grid').get()

lstd= lst_DAY.get()*(lst_DAY.attributes()['scale_factor'])
dt= daytime.get()*(daytime.attributes()['scale_factor'])
lstn= lst_NIGHT.get()*(lst_NIGHT.attributes()['scale_factor'])
nt= nighttime.get()*(nighttime.attributes()['scale_factor'])

# lst = (lstd+lstn)/2
# 选择白天黑夜
lst = lstd
lstinput = lst[1000:1241,5840:6081]
#陆地判断
landinput = land[1000:1241,5840:6081]
SST_d= data_d.variables['sst'][:].data
lon_d = data_d.variables['lon'][:].data
lat_d = data_d.variables['lat'][:].data

SST_n= data_n.variables['sst'][:].data
lon_n = data_n.variables['lon'][:].data
lat_n = data_n.variables['lat'][:].data

input_d = SST_d[1199:1488,7008:7296]+273.15

ysize = (lon_d[-1] - lon_d[0]) / 7200
xsize = (lat_d[-1] - lat_d[0]) / 3600
# 需要矩形经纬度分布
y = np.arange(lon_d[7008], lon_d[7296], ysize)
x = np.arange(lat_d[1199], lat_d[1488], xsize)
output = np.meshgrid(x, y)
# 现存网格经纬度
lonn, latt = np.meshgrid(np.array(lon_d[7008:7296]), np.array(lat_d[1199:1488]))
# 现存网格点个数
points = np.array((lonn.flatten(), latt.flatten())).T

sstinput  = np.zeros([np.shape(lstinput)[0], np.shape(lstinput)[1]], dtype=np.float32)
# sstinput[:,:] = griddata(points, input_d.flatten(), (output[0], output[1]), method='nearest')
sstinput = skimage.transform.resize(input_d,np.shape(lstinput),order=0)


# input[np.where(input<0)]=np.nan
tem = lstinput
# 拼接
lstinput[np.where(landinput<50)]=sstinput[np.where(landinput<50)]

name = r'D:\work\data\ertm\modis\20160711.nc'
savencprofile(y,x,lstinput,name)
print(lstinput)

# file22 = '\\MOD35_L2.A2016245.0135.061.2017327144449.hdf'
# file3 = '\\MOD06_L2.A2016245.0135.061.2017328015115.hdf'
# f3 =SD(path+file3)
# ds_dict = f3.datasets()
# for idx, sds in enumerate(ds_dict.keys()):
#     print(idx, sds)
# # print(f3.attributes())
#
# f2 = SD(path+file22)
# ds_dict = f2.datasets()
# for idx, sds in enumerate(ds_dict.keys()):
#     print(idx, sds)
# # print(f2.attributes())
# cm = f2.select('Cloud_Mask')
# cma = cm.attributes()
# cmd = cm.get()
# cms = f2.select('Cloud_Mask_SPI')
# cmsa = cms.attributes()
# cmsd = cms.get()