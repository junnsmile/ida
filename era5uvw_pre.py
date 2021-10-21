"""
era5_uvw analysis
author:ida
date:2021-07-21
"""

import netCDF4 as nc
import time
import numpy as np
import time

start = time.time()
path = r'D:\work\data\ertm\uvw'
h2o_path = r'D:\work\data\ertm\profile'
file = '\era5_20160301.nc'
data =nc.Dataset(path +file)
h2o = nc.Dataset(h2o_path+file)
print(data.variables.keys())
print('profile keys is***'+str(h2o.variables.keys()))
u = data.variables['u'][:].data[:,:,80:128,128:176]
v = data.variables['v'][:].data[:,:,80:128,128:176]
w = data.variables['w'][:].data[:,:,80:128,128:176]
r = data.variables['r'][:].data[:,:,80:128,128:176]
plevel = data.variables['level'][:].data
lon = data.variables['longitude'][:].data[128:176]
lat = data.variables['latitude'][:].data[80:128]
h2o = h2o.variables['q'][:].data[:,:,80:128,128:176]

result_lon = np.zeros([24, len(lon), len(lat)], dtype=np.float64)
result_lat = np.zeros([24, len(lon), len(lat)], dtype=np.float64)
result_p = np.zeros([24, len(lon), len(lat)], dtype=np.float64)
for t in range(24):
    print('*******************time is********'+str(t))
    for j in range(len(lon)):
        for i in range(len(lat)):
            intergrals_lon = 0
            intergrals_lat = 0
            intergrals_p = 0
            for p in range(len(plevel)):
                if p == 0:
                    print('*****'+str(p))
                    ww = w[t, p, i, j] * (h2o[t, p + 1, i, j] - h2o[t, p, i, j]) / (plevel[p + 1] - plevel[p])
                    if j==0:
                        uu = u[t, p, i, j] * (h2o[t, p, i , j+1] - h2o[t, p, i, j]) / (lon[j + 1] - lon[j])
                    elif j==len(lon)-1:
                        uu =  u[t, p, i, j] * (h2o[t, p, i, j] - h2o[t, p, i, j-1]) / (lon[j] - lon[j- 1])
                    else:
                        uu =  u[t, p, i, j] * (h2o[t, p, i , j+1] - h2o[t, p, i , j-1]) / (lon[j + 1] - lon[j - 1])
                    if i==0:
                        vv = v[t, p, i, j] * (h2o[t, p, i+1, j] - h2o[t, p, i, j]) / (lat[i + 1] - lat[i])
                    elif  i==len(lat)-1:
                        vv = v[t, p, i, j] * (h2o[t, p, i, j] - h2o[t, p, i-1, j]) / (lat[i] - lat[i - 1])
                    else:
                        vv = v[t, p, i, j] * (h2o[t, p, i+1, j] - h2o[t, p, i-1, j]) / (lat[i + 1] - lat[i - 1])

                    intergrals_lon=intergrals_lon+uu
                    intergrals_lat = intergrals_lat + vv
                    intergrals_p = intergrals_p + ww
                elif p==len(plevel)-1:
                    print('*****' + str(p))
                    ww =  w[t, p, i, j] * (h2o[t, p, i, j] - h2o[t, p - 1, i, j]) / (plevel[p] - plevel[p - 1])
                    if j == 0:
                        uu = u[t, p, i, j] * (h2o[t, p, i , j+1] - h2o[t, p, i, j]) / (lon[j + 1] - lon[j])
                    elif j == len(lat) - 1:
                        uu = u[t, p, i, j] * (h2o[t, p, i, j] - h2o[t, p, i, j-1]) / (lon[j] - lon[j - 1])
                    else:
                        uu = u[t, p, i, j] * (h2o[t, p, i, j+1] - h2o[t, p, i, j-1]) / (
                                    lon[j + 1] - lon[j - 1])
                    if i == 0:
                        vv = v[t, p, i, j] * (h2o[t, p, i+1, j] - h2o[t, p, i, j]) / (lat[i + 1] - lat[i])
                    elif i == len(lon) - 1:
                        vv =  v[t, p, i, j] * (h2o[t, p, i, j] - h2o[t, p, i-1, j]) / (lat[i] - lat[i - 1])
                    else:
                        vv = v[t, p, i, j] * (h2o[t, p, i+1, j] - h2o[t, p, i-1, j]) / (
                                    lat[i + 1] - lat[i - 1])

                    intergrals_lon=intergrals_lon+uu
                    intergrals_lat = intergrals_lat + vv
                    intergrals_p = intergrals_p + ww
                else:
                    ww = w[t, p, i, j] * (h2o[t, p + 1, i, j] - h2o[t, p - 1, i, j]) / (plevel[p + 1] - plevel[p - 1])
                    if j == 0:
                        uu = u[t, p, i, j] * (h2o[t, p, i , j+1] - h2o[t, p, i, j]) / (lon[j + 1] - lon[j])
                    elif j == len(lat) - 1:
                        uu =u[t, p, i, j] * (h2o[t, p, i, j] - h2o[t, p, i , j-1]) / (lon[j] - lon[j - 1])
                    else:
                        uu =  u[t, p, i, j] * (h2o[t, p, i, j+1] - h2o[t, p, i, j-1]) / (
                                    lon[j+ 1] - lon[j - 1])
                    if i == 0:
                        vv = v[t, p, i, j] * (h2o[t, p, i+1, j ] - h2o[t, p, i, j]) / (lat[i + 1] - lat[i])
                    elif i == len(lon) - 1:
                        vv = v[t, p, i, j] * (h2o[t, p, i, j] - h2o[t, p, i-1, j]) / (lat[i] - lat[i - 1])
                    else:
                        vv = v[t, p, i, j] * (h2o[t, p, i+ 1, j ] - h2o[t, p, i-1, j]) / (
                                    lat[i + 1] - lat[i - 1])
                    intergrals_lon=intergrals_lon+uu
                    intergrals_lat = intergrals_lat + vv
                    intergrals_p = intergrals_p + ww
            result_lon[t, i, j] = intergrals_lon
            result_lat[t, i, j] = intergrals_lat
            result_p[t, i, j] = intergrals_p
result = (1/9.8)*(result_lon+result_lat+result_p)
print('*****用时%s分钟***' % ((time.time()-start)/60))
print(result)
