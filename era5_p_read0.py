"""
era5 profile data read
481*481 trans 6001*6001
save as txt (o3,h2o,plevel,tlevel )
author:ida
date:2021-07-05
"""
import time
import netCDF4 as nc
from scipy.interpolate import griddata
import numpy as np
import os

def interpola(o3,output,points,x,y,namee):
    data = np.zeros([np.shape(o3)[1], len(x), len(y)], dtype=np.float64)
    for t in range(np.shape(o3)[0]):
        for p in range(np.shape(o3)[1]):
            data[p,:,:] = griddata(points, o3[t][p].flatten(), (output[0], output[1]), method='nearest')
        temp = np.reshape(data, (np.shape(data)[0],np.shape(data)[1]*np.shape(data)[2]), order='F')
        # name = open("D:\work\code\ertm\o3_" + str(t) + '.txt', 'w')
        # for j in range(len(x)):
        #     for i in range(len(y)):
        #         for pp in range(np.shape(o3)[1]):
        #             name.write(str(data[pp,i,j])+'\n')
        #         name.write('\t\t\t\t')
        #         name.write('\n')
        # name.close()
        name = 'D:\work\code\ertm'+namee + str(t) +'.txt'
        np.savetxt(name, np.c_[temp],fmt='%f', delimiter='\t')
        # np.savetxt(name,temp,delimiter = ',')

    print(data.shape)
    print(data.max())
    print(data.min())

    return data

def ncreadERA5Profile(filename,xshape,yshape,file):
    '''
    @description: read era5 profile o3
    '''
    start = time.time()
    ds = nc.Dataset(filename)
    print(ds.variables.keys())
    # 数据经纬度
    lon = ds.variables['longitude'][:]
    lat = ds.variables['latitude'][:]
    # 6001格网经纬度
    xsize = (lon[-1]-lon[0])/xshape
    ysize = (lat[-1]-lat[0])/yshape
    # 需要矩形经纬度分布
    x = np.arange(lon[128],lon[176],xsize)
    y =np.arange(lat[80],lat[128],ysize)
    output = np.meshgrid(x,y)
    lonn,latt = np.meshgrid(np.array(lon[128:176]),np.array(lat[80:128]))
    points = np.array((lonn.flatten(), latt.flatten())).T
    o3 = ds.variables['o3'][:].data
    o33 = o3[:,:,128:176,80:128]
    o3off = getattr(ds['o3'], 'add_offset')
    o3scale= getattr(ds['o3'], 'scale_factor')
    o333 = o33*o3scale+o3off
    oo3 = interpola(o333,output,points,x,y,'\\'+file+'_o3_')
    print('o3计算完成')
    q = ds.variables['q'][:].data
    qq = q[:,:,128:176,80:128]
    qoff = getattr(ds['q'], 'add_offset')
    qscale= getattr(ds['q'], 'scale_factor')
    qq0 = qq*qscale+qoff
    qqq = interpola(qq0,output,points,x,y,'\\'+file+'_q_')
    print('h2o 计算完成')
    t = ds.variables['t'][:].data
    tt = t[:,:,128:176,80:128]
    toff = getattr(ds['t'], 'add_offset')
    tscale= getattr(ds['t'], 'scale_factor')
    tt0 = tt*tscale+toff
    ttt = interpola(tt0,output,points,x,y,'\\'+file+'_tlevel_')
    print('tlevel 计算完成')
    level = ds.variables['level'][:].data
    a = np.reshape(level, (37, 1))
    name = 'D:\work\code\ertm'+'\\'+file+'_plevel.txt'
    np.savetxt(name, a, fmt='%i', delimiter='\t')
    # np.savetxt(name, a, fmt='%i',delimiter=',')
    print('plevel 计算完成')

    end = time.time()
    print('Running time: %s Seconds' % (end - start))
    # return o33, qq, tt

if __name__ == "__main__":
    start= time.time()

    filename = r'D:\VS2010\projects\ERTM_nc_code\ERTM_nc_code\profile.nc'
    # file = (os.path.split(filename)[1].split('.nc'))[0].split('_')[1]
    xshape = 6001
    yshape = 6001
    ncreadERA5Profile(filename,xshape,yshape,'h')

