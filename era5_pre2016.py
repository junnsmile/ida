"""
era5 profile data read
srf data read
481*481 trans 6001*6001
save as nc (o3,h2o,plevel,tlevel,sp,ts )
only one time
author:ida
date:2021-07-05
"""
import time
import netCDF4 as nc
from scipy.interpolate import griddata
import numpy as np
import os
import datetime

def interpola(o3,output,points,x,y,t):
    """
    函数说明：profile插值
    """
    data = np.zeros([np.shape(o3)[1], len(x), len(y)], dtype=np.float64)
    # for t in range(np.shape(o3)[0]):
    for p in range(np.shape(o3)[1]):
        data[p,:,:] = griddata(points, o3[t][p].flatten(), (output[0], output[1]), method='nearest')

    print(data.shape)
    #print(data.max())
    #print(data.min())
    return data

def srfinterpol(o3,output,points,x,y,t):
    """
    函数说明：srf插值
    """
    data = np.zeros([len(x), len(y)], dtype=np.float64)
    # for t in range(np.shape(o3)[0]):
    data[:,:] = griddata(points, o3[t].flatten(), (output[0], output[1]), method='nearest')
    print(data.shape)
    print(data.max())
    print(data.min())
    return data

def ncreadERA5Profile(filename,xshape,yshape,file,timee):
    '''
    函数说明: read era5 profile o3
    '''
    start = time.time()
    ds = nc.Dataset(filename)
    print(ds.variables.keys())
    # 数据经纬度
    lon = ds.variables['longitude'][:]
    lat = ds.variables['latitude'][:]
    # 6001每个格网经纬度
    ysize = (lon[-1]-lon[0])/xshape
    xsize = (lat[-1]-lat[0])/yshape
    # 需要矩形经纬度分布
    y = np.arange(lon[128],lon[176],ysize)
    x =np.arange(lat[80],lat[128],xsize)
    output = np.meshgrid(x,y)
    #现存网格经纬度
    latt,lonn = np.meshgrid(np.array(lat[80:128]),np.array(lon[128:176]))
    #现存网格点个数
    points = np.array((latt.flatten(), lonn.flatten())).T

    #读取插值
    o3 = ds.variables['o3'][:].data
    o33 = o3[:,:,80:128,128:176]
    # o3off = getattr(ds['o3'], 'add_offset')
    # o3scale= getattr(ds['o3'], 'scale_factor')
    # o333 = o33*o3scale+o3off
    oo3 = interpola(o33,output,points,x,y,timee)
    print('o3计算完成')
    q = ds.variables['q'][:].data
    qq = q[:,:,80:128,128:176]
    qqq = interpola(qq,output,points,x,y,timee)
    print('h2o 计算完成')
    t = ds.variables['t'][:].data
    tt = t[:,:,80:128,128:176]
    ttt = interpola(tt,output,points,x,y,timee)
    print('tlevel 计算完成')
    level = ds.variables['level'][:].data
    print('plevel 计算完成')

    end = time.time()
    print('Running time: %s Seconds' % (end - start))
    return oo3, qqq, ttt,level,x,y

def ncreadERA5SRF(srf,xshape,yshape,file,timee):
    '''
    函数说明: read era5 srf data
    '''
    start = time.time()
    ds = nc.Dataset(srf)
    print(ds.variables.keys())
    # 数据经纬度
    lon = ds.variables['longitude'][:]
    lat = ds.variables['latitude'][:]
    # 6001每个格网经纬度
    ysize = (lon[-1] - lon[0]) / xshape
    xsize = (lat[-1] - lat[0]) / yshape
    # 需要矩形经纬度分布
    y = np.arange(lon[128], lon[176], ysize)
    x = np.arange(lat[80], lat[128], xsize)
    output = np.meshgrid(x, y)
    # 现存网格经纬度
    lonn, latt = np.meshgrid(np.array(lon[128:176]), np.array(lat[80:128]))
    # 现存网格点个数
    points = np.array((latt.flatten(), lonn.flatten())).T

    # 读取插值
    skt = ds.variables['skt'][:].data
    ts = skt[ :,80:128, 128:176]
    tss = srfinterpol(ts, output, points, x, y, timee)
    print('skt计算完成')

    sp = ds.variables['sp'][:].data
    spp = sp[ :,80:128, 128:176]
    spp0 = srfinterpol(spp, output, points, x, y,timee)
    print('sp 计算完成')

    end = time.time()
    print('Running time: %s Seconds' % (end - start))
    return spp0,tss



def savencprofile(oo3, qqq, ttt,level,lon,lat,sp,ts,name):
    """
    函数说明：保存nc文件
    """
    new = nc.Dataset(name,'w',format='NETCDF4')
    # 创建维度
    new.createDimension('lev', len(level))
    new.createDimension('lon', len(lon))
    new.createDimension('lat', len(lat))
    #创建变量
    new.createVariable("lev", 'f', ("lev"))
    new.createVariable("lon", 'f', ("lon"))
    new.createVariable("lat", 'f', ("lat"))

    new.createVariable("t",'f',("lev","lat","lon"))
    new.createVariable("o3", 'f', ("lev", "lat", "lon"))
    new.createVariable("q", 'f', ("lev", "lat", "lon"))
    new.createVariable("sp", 'f', ("lat", "lon"))
    new.createVariable("ts", 'f', ("lat", "lon"))

    new.variables['lat'][:] = lat
    new.variables['lon'][:] = lon
    new.variables['lev'][:] = level
    new.variables['t'][:] = ttt
    new.variables['o3'][:] = oo3
    new.variables['q'][:] = qqq
    new.variables['sp'][:] = sp
    new.variables['ts'][:] = ts

    new.close()

if __name__ == "__main__":
    start = time.time()
    startdate = datetime.datetime.strptime('20170101', '%Y%m%d')
    enddate = datetime.datetime.strptime('20171231', '%Y%m%d')
    path = r'D:\work\data\ertm'
    profilepath = path + '\\profile'
    srfpath = path + '\\srf'
    profiles = os.listdir(profilepath)
    srfs = os.listdir(srfpath)
    for files in profiles:
        profile = profilepath + '\\' + files
        file = (os.path.split(profile)[1].split('.nc'))[0].split('_')[1]
        sd = datetime.datetime.strptime(file, '%Y%m%d')
        if (sd - startdate).days >= 0 and (sd - enddate).days <= 0:
            srf = srfpath + '\\' + 'era5_sf' + file + '.nc'
            if os.path.exists(srf) == True:
                xshape = 2401
                yshape = 2401
                for i in range(24):
                    start0 = time.time()
                    # profile插值
                    oo3, qqq, ttt, level, lat, lon = ncreadERA5Profile(profile, xshape, yshape, file, i)
                    # srf插值
                    sp, ts = ncreadERA5SRF(srf, xshape, yshape, file, i)
                    ii = str(i).zfill(2)
                    name = r'D:\work\data\ertm\20160301' + str(ii) + '.nc'
                    # 保存nc
                    savencprofile(oo3, qqq, ttt, level, lon, lat, sp, ts, name)
                    print(str(name) + '**complete!')
                    end0 = time.time()
                    print(str(name) + '**Running time: %s Seconds' % (end0 - start0))
                end = time.time()
                print('complete!!!Running time: %s Seconds' % (end - start))
            else:
                continue
        else:
            continue
