"""
FY-4-RTTOV data preprocesing
ERA5-PROFILE,SRF,10MT2MT ANALYSIS INTERPLATION

author：ida
date:2021-08-25
"""

from collections import Counter
import numpy as np
import pandas as pd
import netCDF4 as nc
from scipy import interpolate
from scipy.interpolate import griddata
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool
import time

def savencprofile(oo3, qqq, ttt,level,lon,lat,sp,ts,name,u10,v10,t2m,saz,saa):
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

    # new.createVariable("t",'f',("lev","lat","lon"))
    # new.createVariable("o3", 'f', ("lev", "lat", "lon"))
    # new.createVariable("q", 'f', ("lev", "lat", "lon"))
    # new.createVariable("sp", 'f', ("lat", "lon"))
    # new.createVariable("ts", 'f', ("lat", "lon"))
    # new.createVariable("u10", 'f', ("lat", "lon"))
    # new.createVariable("v10", 'f', ("lat", "lon"))
    # new.createVariable("t2m", 'f', ("lat", "lon"))

    new.createVariable("t",'f',("lon","lev"))
    new.createVariable("o3", 'f', ("lon","lev"))
    new.createVariable("q", 'f', ("lon","lev"))
    new.createVariable("sp", 'f', ("lon"))
    new.createVariable("ts", 'f', ("lon"))
    new.createVariable("u10", 'f', ("lon"))
    new.createVariable("v10", 'f', ("lon"))
    new.createVariable("t2m", 'f', ("lon"))
    new.createVariable("saz", 'f', ("lon"))
    new.createVariable("saa", 'f', ("lon"))

    new.variables['lat'][:] = lat
    new.variables['lon'][:] = lon
    new.variables['lev'][:] = level
    new.variables['t'][:] = ttt
    new.variables['o3'][:] = oo3
    new.variables['q'][:] = qqq
    new.variables['sp'][:] = sp
    new.variables['ts'][:] = ts
    new.variables['u10'][:] = u10
    new.variables['v10'][:] = v10
    new.variables['t2m'][:] = t2m
    new.variables['saz'][:] = saz
    new.variables['saa'][:] = saa

    new.close()


def get_nearest_point_3d(lon,lat,lons,lats,data,time):
    """
    函数说明：找三维数组（包含时间）最邻近坐标的值
    """
    chalon = lon-lons
    chalat = lat-lats
    lon_index = np.where(abs(chalon) == min(abs(chalon)))
    lat_index = np.where(abs(chalat) == min(abs(chalat)))
    value = data[time,lat_index,lon_index]
    return value

def get_nearest_point_4d(lon,lat,lons,lats,data,time,plevel):
    """
    函数说明：找四维数组（包含时间，大气层）最邻近坐标的值
    """
    chalon = lon-lons
    chalat = lat-lats
    lon_index = np.where(abs(chalon) == min(abs(chalon)))
    lat_index = np.where(abs(chalat) == min(abs(chalat)))
    value = data[time,plevel,lat_index,lon_index]
    return value


def proreadnc(data,lon,lat,t):
    file = nc.Dataset(data)
    #print(file.variables.keys())
    longitude = file.variables['longitude'][:].data
    latitude = file.variables['latitude'][:].data
    o3 = file.variables['o3'][:].data
    q = file.variables['q'][:].data
    pt = file.variables['t'][:].data
    level = file.variables['level'][:].data
    o33 =np.zeros([37],dtype=np.float64)
    h2o = np.zeros([37], dtype=np.float64)
    tt = np.zeros([37], dtype=np.float64)
    for j in range(37):
        o33[j] = get_nearest_point_4d(lon, lat, longitude, latitude, o3, t,j)
        h2o[j] = get_nearest_point_4d(lon, lat, longitude, latitude, q, t,j)
        tt[j] = get_nearest_point_4d(lon, lat, longitude, latitude, pt, t, j)
    return o33,h2o,tt,level

def srfreadnc(data,lon,lat,t):
    file = nc.Dataset(data)
    #print(file.variables.keys())
    longitude = file.variables['longitude'][:].data
    latitude = file.variables['latitude'][:].data
    skt = file.variables['skt'][:].data
    sp = file.variables['sp'][:].data
    # nearest 插值到点
    sktt = get_nearest_point_3d(lon,lat,longitude,latitude,skt,t)
    spp = get_nearest_point_3d(lon,lat,longitude,latitude,sp,t)

    return sktt,spp

def tpreadnc(data,lon,lat,t):
    file = nc.Dataset(data)
    #print(file.variables.keys())
    longitude = file.variables['longitude'][:].data
    latitude = file.variables['latitude'][:].data
    u10 = file.variables['u10'][:].data
    v10 = file.variables['v10'][:].data
    t2m = file.variables['t2m'][:].data

    # nearest 插值到点
    u10m = get_nearest_point_3d(lon,lat,longitude,latitude,u10,t)
    v10m = get_nearest_point_3d(lon,lat,longitude,latitude,v10,t)
    t2mm = get_nearest_point_3d(lon, lat, longitude, latitude,t2m, t)
    return u10m,v10m,t2mm

def mul(file):
    start = time.time()
    print('*****Processing point numbers are*' + str(file))
    fypoints = pd.read_csv(r'/home/data/FY_ERA5/fy_position_azimuth.csv')
    # 年 月 日 小时
    year = str(file)[0:4]
    month = str(file)[4:6]
    day = str(file)[6:8]
    hour = str(file)[8:10]
    # 提取当前日期的所有经纬度坐标
    tem = fypoints.loc[fypoints['eratime'] == file]
    lonss = tem.iloc[:, 2]
    latss = tem.iloc[:, 3]

    # 筛选 一样的经纬度
    b = Counter(lonss)
    lons = list(b)
    c = Counter(latss)
    lats = list(c)

    # saas = tem.iloc[:,5]
    # sazs = tem.iloc[:,6]
    i = 0

    skt = np.zeros([len(lons)], dtype=float)
    sp = np.zeros([len(lons)], dtype=float)
    o3 = np.zeros([len(lons), 37], dtype=float)
    h2o = np.zeros([len(lons), 37], dtype=float)
    pt = np.zeros([len(lons), 37], dtype=float)
    u10 = np.zeros([len(lons)], dtype=float)
    v10 = np.zeros([len(lons)], dtype=float)
    t2m = np.zeros([len(lons)], dtype=float)
    saas = np.zeros([len(lons)], dtype=float)
    sazs = np.zeros([len(lons)], dtype=float)

    # 读取该日期数据
    profiles = profile_path + '/era5_' + year + month + day + '.nc'
    srffiles = srf_path + '/era5_sf' + year + month + day + '.nc'
    tpfiles = t2mq2m_path + '/era5_' + year + month + day + '.nc'
    # 一个日期里所有点的值存储一个nc，一个时刻
    for lon in lons:
        start1 = time.time()
        aa = tem.loc[(tem['fy_lon'] == lon)]
        lat = aa.iloc[0, 3]
        # 找出该经纬度表格数据
        dd = tem.loc[(tem['fy_lon'] == lon) & (tem['fy_lat'] == lat)]
        saa = dd.iloc[:, 5]
        saz = dd.iloc[:, 6]
        saas[i] = saa.values[0]
        sazs[i] = saz.values[0]
        # profile data analysis
        o3[i, :], h2o[i, :], pt[i, :], level = proreadnc(profiles, lon, lat, int(hour))

        # srf data analysis
        skt[i], sp[i] = srfreadnc(srffiles, lon, lat, int(hour))

        # uv  data analysis
        u10[i], v10[i], t2m[i] = tpreadnc(tpfiles, lon, lat, int(hour))
        i = i + 1
        end1 = time.time()
        hs1 = end1 -start1
        print(str(len(lons))+'*****' + str(file) + '******the*' + str(i) + '**round******'+'time is *'+str(hs1))
    end = time.time()
    hs = end-start
    
    name = outpath + '/era5_rt' + year + month + day + hour + '.nc'
    savencprofile(o3, h2o, pt, level, lons, lats, sp, skt, name, u10, v10, t2m, sazs, saas)
    print(str(file)+'used time is*'+str(hs))

fypoints = pd.read_csv(r'/home/data/FY_ERA5/fy_position_azimuth.csv')
profile_path = r'/home/data/FY_ERA5/profile'
srf_path = r'/home/data/FY_ERA5/srf'
t2mq2m_path = r'/home/data/FY_ERA5/t2mq2m'
outpath = r'/home/data/FY_ERA5/fync'
date = fypoints.iloc[:,4]
#(Counter(np.array(date)))
#print(fypoints.shape)
# 查看excel里面一共有哪些日期
a = Counter(date)
dates = list(a)
# a.values()
aa= dates.sort()

numbers = 0
# 遍历日期，取出每个日期对应的所有点
datess = np.array(dates)[65:81]
pool = Pool(8)
pool.map(mul,datess)
pool.close()
pool.join()
# print('poinst numbers is*'+str(len(date))+'*****Processing point numbers are*'+str(numbers))




