"""
land surface emissivity nan filling
author:ida
data:2021-08-18
"""

import os
import netCDF4 as nc
import numpy as np
from bisect import bisect_left
import pandas as pd

def monthd(path,name):
    for root, dirs, files in os.walk(path):
        for file in files:
            if name in file:
                month = nc.Dataset(path + '/'+file)
                m20 = month.variables['e20'][:]
                m22 = month.variables['e22'][:]
                m23 = month.variables['e23'][:]
                m29 = month.variables['e29'][:]
                m31 = month.variables['e31'][:]
                m32 = month.variables['e32'][:]
                mlp = month.variables['land_per'][:]
                return m20,m22,m23,m29,m31,m32

def savenc(e20,e22,e23,e29,e31,e32,land_per,xsize,ysize,outputfile):
    new = nc.Dataset(outputfile, 'w', format='NETCDF4')
    # 创建维度
    new.createDimension('lon', xsize)
    new.createDimension('lat', ysize)
    # 创建变量
    new.createVariable("lon", 'f', ("lon"))
    new.createVariable("lat", 'f', ("lat"))

    new.createVariable("e20", 'f', ("lat", "lon"))
    new.createVariable("e22", 'f', ("lat", "lon"))
    new.createVariable("e23", 'f', ("lat", "lon"))
    new.createVariable("e29", 'f', ("lat", "lon"))
    new.createVariable("e31", 'f', ("lat", "lon"))
    new.createVariable("e32", 'f', ("lat", "lon"))
    new.createVariable("land_per", 'f', ("lat", "lon"))

    new.variables['e20'][:] = e20
    new.variables['e22'][:] = e22
    new.variables['e23'][:] = e23
    new.variables['e29'][:] = e29
    new.variables['e31'][:] = e31
    new.variables['e32'][:] = e32
    new.variables['land_per'][:] = land_per

    new.close()

def filling(path,outpath,xsize,ysize):
    list6 = [1, 32, 61, 92,122,153,183,214,245,275,306,336]
    list7 = [1, 32, 60, 91,121,152,182,213,244,274,305,335]
    for root, dirs, files in os.walk(path):
        for file in files:
            if (file.startswith('MOD11C2')):
                print('开始处理:'+str(file))
                outputfile = os.path.join(outpath, file.split('.')[0]+file.split('.')[1]+'.nc')
                datae = nc.Dataset(path + '/'+file)
                data20 = datae.variables['e20'][:]
                data22 = datae.variables['e22'][:]
                data23 = datae.variables['e23'][:]
                data29 = datae.variables['e29'][:]
                data31 = datae.variables['e31'][:]
                data32 = datae.variables['e32'][:]
                datalp = datae.variables['land_per'][:]
                # 海表设为0.99
                locals = np.where(datalp < 50)
                data20[locals] = 0.99
                data22[locals] = 0.99
                data23[locals] = 0.99
                data29[locals] = 0.99
                data31[locals] = 0.99
                data32[locals] = 0.99
                # 判断缺测位置
                nanlocal20 = np.where(data20 == 0.49)
                nanlocal22 = np.where(data20 == 0.49)
                nanlocal23 = np.where(data20 == 0.49)
                nanlocal29 = np.where(data20 == 0.49)
                nanlocal31 = np.where(data20 == 0.49)
                nanlocal32 = np.where(data20 == 0.49)
                # 判断是否缺测
                if len(nanlocal20[0])>0:
                    # 用月平均匹配
                    day = int(file.split(".")[1][-3:])
                    year = int(file.split(".")[1][1:5])
                    if year==2016:
                        local = bisect_left(list6, day)
                        if local!=len(list6):
                            if day == list6[local]:
                                monthname = 'MOD11C3.A' + str(year) + str(int(list6[local])).zfill(3) + '.'
                            else:
                                monthname = 'MOD11C3.A'+str(year)+str(int(list6[local-1])).zfill(3)+'.'
                            monthdata20,monthdata22,monthdata23,monthdata29,monthdata31,monthdata32=monthd(path,monthname)
                            data20[nanlocal20]=monthdata20[nanlocal20]
                            data22[nanlocal22] = monthdata22[nanlocal22]
                            data23[nanlocal23] = monthdata23[nanlocal23]
                            data29[nanlocal29] = monthdata29[nanlocal29]
                            data31[nanlocal31] = monthdata31[nanlocal31]
                            data32[nanlocal32] = monthdata32[nanlocal32]
                        else:
                            monthname = 'MOD11C3.A' + str(year) + str(int(list6[local - 1])).zfill(3) + '.'
                            monthdata20, monthdata22, monthdata23, monthdata29, monthdata31, monthdata32 = monthd(path,monthname)
                            data20[nanlocal20] = monthdata20[nanlocal20]
                            data22[nanlocal22] = monthdata22[nanlocal22]
                            data23[nanlocal23] = monthdata23[nanlocal23]
                            data29[nanlocal29] = monthdata29[nanlocal29]
                            data31[nanlocal31] = monthdata31[nanlocal31]
                            data32[nanlocal32] = monthdata32[nanlocal32]
                    if year==2017:
                        local = bisect_left(list7, day)
                        if local!=len(list7):
                            if day == list7[local]:
                                monthname = 'MOD11C3.A' + str(year) + str(list7[local]).zfill(3) + '.'
                            else:
                                monthname = 'MOD11C3.A'+str(year)+str(list7[local-1]).zfill(3)+'.'
                            monthdata20, monthdata22, monthdata23, monthdata29, monthdata31, monthdata32 = monthd(path,monthname)
                            data20[nanlocal20] = monthdata20[nanlocal20]
                            data22[nanlocal22] = monthdata22[nanlocal22]
                            data23[nanlocal23] = monthdata23[nanlocal23]
                            data29[nanlocal29] = monthdata29[nanlocal29]
                            data31[nanlocal31] = monthdata31[nanlocal31]
                            data32[nanlocal32] = monthdata32[nanlocal32]
                        else:
                            monthname = 'MOD11C3.A' + str(year) + str(list7[local - 1]).zfill(3) + '.'
                            monthdata20, monthdata22, monthdata23, monthdata29, monthdata31, monthdata32 = monthd(path,monthname)
                            data20[nanlocal20] = monthdata20[nanlocal20]
                            data22[nanlocal22] = monthdata22[nanlocal22]
                            data23[nanlocal23] = monthdata23[nanlocal23]
                            data29[nanlocal29] = monthdata29[nanlocal29]
                            data31[nanlocal31] = monthdata31[nanlocal31]
                            data32[nanlocal32] = monthdata32[nanlocal32]
                # 月平均填充后判断缺测插值
                nanlocal220 = np.where(data20 == 0.49)
                nanlocal222 = np.where(data20 == 0.49)
                nanlocal223 = np.where(data20 == 0.49)
                nanlocal229 = np.where(data20 == 0.49)
                nanlocal231 = np.where(data20 == 0.49)
                nanlocal232 = np.where(data20 == 0.49)

                data20[nanlocal220] = np.nan
                data22[nanlocal222] = np.nan
                data23[nanlocal223] = np.nan
                data29[nanlocal229] = np.nan
                data31[nanlocal231] = np.nan
                data32[nanlocal232] = np.nan

                # 插值
                d20 = pd.DataFrame(data20)
                e20 = np.array(d20.interpolate())
                d22 = pd.DataFrame(data22)
                e22 = np.array(d22.interpolate())
                d23 = pd.DataFrame(data23)
                e23 = np.array(d23.interpolate())
                d29 = pd.DataFrame(data29)
                e29 = np.array(d29.interpolate())
                d31 = pd.DataFrame(data31)
                e31 = np.array(d31.interpolate())
                d32 = pd.DataFrame(data32)
                e32 = np.array(d32.interpolate())

                savenc(e20,e22,e23,e29,e31,e32,datalp,xsize,ysize,outputfile)
                
                print(str(file)+'*处理完成')


path = r'D:\work\data\ertm\modis\mod11'
outpath = r'D:\work\data\ertm\modis\mod11'
xsize = 241
ysize =241
filling(path,outpath,xsize,ysize)
