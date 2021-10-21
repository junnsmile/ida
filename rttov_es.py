"""

rttov emissivity extraction
author:ida
date:2021-08-20
"""

import pandas as pd
import os
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from math import sqrt
import xlsxwriter

def savencbt(lon,lat,bt,name):
    """
    函数说明：保存晴空亮温  。nc文件
    """
    new = nc.Dataset(name,'w',format='NETCDF4')
    # 创建维度
    new.createDimension('lon', len(lon))
    new.createDimension('lat', len(lat))
    #创建变量
    new.createVariable("lon", 'f', ("lon"))
    new.createVariable("lat", 'f', ("lat"))
    new.createVariable("bt8", 'f', ("lat", "lon"))
    new.createVariable("bt9", 'f', ("lat", "lon"))
    new.createVariable("bt10", 'f', ("lat", "lon"))
    new.createVariable("bt11", 'f', ("lat", "lon"))
    new.createVariable("bt12", 'f', ("lat", "lon"))
    new.createVariable("bt13", 'f', ("lat", "lon"))
    new.createVariable("bt14", 'f', ("lat", "lon"))
    new.createVariable("bt15", 'f', ("lat", "lon"))
    new.createVariable("bt16", 'f', ("lat", "lon"))

    new.variables['lat'][:] = lat
    new.variables['lon'][:] = lon
    new.variables['bt8'][:] = bt[0,:,:]
    new.variables['bt9'][:] = bt[1, :, :]
    new.variables['bt10'][:] = bt[2, :, :]
    new.variables['bt11'][:] = bt[3, :, :]
    new.variables['bt12'][:] = bt[4, :, :]
    new.variables['bt13'][:] = bt[5, :, :]
    new.variables['bt14'][:] = bt[6, :, :]
    new.variables['bt15'][:] = bt[7, :, :]
    new.variables['bt16'][:] = bt[8, :, :]
    new.close()

def savences(lon,lat,es,name):
    """
    函数说明：保存地表比辐射率 。nc文件
    """
    new = nc.Dataset(name,'w',format='NETCDF4')
    # 创建维度
    new.createDimension('lon', len(lon))
    new.createDimension('lat', len(lat))
    #创建变量
    new.createVariable("lon", 'f', ("lon"))
    new.createVariable("lat", 'f', ("lat"))
    new.createVariable("es8", 'f', ("lat", "lon"))
    new.createVariable("es9", 'f', ("lat", "lon"))
    new.createVariable("es10", 'f', ("lat", "lon"))
    new.createVariable("es11", 'f', ("lat", "lon"))
    new.createVariable("es12", 'f', ("lat", "lon"))
    new.createVariable("es13", 'f', ("lat", "lon"))
    new.createVariable("es14", 'f', ("lat", "lon"))
    new.createVariable("es15", 'f', ("lat", "lon"))
    new.createVariable("es16", 'f', ("lat", "lon"))

    new.variables['lat'][:] = lat
    new.variables['lon'][:] = lon
    new.variables['es8'][:] = es[0,:,:]
    new.variables['es9'][:] = es[1, :, :]
    new.variables['es10'][:] = es[2, :, :]
    new.variables['es11'][:] = es[3, :, :]
    new.variables['es12'][:] = es[4, :, :]
    new.variables['es13'][:] = es[5, :, :]
    new.variables['es14'][:] = es[6, :, :]
    new.variables['es15'][:] = es[7, :, :]
    new.variables['es16'][:] = es[8, :, :]
    new.close()

def tov(datas):
    """
    读取rttov .DAT只读晴空亮温的
    """
    sentimentlist = []
    for line in datas:
        s = line.strip().split('\t')
        sentimentlist.append(s)
    # datas.close()

    df_train=pd.DataFrame(sentimentlist,columns=['data'])
    btz = np.zeros([9,int((len(df_train)+1)/10)-5],dtype=np.float64)
    for i in range(int((len(df_train)+1)/10)-5):
        print(i)
        # longitude = np.array(df_train.iloc[10*i+4])[0][16:23]
        # latitude = np.array(df_train.iloc[10*i+3])[0][17:23]
        bt = np.array(df_train.iloc[(10*i+6+54):(10*i+8+54)])
        a0 =np.array(bt[0][0].split('  '))
        a1 = np.array(bt[1][0].split('  '))
        # print(a0,a1)
        btz[0,i]=a0[12]
        btz[1,i]= a0[13]
        btz[2,i]= a0[14]
        btz[3, i] = a1[0]
        btz[4, i]= a1[1]
        btz[5, i]= a1[2]
        btz[6, i] = a1[3]
        btz[7, i] = a1[4]
        btz[8, i] = a1[5]
    btall = btz.reshape(9,241,241)
    return btall

def rttovtxt(datas):
    """
    读取RTTOV .TXT 只读亮温
    """
    btz = np.zeros([11, 58081], dtype=np.float64)

    with open(datas, "r") as f:  # 打开文件
        data = f.readlines()  # 读取文件
        i = 0
        for line in data:
            numbers = np.float64(line.split())  # 将数据分隔
            btz[i, :]=numbers.flatten()
            i=i+1
    tem = btz[2:11,:]
    btall = tem.reshape(9, 241, 241)
    return btall

def rttovdat(datas):
    """
    读取RTTOV .DAT 读取晴空亮温和地表发射率
    """
    sentimentlist = []
    for line in datas:
        s = line.strip().split('\t')
        sentimentlist.append(s)
    # datas.close()

    df_train=pd.DataFrame(sentimentlist,columns=['data'])
    btz = np.zeros([9,int((len(df_train)+1)/21)-2],dtype=np.float64)
    esz = np.zeros([9, int((len(df_train) + 1) / 21) - 2], dtype=np.float64)
    for i in range(int((len(df_train)+1)/21)-2):
        print(i)
        # longitude = np.array(df_train.iloc[10*i+4])[0][16:23]
        # latitude = np.array(df_train.iloc[10*i+3])[0][17:23]
        bt = np.array(df_train.iloc[(21*i+6+54):(21*i+8+54)])
        es = np.array(df_train.iloc[(21 * i + 14 + 54):(21 * i + 16 + 54)])
        a0 =np.array(bt[0][0].split('  '))
        a1 = np.array(bt[1][0].split('  '))
        e0 =np.array(es[0][0].split('  '))
        e1 = np.array(es[1][0].split('  '))
        # print(a0,a1)
        btz[0,i]=a0[12]
        btz[1,i]= a0[13]
        btz[2,i]= a0[14]
        btz[3, i] = a1[0]
        btz[4, i]= a1[1]
        btz[5, i]= a1[2]
        btz[6, i] = a1[3]
        btz[7, i] = a1[4]
        btz[8, i] = a1[5]
        # emissivity
        esz[0,i]=e0[7]
        esz[1,i]= e0[8]
        esz[2,i]= e0[9]
        esz[3, i] = e1[0]
        esz[4, i]= e1[1]
        esz[5, i]= e1[2]
        esz[6, i] = e1[3]
        esz[7, i] = e1[4]
        esz[8, i] = e1[5]
    btall = btz.reshape(9,241,241)
    esall = esz.reshape(9, 241, 241)
    return btall,esall

pathdata = r'D:\work\data\ertm\bt'
path = r'D:\work\data\ertm'
path3 = r'D:\work\data\ertm'
i = 0
## read rttov.dat
rtdatas = open(r'D:\work\data\ertm\rttov\2016030507_atlas.dat',encoding='utf-8')
bt,es = rttovdat(rtdatas)
#####################
## read rttov.txt
# rtdata = r'D:\work\data\ertm\rttov\2016082809(1).txt'
# bt = rttovtxt(rtdata)
#####################
file = '\\NC_H08_20160305_0700_L2CLPbet_FLDK.02401_02401.nc'
file2 = '\\2016030507_bt_j.nc'
file3 = '\\NC_H08_20160305_0700_R21_FLDK.02401_02401.nc'
# moddata = 'D:\work\data\ertm\modis\MOD11C2.A2016241.006.2016257120130.nc'

# 读取MODIS LANDPERCENT
# modisdata = nc.Dataset(moddata)
# landa = modisdata.variables['land_per'][:].data
#########################

data =nc.Dataset(path+file)
data2 =nc.Dataset(pathdata+file2)
data3 = nc.Dataset(path3+file3)
print(data.variables.keys())
print(data3.variables.keys())

lon = data2.variables['lon'][:].data
lat = data2.variables['lat'][:].data
llon,llat = np.meshgrid(lon, lat)

cltype = data.variables['CLTYPE'][:].data
qa = data.variables['QA'][:].data

###################
# 保存rttov读取结果
name = 'D:\\work\\data\\ertm\\rttov\\'+os.path.split(file2)[1].split('_bt')[0]+'_rtes.nc'
# h海表设为0.99
# locals = np.where(landa < 50)
# es[0,:,:][locals] = 0.99
# es[1,:,:][locals] = 0.99
# es[2,:,:][locals] = 0.99
# es[3,:,:][locals] = 0.99
# es[4,:,:][locals] = 0.99
# es[5,:,:][locals] = 0.99
# es[6,:,:][locals] = 0.99
# es[7,:,:][locals] = 0.99
# es[8,:,:][locals] = 0.99
savences(lon,lat,es,name)
# savencbt(lon,lat,bt,name)