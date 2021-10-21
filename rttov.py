"""

rttov error analysis
author:ida
date:2021-08-06
"""

import pandas as pd
import os
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from math import sqrt
import xlsxwriter

def savences(lon,lat,es,name):
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
    az = np.zeros([9,int((len(df_train)+1)/10)-5],dtype=np.float64)
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


def rmse(error):
    # local = np.where(error!=np.nan)

    squarederror = (error*error)
    rmse = sqrt(np.nansum(squarederror) / (len(error[~np.isnan(error)])))
    mean = (np.nansum(error))/ len(error[~np.isnan(error)])
    max = np.nanmax(error)
    min = np.nanmin(error)
    return rmse,mean,max,min

def xsl(data):
    file = r'D:\work\code\ertm\result\030507rt.xlsx'
    workbook = xlsxwriter.Workbook(file)  # 建立文件
    worksheet = workbook.add_worksheet()
    # 建立sheet， 可以work.add_worksheet('employee')来指定sheet名，但中文名会报UnicodeDecodeErro的错误
    for i in range(4):
        for j in range(9):
            worksheet.write(i, j, data[i,j])  # 向A1写入
    workbook.close()


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
file2 = '\\2016030507_ebt.nc'
file3 = '\\NC_H08_20160305_0700_R21_FLDK.02401_02401.nc'
# moddata = 'D:\work\data\ertm\modis\MOD11C2.A2016241.006.2016242200752.nc'

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
name = 'D:\\work\\data\\ertm\\rttov\\'+os.path.split(file2)[1].split('_bt')[0]+'_es.nc'
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
####################

qai = qa.astype(int).flatten()
tem = []
for q in qai:
    if q !=1:
        # tem.append(int(str(bin(q)[4])+str(bin(q)[3])+str(bin(q)[2])))
        tem.append(str(bin(q)[-5]) + str(bin(q)[-4]))
    else:
        tem.append(-999)
qa2 = np.array(tem).reshape(qa.shape)
# print(file2+'****percentage*is*****'+str(np.shape(np.where(cltype==0))[1]/(np.shape(cltype)[0]*np.shape(cltype)[1])))
BT8 = bt[0,:,:]
BT9 = bt[1,:,:]
BT10 = bt[2,:,:]
BT11 = bt[3,:,:]
BT12 = bt[4,:,:]
BT13 = bt[5,:,:]
BT14 = bt[6,:,:]
BT15 = bt[7,:,:]
BT16 = bt[8,:,:]

local = np.where(cltype==0)

bt8= data3['tbb_08'][:].data[399:640,639:880]
c8= BT8 -bt8
c8[np.where((BT8<-100)|(BT8>500))] = np.nan
# c8[np.where(cltype[399:640,639:880]!=0)]=np.nan
c8[np.where(qa2[399:640,639:880]!='00')]=np.nan
c8rmse,c8mean,c8max,c8min = rmse(c8)
print('****c8:'+str(format(c8rmse, '.3f'))+'*'+str(format(c8mean, '.3f'))+'*'+str(format(c8max, '.3f')) +'*'+str(format(c8min, '.3f')))

############
bt9 = data3['tbb_09'][:].data[399:640,639:880]
c9= BT9 -bt9
c9[np.where((BT9<-100)|(BT9>500))] = np.nan
# c9[np.where(cltype[399:640,639:880]!=0)]=np.nan
c9[np.where(qa2[399:640,639:880]!='00')]=np.nan
c9rmse,c9mean,c9max,c9min = rmse(c9)
print('****c9:'+str(format(c9rmse, '.3f'))+'*'+str(format(c9mean, '.3f'))+'*'+str(format(c9max, '.3f')) +'*'+str(format(c9min, '.3f')))


bt10 = data3['tbb_10'][:].data[399:640,639:880]
c10= BT10 -bt10
c10[np.where((BT10<-100)|(BT10>500))] = np.nan
# c10[np.where(cltype[399:640,639:880]!=0)]=np.nan
c10[np.where(qa2[399:640,639:880]!='00')]=np.nan
c10rmse,c10mean,c10max,c10min = rmse(c10)
print('****c10:'+str(format(c10rmse, '.3f'))+'*'+str(format(c10mean, '.3f'))+'*'+str(format(c10max, '.3f')) +'*'+str(format(c10min, '.3f')))


bt11 = data3['tbb_11'][:].data[399:640,639:880]
c11= BT11 -bt11
c11[np.where((BT11<-100)|(BT11>500))] = np.nan
# c11[np.where(cltype[399:640,639:880]!=0)]=np.nan
c11[np.where(qa2[399:640,639:880]!='00')]=np.nan
c11rmse,c11mean,c11max,c11min = rmse(c11)
print('****c11:'+str(format(c11rmse, '.3f'))+'*'+str(format(c11mean, '.3f'))+'*'+str(format(c11max, '.3f')) +'*'+str(format(c11min, '.3f')))

bt12 = data3['tbb_12'][:].data[399:640,639:880]
c12= BT12 -bt12
c12[np.where((BT12<-100)|(BT12>500))] = np.nan
# c12[np.where(cltype[399:640,639:880]!=0)]=np.nan
c12[np.where(qa2[399:640,639:880]!='00')]=np.nan
c12rmse,c12mean,c12max,c12min = rmse(c12)
print('****c12:'+str(format(c12rmse, '.3f'))+'*'+str(format(c12mean, '.3f'))+'*'+str(format(c12max, '.3f')) +'*'+str(format(c12min, '.3f')))


bt13 = data3['tbb_13'][:].data[399:640,639:880]
c13= BT13 -bt13
c13[np.where((BT13<-100)|(BT13>500))] = np.nan
# c13[np.where(cltype[399:640,639:880]!=0)]=np.nan
c13[np.where(qa2[399:640,639:880]!='00')]=np.nan
c13rmse,c13mean,c13max,c13min = rmse(c13)
print('****c13:'+str(format(c13rmse, '.3f'))+'*'+str(format(c13mean, '.3f'))+'*'+str(format(c13max, '.3f')) +'*'+str(format(c13min, '.3f')))


bt14 = data3['tbb_14'][:].data[399:640,639:880]
c14= BT14 -bt14
c14[np.where((BT14<-100)|(BT14>500))] = np.nan
# c14[np.where(cltype[399:640,639:880]!=0)]=np.nan
c14[np.where(qa2[399:640,639:880]!='00')]=np.nan
c14rmse,c14mean,c14max,c14min = rmse(c14)
print('****c14:'+str(format(c14rmse, '.3f'))+'*'+str(format(c14mean, '.3f'))+'*'+str(format(c14max, '.3f')) +'*'+str(format(c14min, '.3f')))


bt15 = data3['tbb_15'][:].data[399:640,639:880]
c15= BT15 -bt15
c15[np.where((BT15<-100)|(BT15>500))] = np.nan
# c15[np.where(cltype[399:640,639:880]!=0)]=np.nan
c15[np.where(qa2[399:640,639:880]!='00')]=np.nan
c15rmse,c15mean,c15max,c15min = rmse(c15)
print('****c15:'+str(format(c15rmse, '.3f'))+'*'+str(format(c15mean, '.3f'))+'*'+str(format(c15max, '.3f')) +'*'+str(format(c15min, '.3f')))


bt16 = data3['tbb_16'][:].data[399:640,639:880]
c16= BT16 -bt16
c16[np.where((BT16<-100)|(BT16>500))] = np.nan
c16[np.where(qa2[399:640,639:880]!='00')]=np.nan
c16rmse,c16mean,c16max,c16min = rmse(c16)
print('****c16:'+str(format(c16rmse, '.3f'))+'*'+str(format(c16mean, '.3f'))+'*'+str(format(c16max, '.3f')) +'*'+str(format(c16min, '.3f')))


if (np.isnan(c8rmse)==True) | (np.isnan(c9rmse)==True)| (np.isnan(c10rmse)==True)| (np.isnan(c11rmse)==True)| \
        (np.isnan(c12rmse)==True)| (np.isnan(c13rmse)==True)| (np.isnan(c14rmse)==True)\
        | (np.isnan(c15rmse)==True)| (np.isnan(c16rmse)==True):
    i = i-1

data = np.zeros([4,9],dtype=np.float64)


data[0,0]=c8rmse
data[1,0]=c8max
data[2,0] = c8min
data[3,0]=c8mean

data[0,1]=c9rmse
data[1,1]=c9max
data[2,1]=c9min
data[3,1]=c9mean


data[0,2]=c10rmse
data[1,2]=c10max
data[2,2]=c10min
data[3,2]=c10mean


data[0,3]=c11rmse
data[1,3]=c11max
data[2,3]=c11min
data[3,3]=c11mean


data[0,4]=c12rmse
data[1,4]=c12max
data[2,4]=c12min
data[3,4]=c12mean


data[0,5]=c13rmse
data[1,5]=c13max
data[2,5]=c13min
data[3,5]=c13mean

data[0,6]=c14rmse
data[1,6]=c14max
data[2,6]=c14min
data[3,6]=c14mean

data[0,7]=c15rmse
data[1,7]=c15max
data[2,7]=c15min
data[3,7]=c15mean

data[0,8]=c16rmse
data[1,8]=c16max
data[2,8]=c16min
data[3,8]= c16mean

xsl(data)

