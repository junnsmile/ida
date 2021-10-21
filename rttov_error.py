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

def xsl(data):
    file = r'D:\work\code\ertm\result\1.xlsx'
    workbook = xlsxwriter.Workbook(file)  # 建立文件
    worksheet = workbook.add_worksheet()
    # 建立sheet， 可以work.add_worksheet('employee')来指定sheet名，但中文名会报UnicodeDecodeErro的错误
    for i in range(5):
        for j in range(9):
            worksheet.write(i, j, data[i,j])  # 向A1写入
    workbook.close()


def rmse(error):
    # local = np.where(error!=np.nan)

    squarederror = (error * error)
    rmse = sqrt(np.nansum(squarederror) / (len(error[~np.isnan(error)])))
    mae = (np.nansum(abs(error))) / len(error[~np.isnan(error)])
    mean = (np.nansum(error)) / len(error[~np.isnan(error)])
    max = np.nanmax(error)
    min = np.nanmin(error)
    return rmse, mean, mae, max, min

def errorr(rmsez,maez,meanz,maxz,minz,i):
    ccrmse = np.nansum(np.array(rmsez)) / (len(rmsez)-len(np.where(np.isnan(rmsez))[0]))
    ccmae = np.nansum(np.array(maez)) / (len(maez) - len(np.where(np.isnan(maez))[0]))
    ccmax = np.nansum(np.array(maxz)) / (len(maxz)-len(np.where(np.isnan(maxz))[0]))
    ccmin = np.nansum(np.array(minz)) / (len(minz)-len(np.where(np.isnan(minz))[0]))
    ccmean = np.nansum(np.array(meanz)) / (len(meanz)-len(np.where(np.isnan(meanz))[0]))
    data[0, i] = ccrmse
    data[1, i] = ccmae
    data[3, i] = ccmax
    data[4, i] = ccmin
    data[2, i] = ccmean

rtdata = r'D:\work\data\ertm\rttov'
path = r'D:\work\data\ertm\j82'
path3 = r'D:\work\data\ertm\j8'
c8rmsez =[]
c8maxz = []
c8minz = []
c8meanz = []
c8maez = []
c9rmsez =[]
c9maxz = []
c9minz = []
c9maez = []
c9meanz = []
c10rmsez =[]
c10maxz = []
c10minz = []
c10meanz = []
c10maez = []
c11rmsez =[]
c11maxz = []
c11minz = []
c11meanz = []
c11maez = []
c12rmsez =[]
c12maxz = []
c12minz = []
c12meanz = []
c12maez = []
c13rmsez =[]
c13maxz = []
c13minz = []
c13meanz = []
c13maez = []
c14rmsez =[]
c14maxz = []
c14minz = []
c14meanz = []
c14maez = []
c15rmsez =[]
c15maxz = []
c15minz = []
c15meanz = []
c15maez = []
c16rmsez =[]
c16maxz = []
c16minz = []
c16meanz = []
c16maez = []
for f in os.listdir(rtdata):
    print(f)
    if f.endswith('.dat'):
        name = os.path.splitext(f)[0][7:17]
        year = name[0:4]
        month = name[4:6]
        day = name[6:8]
        time = name[8:10]
    else:
        continue
    file = '\\NC_H08_'+year+month+day+'_'+time+'00_L2CLPbet_FLDK.02401_02401.nc'
    file3 = '\\NC_H08_'+year+month+day+'_'+time+'00_R21_FLDK.02401_02401.nc'
    ## read rttov.dat
    rtpath = rtdata+'\\'+f
    rtdatas = open(rtpath,encoding='utf-8')
    bt,es = rttovdat(rtdatas)
    #####################
    ## read rttov.txt
    # rtdata = r'D:\work\data\ertm\rttov\2016082809(1).txt'
    # bt = rttovtxt(rtdata)
    #####################
    btfile = r'D:\work\data\ertm\bt\2016030507_bt_j.nc'
    #########################
    data = nc.Dataset(path + '\\' +year+month+'\\' + day + '\\' + time + file)
    data2 = nc.Dataset(btfile)
    data3 = nc.Dataset(path3 +'\\' +year+month+ '\\' + day + file3)
    print(data.variables.keys())
    print(data3.variables.keys())

    lon = data2.variables['lon'][:].data
    lat = data2.variables['lat'][:].data
    llon,llat = np.meshgrid(lon, lat)

    cltype = data.variables['CLTYPE'][:].data
    qa = data.variables['QA'][:].data

    if np.nanmax(qa) == np.nanmin(qa):
        i = i-1
        continue
    qai = qa.astype(int).flatten()
    tem = []
    for q in qai:
        a =bin(q)
        #print(a)
        if (q != 1)&(len(a)>3):
            # tem.append(int(str(bin(q)[4])+str(bin(q)[3])+str(bin(q)[2])))
            tem.append(str(bin(q)[-5]) + str(bin(q)[-4]))
        else:
            tem.append(str(-999))
    qa2 = np.array(tem).reshape(qa.shape)
    ###################
    # 保存rttov读取结果
    # name = 'D:\\work\\data\\ertm\\rttov\\'+os.path.split(btfile)[1].split('_bt')[0]+'_rtbt.nc'
    #
    # savencbt(lon,lat,bt,name)
    BT8 = bt[0,:,:]
    BT9 = bt[1,:,:]
    BT10 = bt[2,:,:]
    BT11 = bt[3,:,:]
    BT12 = bt[4,:,:]
    BT13 = bt[5,:,:]
    BT14 = bt[6,:,:]
    BT15 = bt[7,:,:]
    BT16 = bt[8,:,:]

    local = np.where(cltype == 0)

    bt8 = data3['tbb_08'][:].data[399:640, 639:880]
    c8 = BT8 - bt8
    c8[np.where((BT8 < -100) | (BT8 > 500))] = np.nan
    c8[np.where(qa2[399:640, 639:880] != '00')] = np.nan
    c8rmse, c8mean, c8mae, c8max, c8min = rmse(c8)
    print('****c8:' + str(format(c8rmse, '.3f')) + '*' + str(format(c8mean, '.3f')) + '*' + str(
        format(c8max, '.3f')) + '*' + str(format(c8min, '.3f')))
    c8meanz.append(c8mean)
    c8maez.append(c8mae)
    c8maxz.append(c8max)
    c8minz.append(c8min)
    c8rmsez.append(c8rmse)
    ############
    bt9 = data3['tbb_09'][:].data[399:640, 639:880]
    c9 = BT9 - bt9
    c9[np.where((BT9 < -100) | (BT9 > 500))] = np.nan
    c9[np.where(qa2[399:640, 639:880] != '00')] = np.nan
    c9rmse, c9mean, c9mae, c9max, c9min = rmse(c9)
    print('****c9:' + str(format(c9rmse, '.3f')) + '*' + str(format(c9mean, '.3f')) + '*' + str(
        format(c9max, '.3f')) + '*' + str(format(c9min, '.3f')))
    c9meanz.append(c9mean)
    c9maez.append(c9mae)
    c9maxz.append(c9max)
    c9minz.append(c9min)
    c9rmsez.append(c9rmse)

    bt10 = data3['tbb_10'][:].data[399:640, 639:880]
    c10 = BT10 - bt10
    c10[np.where((BT10 < -100) | (BT10 > 500))] = np.nan
    c10[np.where(qa2[399:640, 639:880] != '00')] = np.nan
    c10rmse, c10mean, c10mae, c10max, c10min = rmse(c10)
    print('****c10:' + str(format(c10rmse, '.3f')) + '*' + str(format(c10mean, '.3f')) + '*' + str(
        format(c10max, '.3f')) + '*' + str(format(c10min, '.3f')))
    c10meanz.append(c10mean)
    c10maez.append(c10mae)
    c10maxz.append(c10max)
    c10minz.append(c10min)
    c10rmsez.append(c10rmse)

    bt11 = data3['tbb_11'][:].data[399:640, 639:880]
    c11 = BT11 - bt11
    c11[np.where((BT11 < -100) | (BT11 > 500))] = np.nan
    c11[np.where(qa2[399:640, 639:880] != '00')] = np.nan
    c11rmse, c11mean, c11mae, c11max, c11min = rmse(c11)
    print('****c11:' + str(format(c11rmse, '.3f')) + '*' + str(format(c11mean, '.3f')) + '*' + str(
        format(c11max, '.3f')) + '*' + str(format(c11min, '.3f')))
    c11meanz.append(c11mean)
    c11maez.append(c11mae)
    c11maxz.append(c11max)
    c11minz.append(c11min)
    c11rmsez.append(c11rmse)

    bt12 = data3['tbb_12'][:].data[399:640, 639:880]
    c12 = BT12 - bt12
    c12[np.where((BT12 < -100) | (BT12 > 500))] = np.nan
    c12[np.where(qa2[399:640, 639:880] != '00')] = np.nan
    c12rmse, c12mean, c12mae, c12max, c12min = rmse(c12)
    print('****c12:' + str(format(c12rmse, '.3f')) + '*' + str(format(c12mean, '.3f')) + '*' + str(
        format(c12max, '.3f')) + '*' + str(format(c12min, '.3f')))
    c12meanz.append(c12mean)
    c12maez.append(c12mae)
    c12maxz.append(c12max)
    c12minz.append(c12min)
    c12rmsez.append(c12rmse)

    bt13 = data3['tbb_13'][:].data[399:640, 639:880]
    c13 = BT13 - bt13
    c13[np.where((BT13 < -100) | (BT13 > 500))] = np.nan
    c13[np.where(qa2[399:640, 639:880] != '00')] = np.nan
    c13rmse, c13mean, c13mae, c13max, c13min = rmse(c13)
    print('****c13:' + str(format(c13rmse, '.3f')) + '*' + str(format(c13mean, '.3f')) + '*' + str(
        format(c13max, '.3f')) + '*' + str(format(c13min, '.3f')))
    c13meanz.append(c13mean)
    c13maez.append(c13mae)
    c13maxz.append(c13max)
    c13minz.append(c13min)
    c13rmsez.append(c13rmse)

    bt14 = data3['tbb_14'][:].data[399:640, 639:880]
    c14 = BT14 - bt14
    c14[np.where((BT14 < -100) | (BT14 > 500))] = np.nan
    c14[np.where(qa2[399:640, 639:880] != '00')] = np.nan
    c14rmse, c14mean, c14mae, c14max, c14min = rmse(c14)
    print('****c14:' + str(format(c14rmse, '.3f')) + '*' + str(format(c14mean, '.3f')) + '*' + str(
        format(c14max, '.3f')) + '*' + str(format(c14min, '.3f')))
    c14meanz.append(c14mean)
    c14maez.append(c14mae)
    c14maxz.append(c14max)
    c14minz.append(c14min)
    c14rmsez.append(c14rmse)

    bt15 = data3['tbb_15'][:].data[399:640, 639:880]
    c15 = BT15 - bt15
    c15[np.where((BT15 < -100) | (BT15 > 500))] = np.nan
    c15[np.where(qa2[399:640, 639:880] != '00')] = np.nan
    c15rmse, c15mean, c15mae, c15max, c15min = rmse(c15)
    print('****c15:' + str(format(c15rmse, '.3f')) + '*' + str(format(c15mean, '.3f')) + '*' + str(
        format(c15max, '.3f')) + '*' + str(format(c15min, '.3f')))
    c15meanz.append(c15mean)
    c15maez.append(c15mae)
    c15maxz.append(c15max)
    c15minz.append(c15min)
    c15rmsez.append(c15rmse)

    bt16 = data3['tbb_16'][:].data[399:640, 639:880]
    c16 = BT16 - bt16
    c16[np.where((BT16 < -100) | (BT16 > 500))] = np.nan
    c16[np.where(qa2[399:640, 639:880] != '00')] = np.nan
    c16rmse, c16mean, c16mae, c16max, c16min = rmse(c16)
    print('****c16:' + str(format(c16rmse, '.3f')) + '*' + str(format(c16mean, '.3f')) + '*' + str(
        format(c16max, '.3f')) + '*' + str(format(c16min, '.3f')))
    c16meanz.append(c16mean)
    c16maez.append(c16mae)
    c16maxz.append(c16max)
    c16minz.append(c16min)
    c16rmsez.append(c16rmse)
    # cc8mean = np.nansum(np.array(c8meanz)) / i
    # print('**********************'+str(cc8mean))
    lon = data2.variables['lon'][:].data
    lat = data2.variables['lat'][:].data

    llon, llat = np.meshgrid(lon, lat)
    # name = os.path.split(file2)[1].split('_bt')[0]

    if (np.isnan(c8rmse) == True) | (np.isnan(c9rmse) == True) | (np.isnan(c10rmse) == True) | (
            np.isnan(c11rmse) == True) | \
            (np.isnan(c12rmse) == True) | (np.isnan(c13rmse) == True) | (np.isnan(c14rmse) == True) \
            | (np.isnan(c15rmse) == True) | (np.isnan(c16rmse) == True):
        i = i - 1

data = np.zeros([5, 9], dtype=np.float64)
errorr(c8rmsez, c8maez, c8meanz, c8maxz, c8minz, 0)
errorr(c9rmsez, c9maez, c9meanz, c9maxz, c9minz, 1)
errorr(c10rmsez, c10maez, c10meanz, c10maxz, c10minz, 2)
errorr(c11rmsez, c11maez, c11meanz, c11maxz, c11minz, 3)
errorr(c12rmsez, c12maez, c12meanz, c12maxz, c12minz, 4)
errorr(c13rmsez, c13maez, c13meanz, c13maxz, c13minz, 5)
errorr(c14rmsez, c14maez, c14meanz, c14maxz, c14minz, 6)
errorr(c15rmsez, c15maez, c15meanz, c15maxz, c15minz, 7)
errorr(c16rmsez, c16maez, c16meanz, c16maxz, c16minz, 8)

print(data)
xsl(data)

print(i)