"""
clear sky BT result
one time---add mae
error analysis

author：ida
date:2021-07-15
"""

import netCDF4 as nc
import numpy as np
import os
import matplotlib.pyplot as plt
from math import sqrt
import xlsxwriter
from pandas import DataFrame

def drawHist(heights,name,date):
    #创建直方图
    #第一个参数为待绘制的定量数据，不同于定性数据，这里并没有实现进行频数统计
    #第二个参数为划分的区间个数
    plt.hist(heights,range=(0,80),histtype=u'bar', align=u'left', orientation=u'vertical',
                rwidth=0.4,bins=30)
    plt.xlabel('Difference')
    plt.ylabel('Frequency of points')
    plt.title('BT Difference of '+str(name))
    plt.savefig('D:\\work\\code\\ertm\\result\\2_'+str(name)+str(date))
    # plt.show()
    plt.close()
    print(str(name)+'**draw  success!')
def drawSashdiagram(llon,llat,data,name,date):
    scc = plt.scatter(llon,llat,c=data,marker='o',cmap='Blues')
    plt.colorbar(scc)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('BT Difference of ' + str(name))
    plt.savefig('D:\\work\\code\\ertm\\result\\1_'+str(name)+str(date))
    # plt.show()
    plt.close()
    print(str(name) + '**draw sash success!')

def rmse(error):
    # local = np.where(error!=np.nan)

    squarederror = (error*error)
    rmse = sqrt(np.nansum(squarederror) / (len(error[~np.isnan(error)])))
    mae = (np.nansum(abs(error)))/ len(error[~np.isnan(error)])
    mean = (np.nansum(error))/ len(error[~np.isnan(error)])
    max = np.nanmax(error)
    min = np.nanmin(error)
    return rmse,mean,mae,max,min

def xsl(data):
    file = r'D:\work\code\ertm\result\16082809ngw.xlsx'
    workbook = xlsxwriter.Workbook(file)  # 建立文件
    worksheet = workbook.add_worksheet()

    # 建立sheet， 可以work.add_worksheet('employee')来指定sheet名，但中文名会报UnicodeDecodeErro的错误
    for i in range(5):
        for j in range(9):
            worksheet.write(i, j, data[i,j])  # 向A1写入
    workbook.close()


def rttovtxt(datas):
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


if __name__ == '__main__':
    #rttov
    # rtdata = r'D:\work\data\ertm\rttov\2016082809(1).txt'
    # rtbt = rttovtxt(rtdata)
#######################################
    pathdata = r'D:\work\data\ertm\bt'
    path = r'D:\work\data\ertm'
    path3 = r'D:\work\data\ertm'
    i = 0

    # path0 = path+'/'+f
    # for file in os.listdir(path):
    file = '\\NC_H08_20160828_0900_L2CLPbet_FLDK.02401_02401.nc'
    file2 = '\\2016082809_ng.nc'
    file3 = '\\NC_H08_20160828_0900_R21_FLDK.02401_02401.nc'
    # file = '\\NC_H08_20160331_0300_L2CLPbet_FLDK.02401_02401.nc'
    # file2 = '\\2016033103_planck.nc'
    # file3 = '\\NC_H08_20160331_0300_R21_FLDK.02401_02401.nc'
    # file = '\\NC_H08_20160305_0700_L2CLPbet_FLDK.02401_02401.nc'
    # file2 = '\\2016030507_ng.nc'
    # file3 = '\\NC_H08_20160305_0700_R21_FLDK.02401_02401.nc'
    data =nc.Dataset(path+file)
    data2 =nc.Dataset(pathdata+file2)
    data3 = nc.Dataset(path3+file3)
    print(data.variables.keys())
    print(data2.variables.keys())
    print(data3.variables.keys())

    cltype = data.variables['CLTYPE'][:].data
    qa = data.variables['QA'][:].data

    qai = qa.astype(int).flatten()
    tem = []
    for q in qai:
        a =bin(q)
        #print(a)
        if (q != 1)&(len(a)>3):
            # tem.append(int(str(bin(q)[4])+str(bin(q)[3])+str(bin(q)[2])))
            tem.append(str(bin(q)[-5]) + str(bin(q)[-4]))
        else:
            tem.append((str(-999)))
    qa2 = np.array(tem).reshape(qa.shape)
    print(file2+'****percentage*is*****'+str(np.shape(np.where(cltype==0))[1]/(np.shape(cltype)[0]*np.shape(cltype)[1])))
    BT8 = data2['BT'][:].data[0]
    BT9 = data2['BT'][:].data[1]
    BT10 = data2['BT'][:].data[2]
    BT11 = data2['BT'][:].data[3]
    BT12 = data2['BT'][:].data[4]
    BT13 = data2['BT'][:].data[5]
    BT14 = data2['BT'][:].data[6]
    BT15 = data2['BT'][:].data[7]
    BT16 = data2['BT'][:].data[8]

    local = np.where(cltype==0)

    # bt9 = data3['tbb_09'][:].data[1599:2200,999:1600]
    # c9= BT9 -bt9
    # bt10 = data3['tbb_10'][:].data[1599:2200,999:1600]
    # c10= BT10 -bt10
    # bt11 = data3['tbb_11'][:].data[1599:2200,999:1600]
    # c11= BT11 -bt11
    # bt12 = data3['tbb_12'][:].data[1599:2200,999:1600]
    # c12= BT12 -bt12
    # bt13 = data3['tbb_13'][:].data[1599:2200,999:1600]
    # c13= BT13 -bt13
    # bt14 = data3['tbb_14'][:].data[1599:2200,999:1600]
    # c14= BT14 -bt14
    # bt15 = data3['tbb_15'][:].data[1599:2200,999:1600]
    # c15= BT15 -bt15
    # c15= BT15 -bt15
    # bt16 = data3['tbb_16'][:].data[1599:2200,999:1600]
    # c16= BT16 -bt16

    bt8= data3['tbb_08'][:].data[399:640,639:880]
    c8= BT8 -bt8
    # 去海表
    # c8[np.isnan(rtbt[0])]= np.nan
    # 去异常点NAN
    c8[np.where((BT8<-100)|(BT8>500))] = np.nan
    # c8[np.where(cltype[399:640,639:880]!=0)]=np.nan
    c8[np.where(qa2[399:640,639:880]!='00')]=np.nan
    c8rmse,c8mean,c8mae,c8max,c8min = rmse(c8)
    print('****c8:'+str(format(c8rmse, '.3f'))+'*'+str(format(c8mean, '.3f'))+'*'+str(format(c8max, '.3f')) +'*'+str(format(c8min, '.3f')))

    ############
    bt9 = data3['tbb_09'][:].data[399:640,639:880]
    c9= BT9 -bt9
    # 去海表
    # c9[np.isnan(rtbt[1])]= np.nan
    # 去异常点NAN
    c9[np.where((BT9<-100)|(BT9>500))] = np.nan
    # c9[np.where(cltype[399:640,639:880]!=0)]=np.nan
    c9[np.where(qa2[399:640,639:880]!='00')]=np.nan
    c9rmse,c9mean,c9mae,c9max,c9min = rmse(c9)
    print('****c9:'+str(format(c9rmse, '.3f'))+'*'+str(format(c9mean, '.3f'))+'*'+str(format(c9max, '.3f')) +'*'+str(format(c9min, '.3f')))


    bt10 = data3['tbb_10'][:].data[399:640,639:880]
    c10= BT10 -bt10
    # 去海表
    # c10[np.isnan(rtbt[2])]= np.nan
    # 去异常点NAN
    c10[np.where((BT10<-100)|(BT10>500))] = np.nan
    # c10[np.where(cltype[399:640,639:880]!=0)]=np.nan
    c10[np.where(qa2[399:640,639:880]!='00')]=np.nan
    c10rmse,c10mean,c10mae,c10max,c10min = rmse(c10)
    print('****c10:'+str(format(c10rmse, '.3f'))+'*'+str(format(c10mean, '.3f'))+'*'+str(format(c10max, '.3f')) +'*'+str(format(c10min, '.3f')))


    bt11 = data3['tbb_11'][:].data[399:640,639:880]
    c11= BT11 -bt11
    # 去海表
    # c11[np.isnan(rtbt[3])]= np.nan
    # 去异常点NAN
    c11[np.where((BT11<-100)|(BT11>500))] = np.nan
    # c11[np.where(cltype[399:640,639:880]!=0)]=np.nan
    c11[np.where(qa2[399:640,639:880]!='00')]=np.nan
    c11rmse,c11mean,c11mae,c11max,c11min = rmse(c11)
    print('****c11:'+str(format(c11rmse, '.3f'))+'*'+str(format(c11mean, '.3f'))+'*'+str(format(c11max, '.3f')) +'*'+str(format(c11min, '.3f')))

    bt12 = data3['tbb_12'][:].data[399:640,639:880]
    c12= BT12 -bt12
    # 去海表
    # c12[np.isnan(rtbt[4])]= np.nan
    # 去异常点NAN
    c12[np.where((BT12<-100)|(BT12>500))] = np.nan
    # c12[np.where(cltype[399:640,639:880]!=0)]=np.nan
    c12[np.where(qa2[399:640,639:880]!='00')]=np.nan
    c12rmse,c12mean,c12mae,c12max,c12min = rmse(c12)
    print('****c12:'+str(format(c12rmse, '.3f'))+'*'+str(format(c12mean, '.3f'))+'*'+str(format(c12max, '.3f')) +'*'+str(format(c12min, '.3f')))


    bt13 = data3['tbb_13'][:].data[399:640,639:880]
    c13= BT13 -bt13
    # 去海表
    # c13[np.isnan(rtbt[5])]= np.nan
    # 去异常点NAN
    c13[np.where((BT13<-100)|(BT13>500))] = np.nan
    # c13[np.where(cltype[399:640,639:880]!=0)]=np.nan
    c13[np.where(qa2[399:640,639:880]!='00')]=np.nan
    c13rmse,c13mean,c13mae,c13max,c13min = rmse(c13)
    print('****c13:'+str(format(c13rmse, '.3f'))+'*'+str(format(c13mean, '.3f'))+'*'+str(format(c13max, '.3f')) +'*'+str(format(c13min, '.3f')))


    bt14 = data3['tbb_14'][:].data[399:640,639:880]
    c14= BT14 -bt14
    # 去海表
    # c14[np.isnan(rtbt[6])]= np.nan
    # 去异常点NAN
    c14[np.where((BT14<-100)|(BT14>500))] = np.nan
    # c14[np.where(cltype[399:640,639:880]!=0)]=np.nan
    c14[np.where(qa2[399:640,639:880]!='00')]=np.nan
    c14rmse,c14mean,c14mae,c14max,c14min = rmse(c14)
    print('****c14:'+str(format(c14rmse, '.3f'))+'*'+str(format(c14mean, '.3f'))+'*'+str(format(c14max, '.3f')) +'*'+str(format(c14min, '.3f')))


    bt15 = data3['tbb_15'][:].data[399:640,639:880]
    c15= BT15 -bt15
    # 去海表
    # c15[np.isnan(rtbt[7])]= np.nan
    # 去异常点NAN
    c15[np.where((BT15<-100)|(BT15>500))] = np.nan
    # c15[np.where(cltype[399:640,639:880]!=0)]=np.nan
    c15[np.where(qa2[399:640,639:880]!='00')]=np.nan
    c15rmse,c15mean,c15mae,c15max,c15min = rmse(c15)
    print('****c15:'+str(format(c15rmse, '.3f'))+'*'+str(format(c15mean, '.3f'))+'*'+str(format(c15max, '.3f')) +'*'+str(format(c15min, '.3f')))


    bt16 = data3['tbb_16'][:].data[399:640,639:880]
    c16= BT16 -bt16
    # 去海表
    # c16[np.isnan(rtbt[8])]= np.nan
    # 去异常点NAN
    c16[np.where((BT16<-100)|(BT16>500))] = np.nan
    c16[np.where(qa2[399:640,639:880]!='00')]=np.nan
    c16rmse,c16mean,c16mae,c16max,c16min = rmse(c16)
    print('****c16:'+str(format(c16rmse, '.3f'))+'*'+str(format(c16mean, '.3f'))+'*'+str(format(c16max, '.3f')) +'*'+str(format(c16min, '.3f')))


    lon = data2.variables['lon'][:].data
    lat = data2.variables['lat'][:].data

    llon,llat = np.meshgrid(lon, lat)
    name = os.path.split(file2)[1].split('_bt')[0]

    if (np.isnan(c8rmse)==True) | (np.isnan(c9rmse)==True)| (np.isnan(c10rmse)==True)| (np.isnan(c11rmse)==True)| \
            (np.isnan(c12rmse)==True)| (np.isnan(c13rmse)==True)| (np.isnan(c14rmse)==True)\
            | (np.isnan(c15rmse)==True)| (np.isnan(c16rmse)==True):
        i = i-1

    data = np.zeros([5,9],dtype=np.float64)


    data[0,0]=round(c8rmse,2)
    data[1,0] = round(c8mae,2)
    data[2,0]=round(c8mean,2)
    data[3,0]=round(c8max,2)
    data[4,0] = round(c8min,2)

    data[0,1]=round(c9rmse,2)
    data[1,1] = round(c9mae,2)
    data[2,1]=round(c9mean,2)
    data[3,1]=round(c9max,2)
    data[4,1] = round(c9min,2)


    data[0,2]=round(c10rmse,2)
    data[1,2] = round(c10mae,2)
    data[2,2]=round(c10mean,2)
    data[3,2]=round(c10max,2)
    data[4,2] = round(c10min,2)

    data[0,3]=round(c11rmse,2)
    data[1,3] = round(c11mae,2)
    data[2,3]= round(c11mean,2)
    data[3,3]= round(c11max,2)
    data[4,3] = round(c11min,2)


    data[0,4]=round(c12rmse,2)
    data[1,4] = round(c12mae,2)
    data[2,4]=round(c12mean,2)
    data[3,4]=round(c12max,2)
    data[4,4] = round(c12min,2)

    data[0,5]=round(c13rmse,2)
    data[1,5] = round(c13mae,2)
    data[2,5]=round(c13mean,2)
    data[3,5]=round(c13max,2)
    data[4,5] = round(c13min,2)

    data[0,6]=round(c14rmse,2)
    data[1,6] = round(c14mae,2)
    data[2,6]=round(c14mean,2)
    data[3,6]=round(c14max,2)
    data[4,6] = round(c14min,2)

    data[0,7]=round(c15rmse,2)
    data[1,7] = round(c15mae,2)
    data[2,7]=round(c15mean,2)
    data[3,7]=round(c15max,2)
    data[4,7] = round(c15min,2)

    data[0,8]=round(c16rmse,2)
    data[1,8] = round(c16mae,2)
    data[2,8]=round(c16mean,2)
    data[3,8]=round(c16max,2)
    data[4,8] = round(c16min,2)

    xsl(data)