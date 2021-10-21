"""
clear sky BT result
error analysis

author：ida
date:2021-07-15
"""

import netCDF4 as nc
import numpy as np
import os
import matplotlib.pyplot as plt

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
    scc = plt.scatter(llon,llat,c=data,marker='o',cmap='Wistia')
    plt.colorbar(scc)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('BT Difference of ' + str(name))
    plt.savefig('D:\\work\\code\\ertm\\result\\1_'+str(name)+str(date))
    # plt.show()
    plt.close()
    print(str(name) + '**draw sash success!')

path = r'D:\work\data\ertm'
# for f in os.listdir(path):
# path0 = path+'/'+f
# for file in os.listdir(path0):
file = '\\NC_H08_20160305_0700_L2CLPbet_FLDK.02401_02401.nc'
file2 = 'D:\\work\\data\\ertm\\bt\\2016030507_modis_bt.nc'
file3 = '\\NC_H08_20160305_0700_R21_FLDK.02401_02401.nc'

data =nc.Dataset(path+'\\'+file)
data2 =nc.Dataset(file2)
data3 = nc.Dataset(path+file3)
print(data.variables.keys())
print(data2.variables.keys())
print(data3.variables.keys())

cltype = data.variables['CLTYPE'][:].data
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
c8= abs(BT8 -bt8)
c8[np.where((BT8<-100)|(BT8>500))] = np.nan
c8[np.where(cltype[399:640,639:880]!=0)]=np.nan

bt9 = data3['tbb_09'][:].data[399:640,639:880]
c9= abs(BT9 -bt9)
c9[np.where((BT9<-100)|(BT9>500))] = np.nan
c9[np.where(cltype[399:640,639:880]!=0)]=np.nan

bt10 = data3['tbb_10'][:].data[399:640,639:880]
c10= abs(BT10 -bt10)
c10[np.where((BT10<-100)|(BT10>500))] = np.nan
c10[np.where(cltype[399:640,639:880]!=0)]=np.nan

bt11 = data3['tbb_11'][:].data[399:640,639:880]
c11= abs(BT11 -bt11)
c11[np.where((BT11<-100)|(BT11>500))] = np.nan
c11[np.where(cltype[399:640,639:880]!=0)]=np.nan

bt12 = data3['tbb_12'][:].data[399:640,639:880]
c12= abs(BT12 -bt12)
c12[np.where((BT12<-100)|(BT12>500))] = np.nan
c12[np.where(cltype[399:640,639:880]!=0)]=np.nan

bt13 = data3['tbb_13'][:].data[399:640,639:880]
c13= abs(BT13 -bt13)
c13[np.where((BT13<-100)|(BT13>500))] = np.nan
c13[np.where(cltype[399:640,639:880]!=0)]=np.nan

bt14 = data3['tbb_14'][:].data[399:640,639:880]
c14= abs(BT14 -bt14)
c14[np.where((BT14<-100)|(BT14>500))] = np.nan
c14[np.where(cltype[399:640,639:880]!=0)]=np.nan

bt15 = data3['tbb_15'][:].data[399:640,639:880]
c15= abs(BT15 -bt15)
c15[np.where((BT15<-100)|(BT15>500))] = np.nan
c15[np.where(cltype[399:640,639:880]!=0)]=np.nan

bt16 = data3['tbb_16'][:].data[399:640,639:880]
c16= abs(BT16 -bt16)
c16[np.where((BT16<-100)|(BT16>500))] = np.nan
c16[np.where(cltype[399:640,639:880]!=0)]=np.nan

lon = data2.variables['lon'][:].data
lat = data2.variables['lat'][:].data

llon,llat = np.meshgrid(lon, lat)
name = os.path.split(file2)[1].split('_bt')[0]
#绘制散点图
drawSashdiagram(llon,llat,c8,'c8','_'+str(name)+'.jpg')
drawSashdiagram(llon,llat,c9,'c9','_'+str(name)+'.jpg')
drawSashdiagram(llon,llat,c10,'c10','_'+str(name)+'.jpg')
drawSashdiagram(llon,llat,c11,'c11','_'+str(name)+'.jpg')
drawSashdiagram(llon,llat,c12,'c12','_'+str(name)+'.jpg')
drawSashdiagram(llon,llat,c13,'c13','_'+str(name)+'.jpg')
drawSashdiagram(llon,llat,c14,'c14','_'+str(name)+'.jpg')
drawSashdiagram(llon,llat,c15,'c15','_'+str(name)+'.jpg')
drawSashdiagram(llon,llat,c16,'c16','_'+str(name)+'.jpg')
#
# #绘制柱状图
drawHist(c8.flatten(),'c8','_'+str(name)+'.jpg')
drawHist(c9.flatten(),'c9','_'+str(name)+'.jpg')
drawHist(c10.flatten(),'c10','_'+str(name)+'.jpg')
drawHist(c11.flatten(),'c11','_'+str(name)+'.jpg')
drawHist(c12.flatten(),'c12','_'+str(name)+'.jpg')
drawHist(c13.flatten(),'c13','_'+str(name)+'.jpg')
drawHist(c14.flatten(),'c14','_'+str(name)+'.jpg')
drawHist(c15.flatten(),'c15','_'+str(name)+'.jpg')
drawHist(c16.flatten(),'c16','_'+str(name)+'.jpg')


print(c9.max(),c10.max(),c11.max(),c12.max(),c13.max(),c14.max(),c15.max(),c16.max())
print(c9.min(),c10.min(),c11.min(),c12.min(),c13.min(),c14.min(),c15.min(),c16.min())

