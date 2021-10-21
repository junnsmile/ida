"""
MODIS DATA READ
black-sky albedo
white-sky albedo
AUTHOR:IDA
DATE:2021-09-14
"""
import numpy as np
import pylab as p
from pyhdf.SD import SD
import netCDF4 as nc
import math

def blacksky(fi,fv,fg,thet,land):
    """
    计算黑空反照率
    """
    pi = 3.1415926
    g0i = 1.0
    g1i = 0.0
    g2i = 0.0
    g0v = -0.007574
    g1v = -0.070987
    g2v = 0.307588
    g0g = -1.284909
    g1g = -0.166314
    g2g = 0.041840
    landper = 50
    the = thet*(math.pi/180)
    af = fi*(g0i+g1i*(the**2)+g2i*(the**3))+fv*(g0v+g1v*(the**2)+g2v*(the**3))+fg*(g0g+g1g*the**2+g2g*(the**3))
    af[np.where(land < landper)] = np.nan
    return af

def whitesky(fi,fv,fg,land):
    """
    计算白空反照率
    """
    gi = 1.0
    gv=  0.189184
    gg = -1.377622
    landper = 50

    af = fi*gi+fv*gv+fg*gg
    af[np.where(land < landper)] = np.nan
    return af

def wave_match(ee_mod,j):
    """
    将MODIS短波波段插值到H8
    """
    wav_cen = [0.4703, 0.5105, 0.6399, 0.8563, 1.6098, 2.257, 3.8848, 6.2383, 6.9395, 7.3471, 8.5905, 9.6347, 10.4029, 11.2432, 12.3828, 13.2844 ]
    mod_cen = [0.469,0.5545,0.645, 0.8585,1.240,1.645,2.12]
    print('tansform '+str(j)+' band')
    if (wav_cen[j]<mod_cen[0]):
        emis = ee_mod[0,:,:]

    elif (wav_cen[j]>mod_cen[6]):
        emis = ee_mod[6,:,:]
    else:
        for ii in range(6):
            if (wav_cen[j]>mod_cen[ii])and(wav_cen[j]<mod_cen[ii+1]):
                i = ii
                print(i)
        emis = (ee_mod[i+1,:,:] - ee_mod[i,:,:]) / (mod_cen[i+1]- mod_cen[i]) * (wav_cen[j] - mod_cen[i]) + ee_mod[i,:,:]

    return emis

def datape(data,LST,name):
    """
    MODIS参数*scale+offset
    缺测赋值NAN
    """
    datas = np.zeros([7,241,241])
    for i in range(7):
        datas[i,:,:] = data[i,:,:] * LST.select(name+str(i+1)).attributes()['scale_factor']+LST.select(name+str(i+1)).attributes()['add_offset']
        datas[i,:,:][np.where(data[i,:,:]==32767)]=np.nan
    return datas

def modisa(modisfile,h8file,landfile):
    # # data = nc.Dataset('D:\work\data\ertm\modis\MOD11C2.A2016065.nc')
    # # print(data.variables.keys())
    # # ll = data['land_per'][:].data
    # filespath = r'D:\work\data\ertm\modis\albedo'
    # file = '\\MCD43C1.A2016065.006.2016196185546.hdf'
    # h8file = r'D:\work\data\ertm\NC_H08_20160305_0700_R21_FLDK.02401_02401.nc'
    # landfile = r'D:\work\data\ertm\modis\MOD11C2.A2016065.006.2016242200752.hdf'

    # filespath = r'D:\work\data\ertm\modis\albedo'
    # file = '\\MCD43C1.A2016259.006.2016272231632.hdf'
    # h8file = r'D:\work\data\ertm\NC_H08_20160916_0800_R21_FLDK.02401_02401.nc'
    # landfile = r'D:\work\data\ertm\modis\MOD11C2.A2016257.006.2016271120238.hdf'


    # 读取MODIS LAND - SEA
    LST0 = SD(landfile)
    land = LST0.select('Percent_land_in_grid').get()[1000:1241,5840:6081]

    # 讀取H8
    ds = nc.Dataset(h8file)
    print(ds.variables.keys())
    SOZ = ds.variables['SOZ'][:].data
    # 讀取要计算地表反照率的MODIS
    LST = SD(modisfile)
    # 输出MODIS keys
    ds_dict = LST.datasets()
    for idx, sds in enumerate(ds_dict.keys()):
        print(idx, sds)
    # 读取参数
    band_p1 = np.zeros([7,241,241],dtype=np.int)
    # [1000: 1241, 5840: 6081]
    band_p1[0,:,:] = LST.select('BRDF_Albedo_Parameter1_Band1').get()[1000:1241,5840:6081]
    band_p1[1,:,:] = LST.select('BRDF_Albedo_Parameter1_Band2').get()[1000:1241,5840:6081]
    band_p1[2,:,:] = LST.select('BRDF_Albedo_Parameter1_Band3').get()[1000:1241,5840:6081]
    band_p1[3,:,:] = LST.select('BRDF_Albedo_Parameter1_Band4').get()[1000:1241,5840:6081]
    band_p1[4,:,:] = LST.select('BRDF_Albedo_Parameter1_Band5').get()[1000:1241,5840:6081]
    band_p1[5,:,:] = LST.select('BRDF_Albedo_Parameter1_Band6').get()[1000:1241,5840:6081]
    band_p1[6,:,:] = LST.select('BRDF_Albedo_Parameter1_Band7').get()[1000:1241,5840:6081]
    band_p2 = np.zeros([7,241,241],dtype=np.int)
    band_p2[0,:,:] = LST.select('BRDF_Albedo_Parameter2_Band1').get()[1000:1241,5840:6081]
    band_p2[1,:,:] = LST.select('BRDF_Albedo_Parameter2_Band2').get()[1000:1241,5840:6081]
    band_p2[2,:,:] = LST.select('BRDF_Albedo_Parameter2_Band3').get()[1000:1241,5840:6081]
    band_p2[3,:,:] = LST.select('BRDF_Albedo_Parameter2_Band4').get()[1000:1241,5840:6081]
    band_p2[4,:,:] = LST.select('BRDF_Albedo_Parameter2_Band5').get()[1000:1241,5840:6081]
    band_p2[5,:,:] = LST.select('BRDF_Albedo_Parameter2_Band6').get()[1000:1241,5840:6081]
    band_p2[6,:,:] = LST.select('BRDF_Albedo_Parameter2_Band7').get()[1000:1241,5840:6081]
    band_p3 = np.zeros([7,241,241],dtype=np.int)
    band_p3[0,:,:] = LST.select('BRDF_Albedo_Parameter3_Band1').get()[1000:1241,5840:6081]
    band_p3[1,:,:] = LST.select('BRDF_Albedo_Parameter3_Band2').get()[1000:1241,5840:6081]
    band_p3[2,:,:] = LST.select('BRDF_Albedo_Parameter3_Band3').get()[1000:1241,5840:6081]
    band_p3[3,:,:] = LST.select('BRDF_Albedo_Parameter3_Band4').get()[1000:1241,5840:6081]
    band_p3[4,:,:] = LST.select('BRDF_Albedo_Parameter3_Band5').get()[1000:1241,5840:6081]
    band_p3[5,:,:] = LST.select('BRDF_Albedo_Parameter3_Band6').get()[1000:1241,5840:6081]
    band_p3[6,:,:] = LST.select('BRDF_Albedo_Parameter3_Band7').get()[1000:1241,5840:6081]

    #
    band_p11 = datape(band_p1,LST,'BRDF_Albedo_Parameter1_Band')
    band_p22 = datape(band_p2,LST,'BRDF_Albedo_Parameter2_Band')
    band_p33 = datape(band_p3,LST,'BRDF_Albedo_Parameter3_Band')

    the = SOZ[400:641,640:881]
    # # H8L1-albedo_01##############
    # albedo_01 = ds.variables['albedo_01'][:].data[400:641,640:881]
    # albedo_01[np.where(land<50)]=np.nan
    #############################
    # 黑空反照率
    af = np.zeros([7,241,241])
    for i in range(7):
        af[i,:,:] = blacksky(band_p11[i,:,:],band_p22[i,:,:],band_p33[i,:,:],the,land)

    # MODIS波段插值波段到H8
    albeho = np.zeros([6,241,241])
    for k in range(6):
        albeho[k,:,:] = wave_match(af,k)

    print(af)
    ####################################
    # 白空反照率
    wmaf = np.zeros([7,241,241])
    for i in range(7):
        wmaf[i,:,:] = whitesky(band_p11[i,:,:],band_p22[i,:,:],band_p33[i,:,:],land)

    # MODIS波段插值波段到H8
    wh8albeho = np.zeros([6,241,241])
    for k in range(6):
        wh8albeho[k,:,:] = wave_match(af,k)
    return albeho,af,wmaf,wh8albeho,land