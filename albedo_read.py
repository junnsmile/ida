"""
READ RTTOV ALBEDO,MODISALBEDO,BRDF
AUTHORl:ida
DATE:2021-09-17
"""

import pandas as pd
import os
import numpy as np
import netCDF4 as nc

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
    az = np.zeros([7,int((len(df_train)+1)/33)-1],dtype=np.float64)
    abz = np.zeros([6, int((len(df_train) + 1) / 33) - 1], dtype=np.float64)
    maz = np.zeros([7, int((len(df_train) + 1) / 33) - 1], dtype=np.float64)
    for i in range(int((len(df_train)+1)/33)-1):
        print(i)
        # longitude = np.array(df_train.iloc[10*i+4])[0][16:23]
        # latitude = np.array(df_train.iloc[10*i+3])[0][17:23]
        allbrdf = np.array(df_train.iloc[(33*i+18+59):(33*i+20+59)])
        albedo = np.array(df_train.iloc[(33 * i + 22 + 59):(33 * i + 24 + 59)])
        modisalbedo = np.array(df_train.iloc[(33 * i + 26 + 59):(33 * i + 28 + 59)])
        # a0 =np.array(allbrdf[0][0].split('  '))
        # a1 = np.array(allbrdf[1][0].split('  '))
        # ab0 =np.array(albedo[0][0].split('  '))
        # ab1 = np.array(albedo[1][0].split('  '))
        # ma0 =np.array(modisalbedo[0][0].split('  '))
        # ma1 = np.array(modisalbedo[1][0].split('  '))
        # all brdf
        az[0,i]=np.array(allbrdf[0][0][0:8])
        az[1,i]= np.array(allbrdf[0][0][8:16])
        az[2,i]= np.array(allbrdf[0][0][16:24])
        az[3, i] = np.array(allbrdf[0][0][24:32])
        az[4, i]= np.array(allbrdf[0][0][32:40])
        az[5, i]= np.array(allbrdf[0][0][40:48])
        az[6, i] = np.array(allbrdf[0][0][48:56])
        # all abedo
        abz[0,i]=np.array(albedo[0][0][0:8])
        abz[1,i]= np.array(albedo[0][0][8:16])
        abz[2,i]= np.array(albedo[0][0][16:24])
        abz[3, i] = np.array(albedo[0][0][24:32])
        abz[4, i]= np.array(albedo[0][0][32:40])
        abz[5, i]= np.array(albedo[0][0][40:48])
        # modis albedo
        maz[0,i]=np.array(modisalbedo[0][0][0:8])
        maz[1,i]= np.array(modisalbedo[0][0][8:16])
        maz[2,i]= np.array(modisalbedo[0][0][16:24])
        maz[3, i] = np.array(modisalbedo[0][0][24:32])
        maz[4, i]= np.array(modisalbedo[0][0][32:40])
        maz[5, i]= np.array(modisalbedo[0][0][40:48])
        maz[6, i] = np.array(modisalbedo[0][0][48:56])

    azall = az.reshape(7,241,241)
    abzall = abz.reshape(6, 241, 241)
    mazall = maz.reshape(7, 241, 241)
    return azall,abzall,mazall


def rttota(path):
    ## read rttov.dat

    rtdatas = open(path,encoding='utf-8')
    az,abz,maz = rttovdat(rtdatas)
    a0 = az[3,:,:].T
    b = abz[4,:,:].T
    c = maz[2,:,:].T
    # rtdatas2 = open(r'D:\work\data\ertm\rttov\2016091608.dat',encoding='utf-8')
    # az2,abz2,maz2 = rttovdat(rtdatas2)
    # a2 = az2[3,:,:].T
    # b2 = abz2[4,:,:].T
    # c2 = maz2[2,:,:].T
    return az.T,abz.T,maz.T

# rttota()

