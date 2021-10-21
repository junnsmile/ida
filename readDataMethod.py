'''
Description: read data for project
Author: LJW
LastEditTime: 2021-07-01 14:45:01
LastEditors: LJW
'''
from pyhdf.SD import SD, SDC  # hdf
import xarray as xr           # nc
import numpy as np

def readimawariL1(filename):
    '''
    @desc: read nc: himawari-L1
    '''    
    ds = xr.open_dataset(filename)
    y = ds['tbb_07'][:].shape[0]
    x = ds['tbb_07'][:].shape[1]
    arr = np.zeros((y, x, 10))
    for i in range(7, 17):
        dataName = 'tbb_' + '{:0>2}'.format(str(i))
        arr[:, :, i-7] = ds[dataName][:]
    ds.close()
    return arr

def readimawariL2(filename):
    '''
    @desc: read nc: himawari-L2
    '''    
    ds = xr.open_dataset(filename)
    y = ds['CLOT'][:].shape[0]
    x = ds['CLOT'][:].shape[1]
    keys = ['CLER_23', 'CLOT', 'CLTH', 'CLTT', 'CLTYPE']
    arr = np.zeros((y, x, 5))
    count = 0
    for key in keys:
        arr[:, :, count] = ds[key][:]
        count = count + 1
    ds.close()
    return arr

def readMOD11(filename):
    '''
    @read hdf:pyhdf读取hdf数据集
    '''
    SD_file = SD(filename)
    ds_dict = SD_file.datasets()
    for idx, sds in enumerate(ds_dict.keys()):
        print(idx, sds)

    emis20 = SD_file.select('Emis_20')[:] 
    emis22 = SD_file.select('Emis_22')[:] 
    emis23 = SD_file.select('Emis_23')[:] 
    emis29 = SD_file.select('Emis_29')[:] 
    emis31 = SD_file.select('Emis_31')[:] 
    emis32 = SD_file.select('Emis_32')[:] 

    arr = np.zeros((emis20.shape[0], emis20.shape[1], 6))
    arr[:, :, 0] = emis20
    arr[:, :, 1] = emis22
    arr[:, :, 2] = emis23
    arr[:, :, 3] = emis29
    arr[:, :, 4] = emis31
    arr[:, :, 5] = emis32

    SD_file.end()
    return arr

def readERA5Profile(filename):
    '''
    @description: read era5 profile o3
    '''
    ds = xr.open_dataset(filename)
    print(ds.keys())
    o3 = ds['o3']
    q = ds['q']
    t = ds['t']
    return o3, q, t

def readERA5Surface(filename):
    '''
    @description: read era5 surfacepressure/temperature 
    '''    
    ds = xr.open_dataset(filename)
    print(ds.keys())
    sp = ds['sp']
    skt = ds['skt']
    return sp, skt


if __name__ == "__main__":
    # filename = 'E:/Tablefile/NC_H08_20160102_0010_R21_FLDK.06001_06001.nc'
    # data = readimawariL1(filename)
    # print(data.shape)

    # filename2 = 'E:/Tablefile/NC_H08_20180730_2300_L2CLPbet_FLDK.02401_02401.nc'
    # data = readimawariL2(filename2)
    # print(data)

    # filename3 = r'D:\work\data\ertm\modis\MOD11C3.A2017335.006.2018003155403.hdf'
    # data = readMOD11(filename3)
    # print(data)

    # filename4 = r'E:\Tablefile\cnnProject\surfacedata\era5_sf20170104.nc'
    # sp, skt = readERA5Surface(filename4)
    #
    filename5 = r'D:\work\data\ertm\profile\era5_20170103.nc'
    o3, q, t = readERA5Profile(filename5)
