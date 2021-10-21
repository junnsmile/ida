"""
era5 profile analysis, save as HDF5, xarray interpolate
author:ida
date:2020-07-01
version:2.0
"""

from pyhdf.SD import SD, SDC  # hdf
import xarray as xr           # nc
import numpy as np
import netCDF4 as nc
import skimage.transform
import h5py
import time

def inter(data):
    xsize =6001
    ysize=6001
    shapex = data.shape[2]
    shapey = data.shape[3]
    tem = data[0,:,:,:]
    temp=xr.DataArray(data,[('time',np.arange(0,24,1)),('level',np.arange(0,47,1)),\
                            ('x',np.arange(0,xsize,1)),('y',np.arange(0,ysize,1))])
    data0=temp.interp(time = np.arange(0,23.5,0.5),level=,longitude=6001,latitude=6001,method = 'linear')
    #data0 = skimage.transform.resize(tem, (24,37,6001, 6001), order=1)
    print(data0.shape)
    print(data0.dtype)
    print(data0)
    return data0

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

def ncreadERA5Profile(filename):
    '''
    @description: read era5 profile o3
    '''
    start = time.time()
    ds = nc.Dataset(filename)
    print(ds.variables.keys())
    o3 = ds.variables['o3'][:].data
    o3off = getattr(ds['o3'], 'add_offset')
    o3scale= getattr(ds['o3'], 'scale_factor')
    o33 = o3*o3scale+o3off
    q = ds.variables['q'][:].data
    qoff = getattr(ds['q'], 'add_offset')
    qscale= getattr(ds['q'], 'scale_factor')
    qq = q*qscale+qoff
    t = ds.variables['t'][:].data
    toff = getattr(ds['t'], 'add_offset')
    tscale= getattr(ds['t'], 'scale_factor')
    tt = t*tscale+toff
    end = time.time()
    print('Running time: %s Seconds' % (end - start))
    return o33, qq, tt

if __name__ == "__main__":
    start= time.time()
    filename5 = r'/home/data/ERA5/profile/era5_20170117.nc'
    o3, q, t = ncreadERA5Profile(filename5)
    o33 = inter(o3)
    f= h5py.File('era5profile.hdf5')
    f['/2/o3']=o33
    #end1 = time.time()

    #print('o3 running time: %s seconds' % (end1-start))
    #qq = inter(q)
    #f['/1/q']=qq
    #end2 = time.time()
    #print('h2o running time:%s seconds' % (end2 - end1))
    #tt = inter(t)
    #f['/1/t']=tt
    end = time.time()
    #print('temperature running time:%s seconds' % (end - end2))
    print('all running: %s seconds' % (end-start)) 
