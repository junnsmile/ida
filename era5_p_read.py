"""
era5 profile data read
481*481 trans 6001*6001
author:ida
date:2021-07-05
"""
import time
import netCDF4 as nc
from scipy.interpolate import griddata
import numpy as np

def interpola(o3,output,points,x,y):
    data = np.zeros([np.shape(o3)[1], xshape, yshape], dtype=np.float64)
    for t in range(np.shape(o3)[0]):
        for p in range(np.shape(o3)[1]):
            # for i in range(len(x)):
            #     for j in range(len(y)):
            data[p,:,:] = griddata(points, o3[t][p].flatten(), (output[0], output[1]), method='nearest')
        name = open("D:\work\code\ertm\o3_" + str(t) + '.txt', 'w')
        for j in range(len(x)):
            for i in range(len(y)):
                for pp in range(np.shape(o3)[1]):
                    name.write(str(data[pp,i,j])+'\t')
            name.write('\n')
        name.close()

        # name = "D:\work\code\ertm\o3_"+str(t)+'_'+str(p)+'.txt'
        # np.savetxt(name, np.c_[data],fmt='%f', delimiter='\t')

    print(data.shape)
    print(data.max())
    print(data.min())

    return data

def ncreadERA5Profile(filename,xshape,yshape):
    '''
    @description: read era5 profile o3
    '''
    start = time.time()
    ds = nc.Dataset(filename)
    print(ds.variables.keys())
    lon = ds.variables['longitude'][:]
    lat = ds.variables['latitude'][:]
    xsize = (lon[-1]-lon[0])/xshape
    ysize = (lat[-1]-lat[0])/yshape
    x = np.arange(lon[0],lon[-1],xsize)
    y =np.arange(lat[0],lat[-1],ysize)
    output = np.meshgrid(x,y)
    lonn,latt = np.meshgrid(np.array(lon),np.array(lat))
    points = np.array((lonn.flatten(), latt.flatten())).T
    o3 = ds.variables['o3'][:].data
    o33 = interpola(o3,output,points,x,y)


    # o3 = ds.variables['o3'][:].data
    # o3off = getattr(ds['o3'], 'add_offset')
    # o3scale= getattr(ds['o3'], 'scale_factor')
    # o33 = o3*o3scale+o3off
    # q = ds.variables['q'][:].data
    # qoff = getattr(ds['q'], 'add_offset')
    # qscale= getattr(ds['q'], 'scale_factor')
    # qq = q*qscale+qoff
    # t = ds.variables['t'][:].data
    # toff = getattr(ds['t'], 'add_offset')
    # tscale= getattr(ds['t'], 'scale_factor')
    # tt = t*tscale+toff
    end = time.time()
    print('Running time: %s Seconds' % (end - start))
    # return o33, qq, tt

if __name__ == "__main__":
    start= time.time()
    filename5 = r'D:\work\data\ertm\profile\era5_20170103.nc'
    xshape = 6001
    yshape = 6001
    ncreadERA5Profile(filename5,xshape,yshape)

