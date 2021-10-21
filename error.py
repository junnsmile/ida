import os
import netCDF4 as nc
import numpy as np

path = '/home/data/FY_ERA5/fync'
files  = os.listdir(path)
for file in files:
    data = nc.Dataset(path+'/'+file)
    lon = data.variables['lon'][:].data
    lat = data.variables['lat'][:].data
    print(file)
    print(np.shape(lon)[0])
    # if np.shape(lon)[0]!=np.shape(lat)[0]:
    #     os.remove(path+'/'+file)
    # else:
    #     continue