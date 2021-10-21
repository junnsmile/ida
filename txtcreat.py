"""
read filepath and list the catalog in .txt
author:ida
date:2021-10-18
"""

import os
import netCDF4 as nc

ncpath = r'D:\work\data\ertm\japan\allnc'
filepath = r'D:\work\data\ertm\japan\filepath'
f = open(r'D:\work\data\ertm\japan\data.txt','w')
i = 0
files  = os.listdir(filepath)
for file in files:
    name = file.split('.dat')[0]
    f.write(str(name))
    f.write(' ')
    ds = nc.Dataset(ncpath+'\\'+name.split('_pro')[0]+'.nc')
    lon = ds.variables['lon'][:].data
    f.write(str(len(lon)))
    f.write('\n')
f.close()
