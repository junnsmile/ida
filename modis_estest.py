"""
land surface emissivity nan filling----result test
author:ida
date:2021-08-18
"""

import os
import netCDF4 as nc
import numpy as np

def test(path):
    files = os.listdir(path)
    for file in files:
        data = nc.Dataset(path+'/'+file)
        e20  = data.variables['e20'][:]
        si = np.where(e20==0.49)
        n = np.where(np.isnan(e20))
        print(str(file)+'**0.49数量为：'+str(len(si[0]))+'nan数量为：'+str(len(n[0])))

path = r'/home/data/ertmAuxData/ncPath/005/mod11e'
test(path)
