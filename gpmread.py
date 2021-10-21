"""

read GPM data
author:ida
date:2021.07.09
"""

import h5py

path = r'D:\work\data\ertm'
file = '\\3B-HHR.MS.MRG.3IMERG.20160301-S083000-E085959.0510.V06B.HDF5'
data = h5py.File(path+file)
print(data.keys())
group = data['Grid']
data0 = data['/Grid/precipitationCal'][:]
print('complete!')