"""
download era5 2m-temperature,10m-u,10m-v

author:ida
date:2021-08-24
"""

import cdsapi
import numpy as np
import calendar
import os

# os.chdir("/home/data/FY_ERA5")
c = cdsapi.Client()

dates = {20181000,20181100,20181126,20181200,20190100,20190200,20190300,20190400,20190500,20190600,20190700}

for date in dates:
    year = str(date)[0:4]
    month =str(date)[4:6]
    day = str(date)[6:8]
    m = int(month)
    d = int(day)
    if os.path.exists('era5_' + year + month + day + '.nc') == True:
        continue
    else:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
                ],
                'area': [
                    90, 15, -90,
                    200,
                ],
                'year': year,
                'month':month,
                'day': day,
                'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                ],
            },
            'era5_' + year + month + day + '.nc')



