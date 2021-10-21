import cdsapi
import numpy as np
import calendar
import os

os.chdir("/home/data/ERA5/uvw/")
c = cdsapi.Client()

# define the years you want to download
yearstart = 2016
yearend = 2017
# define the start and end month you want to download
monthstart = 1
monthend = 12
# define the start and end day you want to download
daystart = 1
dayend = 31

years = np.array(range(yearstart, yearend + 1), dtype="str")

for year in years:
    if (int(year) == yearstart) and (int(year) == yearend):
        months = np.array(range(monthstart, monthend + 1), dtype="str")
    elif (year == yearstart):
        months = np.array(range(monthstart, 13), dtype="str")
    elif (year == yearend):
        months = np.array(range(1, monthend + 1), dtype="str")
    else:
        months = np.array(range(1, 13), dtype="str")

    for month in months:
        m = '{:0>2}'.format(str(month))  # 数字补零 (填充左边, 宽度为2)
        # if int(month) < 10:
        #     m = '0' + month
        # else:
        #     m = month

        if (int(year) == yearstart) and (int(year) == yearend) and (int(month) == monthstart) and (
                int(month) == monthend):
            days = list(np.array(range(daystart, dayend + 1), dtype="str"))
        elif (int(year) == yearstart) and (int(month) == monthstart):
            days = list(np.array(range(daystart, calendar.monthrange(int(year), int(month))[1] + 1), dtype="str"))
        elif (int(year) == yearend) and (int(month) == monthend):
            days = list(np.array(range(1, dayend + 1), dtype="str"))
        else:
            days = list(np.array(range(1, calendar.monthrange(int(year), int(month))[1] + 1), dtype="str"))

        for day in days:
            d = '{:0>2}'.format(str(day))
            if os.path.exists('era5_' + year + m + d + '.nc') == True:
                continue
            else:
                # if int(day) < 10:
                #     d = '0' + day
                # else:
                #     d = day

                c.retrieve(
                    'reanalysis-era5-pressure-levels',
                    {
                        'variable': ['relative_humidity', 'u_component_of_wind', 'v_component_of_wind','vertical_velocity',],
                        'product_type': 'reanalysis',
                        'pressure_level': [
                            '1', '2', '3',
                            '5', '7', '10',
                            '20', '30', '50',
                            '70', '100', '125',
                            '150', '175', '200',
                            '225', '250', '300',
                            '350', '400', '450',
                            '500', '550', '600',
                            '650', '700', '750',
                            '775', '800', '825',
                            '850', '875', '900',
                            '925', '950', '975',
                            '1000',
                        ],
                        'year': year,
                        'month': month,
                        'day': day,
                        'time': [
                            '00:00', '01:00', '02:00',
                            '03:00', '04:00', '05:00',
                            '06:00', '07:00', '08:00',
                            '09:00', '10:00', '11:00',
                            '12:00', '13:00', '14:00',
                            '15:00', '16:00', '17:00',
                            '18:00', '19:00', '20:00',
                            '21:00', '22:00', '23:00'
                        ],
                        'area': [
                            60, 80, -60,
                            200,
                        ],
                        'format': 'netcdf',
                    },
                    'era5_' + year + m + d + '.nc')
