"""
water cloud --optical property of particle
author:ida
date:2021-11-03
"""

import scipy.integrate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import gamma
import math

def tesdf(rdata,nr):
    '''
      计算Reff
      验证粒子群C2和G的结果时，作为横轴
    '''
    ddata = rdata
    dfz = 0
    dfm = 0
    for i in range(len(rdata)-1):
        dfz = dfz+(((4 / 3) * np.pi * ddata[i + 1] ** 3*nr[i + 1]) + ((4 / 3) * np.pi * ddata[i] ** 3*nr[i])) * (
                ddata[i + 1] - ddata[i]) * 1 / 2
        dfm = dfm+((np.pi * ddata[i + 1] ** 2*nr[i + 1]) + (np.pi * ddata[i] ** 2*nr[i])) * (ddata[i + 1] - ddata[i]) * 1 / 2

    tem = 3*dfz / (4*dfm)
    return tem

def tesdfx(rdata,nr):
    '''
      计算Reff
      验证粒子群C2和G的结果时，作为横轴
    '''
    ddata = rdata
    dfz = 0
    dfm = 0
    for i in range(len(rdata)-1):
        dfz = dfz+((ddata[i+1] ** 3*nr[i+1])+(ddata[i] ** 3*nr[i]))*(ddata[i+1]-ddata[i])*1/2
        dfm = dfm+((ddata[i+1] ** 2*nr[i+1])+(ddata[i] ** 2*nr[i]))*(ddata[i+1]-ddata[i])*1/2

    tem = dfz / dfm
    return tem

def refff(rdata,q_data,w,nr,c):

    """
    refff(rdata,q_data,w,nr[q,:],ctest[:,p])
    rdata:粒子半径
    q_data:消光系数
    w:单次散射反照率
    nr
    c:勒让德展开系数
    返回： 粒子群勒让德系数
    """
    fz = 0
    fm = 0
    for i in range(len(rdata)-1):
        fz = fz + (((np.pi * rdata[i + 1] ** 2) * q_data[i + 1] * w[i + 1] * c[i + 1] * nr[i + 1]) + ( \
                      (np.pi * rdata[i] ** 2) * q_data[i] * w[i] * c[i] * nr[i])) * (rdata[i + 1] - rdata[i]) * 1 / 2
        fm = fm + (((np.pi*rdata[i+1]**2)*q_data[i+1]*w[i+1]*nr[i+1]) +((np.pi*rdata[i]**2)*q_data[i]*w[i]*nr[i]))\
              *(rdata[i+1]-rdata[i])*1/2
    creff = fz / fm
    return creff

def gg(q_data,ww,asf,nr,rdata):
    fz = 0
    fm = 0
    for i in range(len(rdata)-1):
        fz = fz + (((np.pi * rdata[i + 1] ** 2) * q_data[i + 1] * ww[i + 1] * asf[i + 1] * nr[i + 1]) + ((np.pi * rdata[i] ** 2) * q_data[i] * ww[i] * asf[i] * nr[i])) * (rdata[i + 1] - rdata[i]) * 1 / 2
        fm = fm + (((np.pi * rdata[i + 1] ** 2)*q_data[i+1]*ww[i+1]*nr[i+1]) +((np.pi * rdata[i] ** 2)*q_data[i]*ww[i]*nr[i]))*(rdata[i+1]-rdata[i])*1/2
    gg = fz / fm
    return gg

def reffx(rdata,q_data,w,nr,c):

    """
    refff(rdata,q_data,w,nr[q,:],ctest[:,p])
    rdata:粒子半径
    q_data:消光系数
    w:单次散射反照率
    nr
    c:勒让德展开系数
    返回： 粒子群勒让德系数
    """
    fz = 0
    fm = 0
    for i in range(len(rdata)-1):
        fz = fz + ((np.pi * rdata[i] ** 2) * q_data[i] * w[i] * c[i] * nr[i])
        fm = fm + ((np.pi*rdata[i]**2)*q_data[i]*w[i]*nr[i])
    creff = fz / fm
    return creff

# def cn(tht,x,p,n):
#     c = 0
#     for i in range(len(x)):
#         c = c + (fx[i]*p[i]+fx[i+1]*p[i+1])*(x[i+1]-x[i])*1/2
#
#     cn = ((2 * n + 1) / 2)*c
#     return cn

veff = 1/3 #方差
reff1 = np.arange(0.01,10,0.01)
reff2 = np.arange(10,70,0.01)
reff = np.hstack((reff1,reff2)) #有效粒子半径
N = 2000

# tht = np.arange(0,190,0.001) #角度
tht = [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.0125,
      0.015, 0.0175, 0.02, 0.0225, 0.025, 0.0275,
      0.03, 0.0325, 0.035, 0.0375, 0.04, 0.0425,
      0.045, 0.0475, 0.05, 0.0525, 0.055, 0.0575,
      0.06, 0.0625, 0.065, 0.0675, 0.07, 0.0725,
      0.075, 0.0775, 0.08, 0.0825, 0.085, 0.0875,
      0.09, 0.0925, 0.095, 0.0975, 0.1, 0.1025,
      0.105, 0.1075, 0.11, 0.1125, 0.115, 0.1175,
      0.12, 0.1225, 0.125, 0.1275, 0.13, 0.1325,
      0.135, 0.1375, 0.14, 0.1425, 0.145, 0.1475,
      0.15, 0.1525, 0.155, 0.1575, 0.16, 0.1625,
      0.165, 0.1675, 0.17, 0.1725, 0.175, 0.1775,
      0.18, 0.1825, 0.185, 0.1875, 0.19, 0.1925,
      0.195, 0.1975, 0.2, 0.2025, 0.205, 0.2075,
      0.21, 0.2125, 0.215, 0.2175, 0.22, 0.2225,
      0.225, 0.2275, 0.23, 0.2325, 0.235, 0.2375,
      0.24, 0.2425, 0.245, 0.2475, 0.25, 0.2525,
      0.255, 0.2575, 0.26, 0.2625, 0.265, 0.2675,
      0.27, 0.2725, 0.275, 0.2775, 0.28, 0.2825,
      0.285, 0.2875, 0.29, 0.2925, 0.295, 0.2975,
      0.3, 0.3025, 0.305, 0.3075, 0.31, 0.3125,
      0.315, 0.3175, 0.32, 0.3225, 0.325, 0.3275,
      0.33, 0.3325, 0.335, 0.3375, 0.34, 0.3425,
      0.345, 0.3475, 0.35, 0.3525, 0.355, 0.3575,
      0.36, 0.3625, 0.365, 0.3675, 0.37, 0.3725,
      0.375, 0.3775, 0.38, 0.3825, 0.385, 0.3875,
      0.39, 0.3925, 0.395, 0.3975, 0.4, 0.4025,
      0.405, 0.4075, 0.41, 0.4125, 0.415, 0.4175,
      0.42, 0.4225, 0.425, 0.4275, 0.43, 0.4325,
      0.435, 0.4375, 0.44, 0.4425, 0.445, 0.4475,
      0.45, 0.4525, 0.455, 0.4575, 0.46, 0.4625,
      0.465, 0.4675, 0.47, 0.4725, 0.475, 0.4775,
      0.48, 0.4825, 0.485, 0.4875, 0.49, 0.4925,
      0.495, 0.4975, 0.5, 0.505, 0.51, 0.515,
      0.52, 0.525, 0.53, 0.535, 0.54, 0.545,
      0.55, 0.555, 0.56, 0.565, 0.57, 0.575,
      0.58, 0.585, 0.59, 0.595, 0.6, 0.605,
      0.61, 0.615, 0.62, 0.625, 0.63, 0.635,
      0.64, 0.645, 0.65, 0.655, 0.66, 0.665,
      0.67, 0.675, 0.68, 0.685, 0.69, 0.695,
      0.7, 0.705, 0.71, 0.715, 0.72, 0.725,
      0.73, 0.735, 0.74, 0.745, 0.75, 0.755,
      0.76, 0.765, 0.77, 0.775, 0.78, 0.785,
      0.79, 0.795, 0.8, 0.805, 0.81, 0.815,
      0.82, 0.825, 0.83, 0.835, 0.84, 0.845,
      0.85, 0.855, 0.86, 0.865, 0.87, 0.875,
      0.88, 0.885, 0.89, 0.895, 0.9, 0.905,
      0.91, 0.915, 0.92, 0.925, 0.93, 0.935,
      0.94, 0.945, 0.95, 0.955, 0.96, 0.965,
      0.97, 0.975, 0.98, 0.985, 0.99, 0.995,
      1.0, 1.01, 1.02, 1.03, 1.04, 1.05,
      1.06, 1.07, 1.08, 1.09, 1.1, 1.11,
      1.12, 1.13, 1.14, 1.15, 1.16, 1.17,
      1.18, 1.19, 1.2, 1.21, 1.22, 1.23,
      1.24, 1.25, 1.26, 1.27, 1.28, 1.29,
      1.3, 1.31, 1.32, 1.33, 1.34, 1.35,
      1.36, 1.37, 1.38, 1.39, 1.4, 1.41,
      1.42, 1.43, 1.44, 1.45, 1.46, 1.47,
      1.48, 1.49, 1.5, 1.51, 1.52, 1.53,
      1.54, 1.55, 1.56, 1.57, 1.58, 1.59,
      1.6, 1.61, 1.62, 1.63, 1.64, 1.65,
      1.66, 1.67, 1.68, 1.69, 1.7, 1.71,
      1.72, 1.73, 1.74, 1.75, 1.76, 1.77,
      1.78, 1.79, 1.8, 1.81, 1.82, 1.83,
      1.84, 1.85, 1.86, 1.87, 1.88, 1.89,
      1.9, 1.91, 1.92, 1.93, 1.94, 1.95,
      1.96, 1.97, 1.98, 1.99, 2.0, 2.01,
      2.02, 2.03, 2.04, 2.05, 2.06, 2.07,
      2.08, 2.09, 2.1, 2.11, 2.12, 2.13,
      2.14, 2.15, 2.16, 2.17, 2.18, 2.19,
      2.2, 2.21, 2.22, 2.23, 2.24, 2.25,
      2.26, 2.27, 2.28, 2.29, 2.3, 2.31,
      2.32, 2.33, 2.34, 2.35, 2.36, 2.37,
      2.38, 2.39, 2.4, 2.41, 2.42, 2.43,
      2.44, 2.45, 2.46, 2.47, 2.48, 2.49,
      2.5, 2.51, 2.52, 2.53, 2.54, 2.55,
      2.56, 2.57, 2.58, 2.59, 2.6, 2.61,
      2.62, 2.63, 2.64, 2.65, 2.66, 2.67,
      2.68, 2.69, 2.7, 2.71, 2.72, 2.73,
      2.74, 2.75, 2.76, 2.77, 2.78, 2.79,
      2.8, 2.81, 2.82, 2.83, 2.84, 2.85,
      2.86, 2.87, 2.88, 2.89, 2.9, 2.91,
      2.92, 2.93, 2.94, 2.95, 2.96, 2.97,
      2.98, 2.99, 3.0, 3.02, 3.04, 3.06,
      3.08, 3.1, 3.12, 3.14, 3.16, 3.18,
      3.2, 3.22, 3.24, 3.26, 3.28, 3.3,
      3.32, 3.34, 3.36, 3.38, 3.4, 3.42,
      3.44, 3.46, 3.48, 3.5, 3.52, 3.54,
      3.56, 3.58, 3.6, 3.62, 3.64, 3.66,
      3.68, 3.7, 3.72, 3.74, 3.76, 3.78,
      3.8, 3.82, 3.84, 3.86, 3.88, 3.9,
      3.92, 3.94, 3.96, 3.98, 4.0, 4.02,
      4.04, 4.06, 4.08, 4.1, 4.12, 4.14,
      4.16, 4.18, 4.2, 4.22, 4.24, 4.26,
      4.28, 4.3, 4.32, 4.34, 4.36, 4.38,
      4.4, 4.42, 4.44, 4.46, 4.48, 4.5,
      4.52, 4.54, 4.56, 4.58, 4.6, 4.62,
      4.64, 4.66, 4.68, 4.7, 4.72, 4.74,
      4.76, 4.78, 4.8, 4.82, 4.84, 4.86,
      4.88, 4.9, 4.92, 4.94, 4.96, 4.98,
      5.0, 5.02, 5.04, 5.06, 5.08, 5.1,
      5.12, 5.14, 5.16, 5.18, 5.2, 5.22,
      5.24, 5.26, 5.28, 5.3, 5.32, 5.34,
      5.36, 5.38, 5.4, 5.42, 5.44, 5.46,
      5.48, 5.5, 5.52, 5.54, 5.56, 5.58,
      5.6, 5.62, 5.64, 5.66, 5.68, 5.7,
      5.72, 5.74, 5.76, 5.78, 5.8, 5.82,
      5.84, 5.86, 5.88, 5.9, 5.92, 5.94,
      5.96, 5.98, 6.0, 6.04, 6.08, 6.12,
      6.16, 6.2, 6.24, 6.28, 6.32, 6.36,
      6.4, 6.44, 6.48, 6.52, 6.56, 6.6,
      6.64, 6.68, 6.72, 6.76, 6.8, 6.84,
      6.88, 6.92, 6.96, 7.0, 7.04, 7.08,
      7.12, 7.16, 7.2, 7.24, 7.28, 7.32,
      7.36, 7.4, 7.44, 7.48, 7.52, 7.56,
      7.6, 7.64, 7.68, 7.72, 7.76, 7.8,
      7.84, 7.88, 7.92, 7.96, 8.0, 8.04,
      8.08, 8.12, 8.16, 8.2, 8.24, 8.28,
      8.32, 8.36, 8.4, 8.44, 8.48, 8.52,
      8.56, 8.6, 8.64, 8.68, 8.72, 8.76,
      8.8, 8.84, 8.88, 8.92, 8.96, 9.0,
      9.04, 9.08, 9.12, 9.16, 9.2, 9.24,
      9.28, 9.32, 9.36, 9.4, 9.44, 9.48,
      9.52, 9.56, 9.6, 9.64, 9.68, 9.72,
      9.76, 9.8, 9.84, 9.88, 9.92, 9.96,
      10.0, 10.05, 10.1, 10.15, 10.2, 10.25,
      10.3, 10.35, 10.4, 10.45, 10.5, 10.55,
      10.6, 10.65, 10.7, 10.75, 10.8, 10.85,
      10.9, 10.95, 11.0, 11.05, 11.1, 11.15,
      11.2, 11.25, 11.3, 11.35, 11.4, 11.45,
      11.5, 11.55, 11.6, 11.65, 11.7, 11.75,
      11.8, 11.85, 11.9, 11.95, 12.0, 12.05,
      12.1, 12.15, 12.2, 12.25, 12.3, 12.35,
      12.4, 12.45, 12.5, 12.55, 12.6, 12.65,
      12.7, 12.75, 12.8, 12.85, 12.9, 12.95,
      13.0, 13.05, 13.1, 13.15, 13.2, 13.25,
      13.3, 13.35, 13.4, 13.45, 13.5, 13.55,
      13.6, 13.65, 13.7, 13.75, 13.8, 13.85,
      13.9, 13.95, 14.0, 14.05, 14.1, 14.15,
      14.2, 14.25, 14.3, 14.35, 14.4, 14.45,
      14.5, 14.55, 14.6, 14.65, 14.7, 14.75,
      14.8, 14.85, 14.9, 14.95, 15.0, 15.1,
      15.2, 15.3, 15.4, 15.5, 15.6, 15.7,
      15.8, 15.9, 16.0, 16.1, 16.2, 16.3,
      16.4, 16.5, 16.6, 16.7, 16.8, 16.9,
      17.0, 17.1, 17.2, 17.3, 17.4, 17.5,
      17.6, 17.7, 17.8, 17.9, 18.0, 18.1,
      18.2, 18.3, 18.4, 18.5, 18.6, 18.7,
      18.8, 18.9, 19.0, 19.1, 19.2, 19.3,
      19.4, 19.5, 19.6, 19.7, 19.8, 19.9,
      20.0, 20.1, 20.2, 20.3, 20.4, 20.5,
      20.6, 20.7, 20.8, 20.9, 21.0, 21.1,
      21.2, 21.3, 21.4, 21.5, 21.6, 21.7,
      21.8, 21.9, 22.0, 22.1, 22.2, 22.3,
      22.4, 22.5, 22.6, 22.7, 22.8, 22.9,
      23.0, 23.1, 23.2, 23.3, 23.4, 23.5,
      23.6, 23.7, 23.8, 23.9, 24.0, 24.1,
      24.2, 24.3, 24.4, 24.5, 24.6, 24.7,
      24.8, 24.9, 25.0, 25.1, 25.2, 25.3,
      25.4, 25.5, 25.6, 25.7, 25.8, 25.9,
      26.0, 26.1, 26.2, 26.3, 26.4, 26.5,
      26.6, 26.7, 26.8, 26.9, 27.0, 27.1,
      27.2, 27.3, 27.4, 27.5, 27.6, 27.7,
      27.8, 27.9, 28.0, 28.1, 28.2, 28.3,
      28.4, 28.5, 28.6, 28.7, 28.8, 28.9,
      29.0, 29.1, 29.2, 29.3, 29.4, 29.5,
      29.6, 29.7, 29.8, 29.9, 30.0, 30.1,
      30.2, 30.3, 30.4, 30.5, 30.6, 30.7,
      30.8, 30.9, 31.0, 31.1, 31.2, 31.3,
      31.4, 31.5, 31.6, 31.7, 31.8, 31.9,
      32.0, 32.1, 32.2, 32.3, 32.4, 32.5,
      32.6, 32.7, 32.8, 32.9, 33.0, 33.1,
      33.2, 33.3, 33.4, 33.5, 33.6, 33.7,
      33.8, 33.9, 34.0, 34.1, 34.2, 34.3,
      34.4, 34.5, 34.6, 34.7, 34.8, 34.9,
      35.0, 35.1, 35.2, 35.3, 35.4, 35.5,
      35.6, 35.7, 35.8, 35.9, 36.0, 36.1,
      36.2, 36.3, 36.4, 36.5, 36.6, 36.7,
      36.8, 36.9, 37.0, 37.1, 37.2, 37.3,
      37.4, 37.5, 37.6, 37.7, 37.8, 37.9,
      38.0, 38.1, 38.2, 38.3, 38.4, 38.5,
      38.6, 38.7, 38.8, 38.9, 39.0, 39.1,
      39.2, 39.3, 39.4, 39.5, 39.6, 39.7,
      39.8, 39.9, 40.0, 40.1, 40.2, 40.3,
      40.4, 40.5, 40.6, 40.7, 40.8, 40.9,
      41.0, 41.1, 41.2, 41.3, 41.4, 41.5,
      41.6, 41.7, 41.8, 41.9, 42.0, 42.1,
      42.2, 42.3, 42.4, 42.5, 42.6, 42.7,
      42.8, 42.9, 43.0, 43.1, 43.2, 43.3,
      43.4, 43.5, 43.6, 43.7, 43.8, 43.9,
      44.0, 44.1, 44.2, 44.3, 44.4, 44.5,
      44.6, 44.7, 44.8, 44.9, 45.0, 45.1,
      45.2, 45.3, 45.4, 45.5, 45.6, 45.7,
      45.8, 45.9, 46.0, 46.1, 46.2, 46.3,
      46.4, 46.5, 46.6, 46.7, 46.8, 46.9,
      47.0, 47.1, 47.2, 47.3, 47.4, 47.5,
      47.6, 47.7, 47.8, 47.9, 48.0, 48.1,
      48.2, 48.3, 48.4, 48.5, 48.6, 48.7,
      48.8, 48.9, 49.0, 49.1, 49.2, 49.3,
      49.4, 49.5, 49.6, 49.7, 49.8, 49.9,
      50.0, 50.1, 50.2, 50.3, 50.4, 50.5,
      50.6, 50.7, 50.8, 50.9, 51.0, 51.1,
      51.2, 51.3, 51.4, 51.5, 51.6, 51.7,
      51.8, 51.9, 52.0, 52.1, 52.2, 52.3,
      52.4, 52.5, 52.6, 52.7, 52.8, 52.9,
      53.0, 53.1, 53.2, 53.3, 53.4, 53.5,
      53.6, 53.7, 53.8, 53.9, 54.0, 54.1,
      54.2, 54.3, 54.4, 54.5, 54.6, 54.7,
      54.8, 54.9, 55.0, 55.1, 55.2, 55.3,
      55.4, 55.5, 55.6, 55.7, 55.8, 55.9,
      56.0, 56.1, 56.2, 56.3, 56.4, 56.5,
      56.6, 56.7, 56.8, 56.9, 57.0, 57.1,
      57.2, 57.3, 57.4, 57.5, 57.6, 57.7,
      57.8, 57.9, 58.0, 58.1, 58.2, 58.3,
      58.4, 58.5, 58.6, 58.7, 58.8, 58.9,
      59.0, 59.1, 59.2, 59.3, 59.4, 59.5,
      59.6, 59.7, 59.8, 59.9, 60.0, 60.1,
      60.2, 60.3, 60.4, 60.5, 60.6, 60.7,
      60.8, 60.9, 61.0, 61.1, 61.2, 61.3,
      61.4, 61.5, 61.6, 61.7, 61.8, 61.9,
      62.0, 62.1, 62.2, 62.3, 62.4, 62.5,
      62.6, 62.7, 62.8, 62.9, 63.0, 63.1,
      63.2, 63.3, 63.4, 63.5, 63.6, 63.7,
      63.8, 63.9, 64.0, 64.1, 64.2, 64.3,
      64.4, 64.5, 64.6, 64.7, 64.8, 64.9,
      65.0, 65.1, 65.2, 65.3, 65.4, 65.5,
      65.6, 65.7, 65.8, 65.9, 66.0, 66.1,
      66.2, 66.3, 66.4, 66.5, 66.6, 66.7,
      66.8, 66.9, 67.0, 67.1, 67.2, 67.3,
      67.4, 67.5, 67.6, 67.7, 67.8, 67.9,
      68.0, 68.1, 68.2, 68.3, 68.4, 68.5,
      68.6, 68.7, 68.8, 68.9, 69.0, 69.1,
      69.2, 69.3, 69.4, 69.5, 69.6, 69.7,
      69.8, 69.9, 70.0, 70.1, 70.2, 70.3,
      70.4, 70.5, 70.6, 70.7, 70.8, 70.9,
      71.0, 71.1, 71.2, 71.3, 71.4, 71.5,
      71.6, 71.7, 71.8, 71.9, 72.0, 72.1,
      72.2, 72.3, 72.4, 72.5, 72.6, 72.7,
      72.8, 72.9, 73.0, 73.1, 73.2, 73.3,
      73.4, 73.5, 73.6, 73.7, 73.8, 73.9,
      74.0, 74.1, 74.2, 74.3, 74.4, 74.5,
      74.6, 74.7, 74.8, 74.9, 75.0, 75.1,
      75.2, 75.3, 75.4, 75.5, 75.6, 75.7,
      75.8, 75.9, 76.0, 76.1, 76.2, 76.3,
      76.4, 76.5, 76.6, 76.7, 76.8, 76.9,
      77.0, 77.1, 77.2, 77.3, 77.4, 77.5,
      77.6, 77.7, 77.8, 77.9, 78.0, 78.1,
      78.2, 78.3, 78.4, 78.5, 78.6, 78.7,
      78.8, 78.9, 79.0, 79.1, 79.2, 79.3,
      79.4, 79.5, 79.6, 79.7, 79.8, 79.9,
      80.0, 80.1, 80.2, 80.3, 80.4, 80.5,
      80.6, 80.7, 80.8, 80.9, 81.0, 81.1,
      81.2, 81.3, 81.4, 81.5, 81.6, 81.7,
      81.8, 81.9, 82.0, 82.1, 82.2, 82.3,
      82.4, 82.5, 82.6, 82.7, 82.8, 82.9,
      83.0, 83.1, 83.2, 83.3, 83.4, 83.5,
      83.6, 83.7, 83.8, 83.9, 84.0, 84.1,
      84.2, 84.3, 84.4, 84.5, 84.6, 84.7,
      84.8, 84.9, 85.0, 85.1, 85.2, 85.3,
      85.4, 85.5, 85.6, 85.7, 85.8, 85.9,
      86.0, 86.1, 86.2, 86.3, 86.4, 86.5,
      86.6, 86.7, 86.8, 86.9, 87.0, 87.1,
      87.2, 87.3, 87.4, 87.5, 87.6, 87.7,
      87.8, 87.9, 88.0, 88.1, 88.2, 88.3,
      88.4, 88.5, 88.6, 88.7, 88.8, 88.9,
      89.0, 89.1, 89.2, 89.3, 89.4, 89.5,
      89.6, 89.7, 89.8, 89.9, 90.0, 90.25,
      90.5, 90.75, 91.0, 91.25, 91.5, 91.75,
      92.0, 92.25, 92.5, 92.75, 93.0, 93.25,
      93.5, 93.75, 94.0, 94.25, 94.5, 94.75,
      95.0, 95.25, 95.5, 95.75, 96.0, 96.25,
      96.5, 96.75, 97.0, 97.25, 97.5, 97.75,
      98.0, 98.25, 98.5, 98.75, 99.0, 99.25,
      99.5, 99.75, 100.0, 100.25, 100.5, 100.75,
      101.0, 101.25, 101.5, 101.75, 102.0, 102.25,
      102.5, 102.75, 103.0, 103.25, 103.5, 103.75,
      104.0, 104.25, 104.5, 104.75, 105.0, 105.25,
      105.5, 105.75, 106.0, 106.25, 106.5, 106.75,
      107.0, 107.25, 107.5, 107.75, 108.0, 108.25,
      108.5, 108.75, 109.0, 109.25, 109.5, 109.75,
      110.0, 110.25, 110.5, 110.75, 111.0, 111.25,
      111.5, 111.75, 112.0, 112.25, 112.5, 112.75,
      113.0, 113.25, 113.5, 113.75, 114.0, 114.25,
      114.5, 114.75, 115.0, 115.25, 115.5, 115.75,
      116.0, 116.25, 116.5, 116.75, 117.0, 117.25,
      117.5, 117.75, 118.0, 118.25, 118.5, 118.75,
      119.0, 119.25, 119.5, 119.75, 120.0, 120.25,
      120.5, 120.75, 121.0, 121.25, 121.5, 121.75,
      122.0, 122.25, 122.5, 122.75, 123.0, 123.25,
      123.5, 123.75, 124.0, 124.25, 124.5, 124.75,
      125.0, 125.25, 125.5, 125.75, 126.0, 126.25,
      126.5, 126.75, 127.0, 127.25, 127.5, 127.75,
      128.0, 128.25, 128.5, 128.75, 129.0, 129.25,
      129.5, 129.75, 130.0, 130.25, 130.5, 130.75,
      131.0, 131.25, 131.5, 131.75, 132.0, 132.25,
      132.5, 132.75, 133.0, 133.25, 133.5, 133.75,
      134.0, 134.25, 134.5, 134.75, 135.0, 135.25,
      135.5, 135.75, 136.0, 136.25, 136.5, 136.75,
      137.0, 137.25, 137.5, 137.75, 138.0, 138.25,
      138.5, 138.75, 139.0, 139.25, 139.5, 139.75,
      140.0, 140.25, 140.5, 140.75, 141.0, 141.25,
      141.5, 141.75, 142.0, 142.25, 142.5, 142.75,
      143.0, 143.25, 143.5, 143.75, 144.0, 144.25,
      144.5, 144.75, 145.0, 145.25, 145.5, 145.75,
      146.0, 146.25, 146.5, 146.75, 147.0, 147.25,
      147.5, 147.75, 148.0, 148.25, 148.5, 148.75,
      149.0, 149.25, 149.5, 149.75, 150.0, 150.25,
      150.5, 150.75, 151.0, 151.25, 151.5, 151.75,
      152.0, 152.25, 152.5, 152.75, 153.0, 153.25,
      153.5, 153.75, 154.0, 154.25, 154.5, 154.75,
      155.0, 155.25, 155.5, 155.75, 156.0, 156.25,
      156.5, 156.75, 157.0, 157.25, 157.5, 157.75,
      158.0, 158.25, 158.5, 158.75, 159.0, 159.25,
      159.5, 159.75, 160.0, 160.25, 160.5, 160.75,
      161.0, 161.25, 161.5, 161.75, 162.0, 162.25,
      162.5, 162.75, 163.0, 163.25, 163.5, 163.75,
      164.0, 164.25, 164.5, 164.75, 165.0, 165.25,
      165.5, 165.75, 166.0, 166.25, 166.5, 166.75,
      167.0, 167.25, 167.5, 167.75, 168.0, 168.25,
      168.5, 168.75, 169.0, 169.25, 169.5, 169.75,
      170.0, 170.25, 170.5, 170.75, 171.0, 171.25,
      171.5, 171.75, 172.0, 172.25, 172.5, 172.75,
      173.0, 173.25, 173.5, 173.75, 174.0, 174.25,
      174.5, 174.75, 175.0, 175.25, 175.5, 175.75,
      176.0, 176.25, 176.5, 176.75, 177.0, 177.25,
      177.5, 177.75, 178.0, 178.25, 178.5, 178.75,
      179.0, 179.25, 179.5, 179.75, 180.0]
a = np.array(tht)
x = np.cos(a*np.pi/180)
# rs = 5.5 #粒子最小半径
# re = 11.5 #粒子最大半径
nbt = 11.2 # 波长
# mesurement = 4  #尺度参数

f =open(r'D:\work\code\Mie\MIE_lh\6.2um\log6.2.txt',encoding='utf-8')

sentimentlist = []
for line in f:
    s = line.strip().split('\t')
    sentimentlist.append(s)
df_train=pd.DataFrame(sentimentlist,columns=['data'])
q_data  = np.zeros(int((len(np.array(df_train)))/266),dtype=np.float64)
sca_data = np.zeros(int((len(np.array(df_train)))/266),dtype=np.float64)
abs  = np.zeros(int((len(np.array(df_train)))/266),dtype=np.float64)
asf  = np.zeros(int((len(np.array(df_train)))/266),dtype=np.float64)
rdata = np.zeros(int((len(np.array(df_train)))/266),dtype=np.float64)
ctest = np.zeros([int((len(np.array(df_train)))/266),258], dtype=np.float64)
for j in range(int((len(np.array(df_train)))/266)):
    rdata[j] = np.array(df_train)[266*j+1]
    q_data[j] = np.array(df_train)[266*j+2]
    sca_data[j] = np.array(df_train)[266*j+3]
    abs[j] = np.array(df_train)[266*j+4]
    asf[j] = np.array(df_train)[266*j+5]
    # Deff = 3/2*rdata
    c = np.array(df_train)[266*j+8:266*j+266]

    for k in range(258):
          ctem = c[k]
          ctest[j,k] = np.float64(ctem[0]) * (2 * k + 1)
    print('第%i个半径的粒子'%j)

measurement = (2*np.pi*rdata)/11.2
w= sca_data/q_data
# 粒子半径
# rdata = np.arange(0.01,6.01,0.01)
################### 计算粒子的gamma分布#########################
af = 1/veff-3
b = (af+3)/reff
# nr = np.ones([len(reff),len(rdata)],dtype=np.float64)
nr = np.zeros([len(reff),len(rdata)],dtype=np.float64)
# for m in range(len(reff)):
#       nr[m,:] = (N*b[m]**(af+1)*(rdata**af)*np.exp(-b[m]*rdata))/gamma(af+1)
for m in range(len(rdata)):
      nr[:,m] = (N*b**(af+1))*(rdata[m]**af)*(np.exp(-b*rdata[m]))/gamma(af+1)

g2 = np.zeros([len(reff)],dtype=np.float64)
for n in range(len(reff)): # 不同的Reff
    g2[n]= gg(q_data,w,asf,nr[n,:],rdata)
############ r 的gamma分布#########################################
plt.plot(rdata,nr[10,:],color = 'red',linewidth = 1.0,linestyle = '-')
plt.plot(rdata,nr[688,:],color = 'yellow',linewidth = 2.0,linestyle = '--')
plt.legend(['reff=%f'%reff[10],'reff=%f'%reff[688]])
my_x_ticks = np.arange(0,rdata[-1],10)
plt.xticks(my_x_ticks)
plt.show()
plt.close()
################# 计算Reff###############################################
deff = np.zeros(len(reff),dtype=np.float64)
for n in range(len(reff)):
      deff[n] = tesdf(rdata,nr[n,:])
# ############粒子群勒让德多项式系数#########################################################################
# # 粒子群相函数
# 计算勒让德多项式系数
cnn = np.zeros([len(reff),2],dtype=np.float64)
for p in range(2): # 不同阶数
      for q in range(len(reff)): # 不同的Reff
            cnn[q,p]= refff(rdata,q_data,w,nr[q,:],ctest[:,p])
            print(p,q)

g = cnn/3.0

my_x_ticks = np.arange(0,60,10)
plt.xticks(my_x_ticks)

fig,ax = plt.subplots()
# ax.set_yscale("log")
# ax.set_ylim(1e-3, 1e4)

# 相函数对比
# defflocal = np.abs(np.subtract.outer(deff,60)).argmin(0)
# minlocal = np.abs(np.subtract.outer(deff,2)).argmin(0)
# ax.plot(deff[:defflocal],g[:defflocal,1],linewidth= 2.0,color='blue',linestyle = '-')
# ax.plot(deff[:defflocal],g2[:defflocal],linewidth= 1.5,color='green',linestyle = '--')

ax.plot(deff,g[:,1],linewidth= 2.0,color='blue',linestyle = '-')
ax.plot(deff,g2,linewidth= 1.5,color='green',linestyle = '--')
my_y_ticks = np.arange(0.2,1.05,0.05)
plt.yticks(my_y_ticks)
ax.set_adjustable("box")
plt.legend(['legendre C1','asymmetry factor'])
plt.xlabel("effective radius")
plt.ylabel("value")
#  折射指数

# plt.xlim(0,None)
# plt.text(0.5,0.01,r'$\alpha=%.3f$'% measurement,
#          fontdict={'size':12,'color':'r'})
# plt.savefig(savepng)
plt.show()
##粒子群相函数计算###################################################################
# # 计算勒让德展开式
p1 = x
# p2 = 1/2*(3*x**2-1)
# p3 = 1/2*(5*x**3-3*x)
# # p4 = 1/8*(35*x**4-30*x**2+3)
# # p5 = 1/8*(63*x**5-70*x**3+15*x)
# # p6 = 1/16*(231*x**6-315*x**4+105*x**2-5)
# # p7 = (13*x*p6-6*p5)/7
# # p8 = (15*x*p7-7*p6)/8
# # p9 = (17*x*p8-8*p7)/9
# # p10 = (19*x*p9-9*p8)/10
# # p11= (21*x*p10-10*p9)/11
# # p12 = (23*x*p11-11*p10)/12
#
# pxh = np.zeros([258,len(x)],dtype=np.float64)
# pxh[0,:] = 1
# pxh[1,:] = p1
# pxh[2,:] = p2
# pxh[3,:] = p3
# for n in range(3,258,1):
#     pxh[n,:] = ((2*n-1)*x*pxh[n-1,:]-(n-1)*pxh[n-2,:])/n
#
# ##############################################################################
# # 计算P11(多阶)
# p11s = np.zeros(len(x),dtype = np.float64)
# p11s = cnn[1]
# for k in range(2,258,1):
#     p11s = p11s+cnn[k]*pxh[k,:]
#
# # 粒子群相函数画图
# my_x_ticks = np.arange(0,190,10)
# plt.xticks(my_x_ticks)
#
# fig,ax = plt.subplots()
# ax.set_yscale("log")
# ax.set_adjustable("datalim")
# # 相函数对比
# ax.plot(tht,p11s,linewidth= 1.0,color='blue',linestyle = '-')
# ax.set_ylim(1e-3, 1e4)
# plt.legend(['OURS'])
# plt.xlabel("scattering angle")
# plt.ylabel("phase function")
# #  折射指数
#
# # plt.xlim(0,None)
# # plt.text(0.5,0.01,r'$\alpha=%.3f$'% measurement,
# #          fontdict={'size':12,'color':'r'})
# # plt.savefig(savepng)
# plt.show()

#####################################################################
