"""
ice cloud --optical property of particle
single point
author:ida
date:2021-11-03
"""

import scipy.integrate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import gamma

import math

def cn(x,p,n,fx):
    """
    x:散射角
    p:勒让德多项式
    fx:相函数
    """
    c = 0
    for i in range(len(x)-1):
        c = c + (fx[i]*p[i]+fx[i+1]*p[i+1])*(x[i]-x[i+1])*1/2
        # print(i)
    cn = ((2 * n + 1) / 2)*c
    return cn

veff = 0.172 #方差
reff = 5.89 #有效粒子半径
N = 5
# tht = np.arange(0,190,0.01) #角度
# x = np.cos(tht*np.pi/180)

datapath = 'D:\work\code\CLOUD\MTCpack2018\MTCpack2018'
f =open(datapath+'\\MTC_vo_00210_g1_000282',encoding='utf-8')

sentimentlist = []
for line in f:
    s = line.strip().split('\t')
    sentimentlist.append(s)
df_train=pd.DataFrame(sentimentlist,columns=['data'])

rdata = float(np.array(df_train)[6][0].split(' ')[0])*10**6
q_data = float(np.array(df_train)[2][0].split(' ')[0])
wavelength = float(np.array(df_train)[1][0].split(' ')[0])*10**6
sca_data = float(np.array(df_train)[3][0].split(' ')[0])
asf = float(np.array(df_train)[8][0].split(' ')[0])
area = float(np.array(df_train)[4][0].split(' ')[0])*10**12
ww = float(np.array(df_train)[7][0].split(' ')[0])
tht = np.zeros(len(df_train)-9,dtype=np.float64)
cfx1 =np.zeros(len(df_train)-9,dtype=np.float64)
cfx2 = np.zeros(len(df_train)-9,dtype=np.float64)
cfx3 = np.zeros(len(df_train)-9,dtype=np.float64)
cfx4 = np.zeros(len(df_train)-9,dtype=np.float64)
cfx5 = np.zeros(len(df_train)-9,dtype=np.float64)
cfx6 = np.zeros(len(df_train)-9,dtype=np.float64)
for j in range(len(df_train)-9):
    b = [' ']
    a = np.array(df_train)[9+j][0].split(' ')
    a = [i for i in a if i != ' ']
    aa = [i for i in a if i != '']
    tht[j] = float(aa[0])
    cfx1[j] =float(aa[1])
    cfx2[j] =float(aa[2])
    cfx3[j] =float(aa[3])
    cfx4[j] =float(aa[4])
    cfx5[j] =float(aa[5])
    cfx6[j] =float(aa[6])
    print(j)

measurement = (2*np.pi*rdata)/wavelength
x = np.cos(tht*np.pi/180)
# 粒子半径
# rdata = np.arange(0.01,6.01,0.01)
# 计算粒子的gamma分布
# af = 1/veff-3
# b = (af+3)/reff
# nr = np.zeros(len(rdata),dtype=np.float64)
# nr = (N*b**(af+1)*(rdata**af)*np.exp(-b*rdata))/gamma(af+1)
# # r 的gamma分布
# plt.plot(rdata,nr,color = 'red',linewidth = 1.0,linestyle = '-')
# my_x_ticks = np.arange(0.01,6.01,0.1)
# plt.xticks(my_x_ticks)
# plt.show()
# plt.close()
#########
# 粒子群相函数
# c0reff =  refff(rdata,q_data,w,nr,c0)
# c1reff = refff(rdata,q_data,w,nr,c1)
# c2reff = refff(rdata,q_data,w,nr,c2)
# c3reff = refff(rdata,q_data,w,nr,c3)
# c4reff = refff(rdata,q_data,w,nr,c4)
# c5reff = refff(rdata,q_data,w,nr,c5)
# c6reff = refff(rdata,q_data,w,nr,c6)
# c7reff = refff(rdata,q_data,w,nr,c7)
# c8reff = refff(rdata,q_data,w,nr,c8)

p1 = np.zeros(len(tht),dtype = np.float64)
p2 = np.zeros(len(tht),dtype = np.float64)
p3 = np.zeros(len(tht),dtype = np.float64)
p4 = np.zeros(len(tht),dtype = np.float64)
p5 = np.zeros(len(tht),dtype = np.float64)
p6 = np.zeros(len(tht),dtype = np.float64)
p7 = np.zeros(len(tht),dtype = np.float64)
p8 = np.zeros(len(tht),dtype = np.float64)

p1 = x
p2 = 1/2*(3*x**2-1)
p3 = 1/2*(5*x**3-3*x)
p4 = 1/8*(35*x**4-30*x**2+3)
p5 = 1/8*(63*x**5-70*x**3+15*x)
p6 = 1/16*(231*x**6-315*x**4+105*x**2-5)
# p7 = (13*x*p6-6*p5)/7
# p8 = (15*x*p7-7*p6)/8
# p9 = (17*x*p8-8*p7)/9
# p10 = (19*x*p9-9*p8)/10
# p11= (21*x*p10-10*p9)/11
# p12 = (23*x*p11-11*p10)/12
pxh = np.zeros([257,len(x)],dtype=np.float64)
pxh[0,:] = 1
pxh[1,:] = p1
pxh[2,:] = p2
pxh[3,:] = p3
pxh[4,:] = p4
pxh[5,:] = p5
pxh[6,:] = p6
for n in range(7,257,1):
    pxh[n,:] = ((2*n-1)*x*pxh[n-1,:]-(n-1)*pxh[n-2,:])/n

###### 单个粒子Cn
cnn = np.zeros(257,dtype=np.float64)
for p in range(257):
      cnn[p]= cn(x,pxh[p,:],p,cfx1)


# 计算P11(多阶)
p11s = np.zeros(len(x),dtype = np.float64)
p11s = cnn[0]
for k in range(1,257,1):
    p11s = p11s+cnn[k]*pxh[k,:]
# 单个粒子相函数画图
my_x_ticks = np.arange(0,190,10)
plt.xticks(my_x_ticks)

fig,ax = plt.subplots()
ax.set_yscale("log")
ax.set_adjustable("datalim")
# 相函数对比
ax.plot(tht,p11s,linewidth= 1.0,color='blue',linestyle = '-')
ax.set_ylim(1e-3, 1e4)
plt.legend(['OURS'])
plt.xlabel("scattering angle")
plt.ylabel("phase function")
#  折射指数

plt.xlim(0,None)
plt.text(0.5,0.01,r'$\alpha=%.3f$'% measurement,
         fontdict={'size':12,'color':'r'})
# plt.savefig(savepng)
plt.show()

# 计算积分验证结果
# fx = lambda xx:(c0test+c1test*np.cos(xx)+c2test*(1/2*(3*(np.cos(xx)**2)-1))+c3test*\
#                 (1/2*(5*(np.cos(xx)**3)-3*np.cos(xx)))+c4test*(1/8*(35*(np.cos(xx)**4)-30*(np.cos(xx)**2)+3)))*np.sin(xx)/(4*np.pi)
# y=scipy.integrate.quad(fx,0,np.pi)
#
# fx2 = lambda xxx:y[0]
# print('单个粒子结果验证')
# print(y[0])
# print(scipy.integrate.quad(fx2,0,2*np.pi)[0])
# # 二重积分结果
# fy = lambda xx,yy:(c0test+c1test*np.cos(xx)+c2test*(1/2*(3*(np.cos(xx)**2)-1))+c3test*\
#                 (1/2*(5*(np.cos(xx)**3)-3*np.cos(xx)))+c4test*(1/8*(35*(np.cos(xx)**4)-30*(np.cos(xx)**2)+3)))*np.sin(xx)/(4*np.pi)
# p,err = scipy.integrate.dblquad(fy,0,2*np.pi,lambda g:0,lambda h:np.pi)
# print(p)

# print(p11)