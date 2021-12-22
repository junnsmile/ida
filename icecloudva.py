

"""
ice cloud --optical property of particle
author:ida
date:2021-12-03
"""

import scipy.integrate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import gamma
import glob
import os

def gg(area,q_data,ww,asf,nr,rdata):
    fz = np.float64(0)
    fm = 0.0000000
    for i in range(len(rdata) - 1):
        fz = fz + ((area[i+1] * q_data[i + 1] * ww[i + 1] * asf[i + 1] * nr[i + 1]) + (area[i] * q_data[i] * ww[i] * asf[i] * nr[i])) * (rdata[i + 1] - rdata[i]) * 1 / 2
        fm = fm + ((area[i+1]*q_data[i+1]*ww[i+1]*nr[i+1]) +(area[i]*q_data[i]*ww[i]*nr[i]))*(rdata[i+1]-rdata[i])*1/2
    gg = fz / fm
    return gg

def tesdfx(rdata, nr):
    '''
      计算Reff
      验证粒子群C2和G的结果时，作为横轴
    '''
    ddata = rdata
    dfz = 0
    dfm = 0
    for i in range(len(rdata) - 1):
        dfz = dfz + ((ddata[i + 1] ** 3 * nr[i + 1]) + (ddata[i] ** 3 * nr[i])) * (ddata[i + 1] - ddata[i]) * 1 / 2
        dfm = dfm + ((ddata[i + 1] ** 2 * nr[i + 1]) + (ddata[i] ** 2 * nr[i])) * (ddata[i + 1] - ddata[i]) * 1 / 2

    tem = dfz / dfm
    return tem


def refff(rdata,q_data,w,nr,c,area):
    """
    rdata:粒子直径
    q_data:消光系数
    w:单次散射反照率
    nr
    c:勒让德展开系数
    返回： 粒子群勒让德系数
    """
    fz = 0
    fm = 0
    for i in range(len(rdata) - 1):
        fz = fz + ((area[i+1] * q_data[i + 1] * w[i + 1] * c[i + 1] * nr[i + 1]) + (area[i] * q_data[i] * w[i] * c[i] * nr[i])) * (rdata[i + 1] - rdata[i]) * 1 / 2
        fm = fm + ((area[i+1]*q_data[i+1]*w[i+1]*nr[i+1]) +(area[i]*q_data[i]*w[i]*nr[i]))*(rdata[i+1]-rdata[i])*1/2
    creff = fz / fm
    return creff

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

def tesdf(rdata,vol,area,nr):
    '''
      计算Reff
      验证粒子群C2和G的结果时，作为横轴
    '''
    dfz = 0
    dfm = 0
    ddata = rdata
    for i in range(len(ddata) - 1):
        dfz = dfz+(vol[i+1]*nr[i + 1] + vol[i]*nr[i]) * (ddata[i + 1] - ddata[i]) * 1 / 2
        dfm = dfm+(area[i + 1]*nr[i + 1] + area[i]*nr[i]) * (ddata[i + 1] - ddata[i]) * 1 / 2

    tem = 3*dfz / (2*dfm)
    return tem


veff = 1/3 #方差
reff1 = np.arange(0.1,10.580, 0.01)  #有效粒子直径
reff2 = np.arange(10.580, 600, 0.01)
reff = np.hstack((reff1,reff2))
N = 5
# tht = np.arange(0,190,0.01) #角度
# x = np.cos(tht*np.pi/180)

datapath = 'D:\work\code\CLOUD\MTCpack2018\MTCpack2018'
os.chdir(datapath)
filesfd = glob.glob('MTC_vo_06200_fd_*')
filesg1 = glob.glob('MTC_vo_06200_g1_*')
filesg2 = glob.glob('MTC_vo_06200_g2_*')
files = filesg1 +filesfd+filesg2
files = sorted(files)
k = 0
rdata0 = np.zeros(len(files),dtype=np.float64)
q_data0 = np.zeros(len(files),dtype=np.float64)
wavelength0 = np.zeros(len(files),dtype=np.float64)
sca_data0 = np.zeros(len(files),dtype=np.float64)
asf0 = np.zeros(len(files),dtype=np.float64)
area0 = np.zeros(len(files),dtype=np.float64)
ww0 = np.zeros(len(files),dtype=np.float64)
vol0 = np.zeros(len(files),dtype=np.float64)
tht0 = np.zeros([len(files),1801], dtype=np.float64)
cfx10 = np.zeros([len(files),1801], dtype=np.float64)
cfx20 = np.zeros([len(files),1801], dtype=np.float64)
cfx30 = np.zeros([len(files),1801], dtype=np.float64)
cfx40 = np.zeros([len(files),1801], dtype=np.float64)
cfx50 = np.zeros([len(files),1801], dtype=np.float64)
cfx60 = np.zeros([len(files),1801], dtype=np.float64)
for file in files:
    f =open(datapath+'\\'+file,encoding='utf-8')
    sentimentlist = []
    for line in f:
        s = line.strip().split('\t')
        sentimentlist.append(s)
    df_train=pd.DataFrame(sentimentlist,columns=['data'])

    rdata0[k] = float(np.array(df_train)[6][0].split(' ')[0])*10**6
    q_data0[k] = float(np.array(df_train)[2][0].split(' ')[0])
    wavelength0[k]= float(np.array(df_train)[1][0].split(' ')[0])*10**6
    sca_data0[k] = float(np.array(df_train)[3][0].split(' ')[0])
    asf0[k] = float(np.array(df_train)[8][0].split(' ')[0])
    area0[k] = float(np.array(df_train)[4][0].split(' ')[0])*10**12
    ww0[k] = float(np.array(df_train)[7][0].split(' ')[0])
    vol0[k] = float(np.array(df_train)[5][0].split(' ')[0])*10**18
    for j in range(len(df_train)-9):
        b = [' ']
        a = np.array(df_train)[9+j][0].split(' ')
        a = [i for i in a if i != ' ']
        aa = [i for i in a if i != '']
        tht0[k,j] = float(aa[0])
        cfx10[k,j] =float(aa[1])
        cfx20[k,j] =float(aa[2])
        cfx30[k,j] =float(aa[3])
        cfx40[k,j] =float(aa[4])
        cfx50[k,j] =float(aa[5])
        cfx60[k,j] =float(aa[6])
        print(j)
    k = k +1

rdata=rdata0[np.argsort(rdata0)]
q_data = q_data0[np.argsort(rdata0)]
wavelength = wavelength0[np.argsort(rdata0)]
sca_data = sca_data0[np.argsort(rdata0)]
asf = asf0[np.argsort(rdata0)]
area = area0[np.argsort(rdata0)]
ww = ww0[np.argsort(rdata0)]
vol = vol0[np.argsort(rdata0)]
tht =tht0[np.argsort(rdata0)]
cfx1 = cfx10[np.argsort(rdata0),:]
cfx2 = cfx20[np.argsort(rdata0),:]
cfx3 = cfx30[np.argsort(rdata0),:]
cfx4 = cfx40[np.argsort(rdata0),:]
cfx5 = cfx50[np.argsort(rdata0),:]
cfx6 = cfx60[np.argsort(rdata0),:]

measurement = (2*np.pi*rdata/2)/wavelength
x = np.cos(tht*np.pi/180)
# 粒子半径
# rdata = np.arange(0.01,6.01,0.01)
################### 计算粒子的gamma分布#########################
af = 1/veff-3
b = (af+3)/reff
nr = np.zeros([len(reff),len(rdata)],dtype=np.float64)
NR = np.zeros([len(reff),len(rdata)],dtype=np.float64)
# for m in range(len(reff)):
#       nr[m,:] = (N*(b[m]**(af+1))*((rdata/2)**af))*np.exp(-b[m]*(rdata/2))/gamma(af+1)
for m in range(len(rdata)):
    nr[:,m] = (N*b**(af+1))*((rdata[m]/2)**af)*(np.exp(-b*rdata[m]/2))/gamma(af+1)
    NR[:, m] = (N * b ** (af + 1)) * (rdata[m] ** af) * (np.exp(-b * rdata[m])) / gamma(af + 1)
############ r 的gamma分布#########################################
# plt.plot(rdata,nr[0,:],color = 'red',linewidth = 1.0,linestyle = '-')
# plt.plot(rdata,nr[15,:],color = 'yellow',linewidth = 1.0,linestyle = '-')
# plt.legend(['reff=%f'%reff[0],'reff=%f'%reff[15]])
# my_x_ticks = np.arange(rdata[0],rdata[-1],3)
# plt.xticks(my_x_ticks)
# plt.show()
# plt.close()

################# 计算Reff###############################################
deff = np.zeros(len(reff),dtype=np.float64)
for n in range(len(reff)):
    # deff[n] = tesdfx(rdata,nr[n,:])
    deff[n] = tesdf(rdata,vol,area,NR[n,:])


##### 勒让德展开多项式###############
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
pxh = np.zeros([257,len(files),1801],dtype=np.float64)
pxh[0,:,:] = 1
pxh[1,:,:] = p1
pxh[2,:,:] = p2
pxh[3,:,:] = p3
pxh[4,:,:] = p4
pxh[5,:,:] = p5
pxh[6,:,:] = p6
for n in range(7,257,1):
    pxh[n,:,:] = ((2*n-1)*x*pxh[n-1,:,:]-(n-1)*pxh[n-2,:,:])/n

########单个粒子Cn####################
cnn = np.zeros([len(rdata),257],dtype=np.float64)
for p in range(257):
    for q in range(len(rdata)):
        cnn[q,p]= cn(x[q,:],pxh[p,q,:],p,cfx1[q,:])

# # ############粒子群勒让德多项式系数#########################################################################
# # # 粒子群相函数# # 计算勒让德多项式系数
cnnq = np.zeros([len(reff),2],dtype=np.float64)
for p in range(2): # 不同阶数
      for q in range(len(reff)): # 不同的Reff
            cnnq[q,p]= refff(rdata,q_data,ww,NR[q,:],cnn[:,p],area)
            print(p,q)

### 粒子群勒让德多项式系数画图
# g = cnnq/3.0
g = np.zeros([len(reff)],dtype=np.float64)
for n in range(len(reff)): # 不同的Reff
    g[n]= gg(area,q_data,ww,asf,NR[n,:],rdata)

fig,ax = plt.subplots()

# 相函数对比
defflocal = np.abs(np.subtract.outer(deff,200)).argmin(0)
minlocal = np.abs(np.subtract.outer(deff,4)).argmin(0)
ax.plot(deff[minlocal:defflocal],g[minlocal:defflocal],linewidth= 2.5,color='blue',linestyle = '--')
ax.plot(deff[minlocal:defflocal],cnnq[minlocal:defflocal,1]/3,linewidth= 1.5,color='green',linestyle = '-')
my_x_ticks = np.arange(0,200,10)
plt.xticks(my_x_ticks)
my_y_ticks = np.arange(0.5,1.05,0.05)
plt.yticks(my_y_ticks)
plt.legend(['asymmetry factor','legendre C1'])
plt.xlabel("effective radius")
plt.ylabel("value")

ax.set_adjustable("datalim")
plt.show()

# 计算P11(多阶)
# p11s = np.zeros(len(x),dtype = np.float64)
p11s = cnn[0]
# for k in range(1,257,1):
#     p11s = p11s+cnn[k]*pxh[k,:]
# # 单个粒子相函数画图
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
# plt.xlim(0,None)
# plt.text(0.5,0.01,r'$\alpha=%.3f$'% measurement,
#          fontdict={'size':12,'color':'r'})
# # plt.savefig(savepng)
# plt.show()

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