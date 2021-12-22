"""
index of refraction--water cloud
未排序区间插值到排序区间间隔
排序区间插值（每个子区间个数需要输入）
author:ida
date:2021-12-15
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import polyfit,poly1d
import glob
import os
import pandas as pd

class dataku(object):
    def __init__(self,band):
        self.band = band

    def band6(self):
        # 每个子区间对应WN
        wn0 = [860,950]
        wl0 = [10.5263,11.6279]
        # 间隔数
        nv = 600000
        #间隔大小
        lv = (wl0[0]-wl0[1])/nv
        ngw = 9
        dv = 0.00015
        delta_nv = [126000,126000,126000,120000,60000,18000,18000,3000,3000]
        wn = np.arange(wn0[0],wn0[1],dv)
        wl = np.arange(wl0[1],wl0[0],lv)[::-1]
        wn1 = wn[0:delta_nv[0]]
        wn2 = wn[delta_nv[0]:delta_nv[0]+delta_nv[1]]
        wn3 = wn[delta_nv[0]+delta_nv[1]:delta_nv[0]+delta_nv[1]+delta_nv[2]]
        wn4 = wn[delta_nv[0]+delta_nv[1]+delta_nv[2]:delta_nv[0]+delta_nv[1]+delta_nv[2]+delta_nv[3]]
        wn5 = wn[delta_nv[0]+delta_nv[1]+delta_nv[2]+delta_nv[3]:delta_nv[0]+delta_nv[1]+delta_nv[2]+delta_nv[3]+delta_nv[4]]
        wn6 = wn[delta_nv[0]+delta_nv[1]+delta_nv[2]+delta_nv[3]+delta_nv[4]:delta_nv[0]+delta_nv[1]+delta_nv[2]+delta_nv[3]+delta_nv[4]+delta_nv[5]]
        wn7 = wn[delta_nv[0]+delta_nv[1]+delta_nv[2]+delta_nv[3]+delta_nv[4]+delta_nv[5]:delta_nv[0]+delta_nv[1]+delta_nv[2]+delta_nv[3]+delta_nv[4]+delta_nv[5]+delta_nv[6]]
        wn8 = wn[delta_nv[0]+delta_nv[1]+delta_nv[2]+delta_nv[3]+delta_nv[4]+delta_nv[5]+delta_nv[6]:\
                 delta_nv[0]+delta_nv[1]+delta_nv[2]+delta_nv[3]+delta_nv[4]+delta_nv[5]+delta_nv[6]+delta_nv[7]]
        wn9 = wn[delta_nv[0]+delta_nv[1]+delta_nv[2]+delta_nv[3]+delta_nv[4]+delta_nv[5]+delta_nv[6]+delta_nv[7]: \
                  delta_nv[0]+delta_nv[1]+delta_nv[2]+delta_nv[3]+delta_nv[4]+delta_nv[5]+delta_nv[6]+delta_nv[7]+delta_nv[8]]

        return wn1,wn2,wn3,wn4,wn5,wn6,wn7,wn8,wn9,wl,nv,dv,wn

    def band7(self,data):
        wn = [950,960,970,980,990,1000,1010,1020,1030,1040,1050,1060,1070,1080,1090,1100,1110,1120,1130]
        wl = [10.526,10.417,10.309,10.204,10.101,10.000,9.901,9.804,9.709,9.615,9.524,9.434,9.346,9.259,9.174,9.091,9.009,
              8.929,8.850]
        nr = [1.264,1.261 ,1.259 ,1.256 ,1.253,1.249 ,1.246 ,1.242 ,1.238,
                1.234 ,1.230,1.224,1.220 ,1.214 ,1.208,1.202,1.194 ,1.189 ,1.181]
        ni =[0.0392 ,0.0398  ,0.0405  ,0.0411 ,0.0417 ,0.0424,0.0434,0.0443 ,0.0453
            ,0.0467  , 0.0481 ,0.0497 ,0.515 ,0.0534,0.0557,0.0589 ,0.0622 ,0.0661 ,0.0707 ]

    def band9(self):
        # 每个子区间对应WN
        wn0 = [1330, 1420]
        wl0 = [7.519, 7.042]
        # 间隔数
        nv = 360000
        #间隔大小
        lv = (wl0[0]-wl0[1])/nv
        ngw = 7
        dv = 0.00025
        delta_nv = [136800,90000,72000,36000,21600,1800,1800]
        wn = np.arange(wn0[0],wn0[1],dv)
        wl = np.arange(wl0[1],wl0[0],lv)[::-1]
        wn1 = wn[0:delta_nv[0]]
        wn2 = wn[delta_nv[0]:delta_nv[0]+delta_nv[1]]
        wn3 = wn[delta_nv[0]+delta_nv[1]:delta_nv[0]+delta_nv[1]+delta_nv[2]]
        wn4 = wn[delta_nv[0]+delta_nv[1]+delta_nv[2]:delta_nv[0]+delta_nv[1]+delta_nv[2]+delta_nv[3]]
        wn5 = wn[delta_nv[0]+delta_nv[1]+delta_nv[2]+delta_nv[3]:delta_nv[0]+delta_nv[1]+delta_nv[2]+delta_nv[3]+delta_nv[4]]
        wn6 = wn[delta_nv[0]+delta_nv[1]+delta_nv[2]+delta_nv[3]+delta_nv[4]:delta_nv[0]+delta_nv[1]+delta_nv[2]+delta_nv[3]+delta_nv[4]+delta_nv[5]]
        wn7 = wn[delta_nv[0]+delta_nv[1]+delta_nv[2]+delta_nv[3]+delta_nv[4]+delta_nv[5]:delta_nv[0]+delta_nv[1]+delta_nv[2]+delta_nv[3]+delta_nv[4]+delta_nv[5]+delta_nv[6]]

        return wn1,wn2,wn3,wn4,wn5,wn6,wn7,wl,nv,dv,wn


class index_re(object):
    def __init__(self,band,nvl,nv,path,w1n,w2n,w3n,w4n,w5n,w6n,w7n,w8n,w9n):
        self.band = band
        self.nvl = nvl
        self.nv = nv
        self.path = path
        self.w1n = w1n
        self.w2n = w2n
        self.w3n = w3n
        self.w4n = w4n
        self.w5n = w5n
        self.w6n = w6n
        self.w7n = w7n
        self.w8n = w8n
        self.w9n = w9n

    def selec_idx(self,a,thresh):
        # thresh 每隔多少取一个值
        # int(np.shape(a)[0]/thresh) 该区间一共有的子区间数
        ret = []
        for i in range(int(np.shape(a)[0]/thresh)):
            ret.append(a[int(i*thresh)])
            print(i)
        ret.append(a[-1])
        return np.array(ret)

    def band9i(self):
        # 多项式结果曲线拟合  结果不好
        nr = [1.302,1.303,1.305,1.306,1.307,1.308,1.309,1.310,1.311,1.313]
        ni = [0.0342,0.0342,0.0342,0.0342,0.0343,0.0342,0.0342,0.0342,0.0343,0.0346]
        wn = [1330,1340,1350,1360,1370,1380,1390,1400,1410,1420]
        plt.plot(wn,nr,'rx',label = 'original values')
        # 建立nr 和wn 的多项式关系
        coeffr = polyfit(wn,nr,3)
        pr = poly1d(coeffr)
        print(pr)
        # 另一种拟合y值的方法
        # yvals = np.polyval(coeff,nr)
        yvalsr = pr(wn)
        plt.plot(wn,yvalsr,'b-',label = 'polyfit values')
        # 利用多项式插值
        nv = self.nv
        wnlist = np.arange(wn[0],wn[-1],dv)
        nrvalue = coeffr[0]*wnlist**3+coeffr[1]*wnlist**2+coeffr[2]*wnlist+coeffr[3]
        # plt.plot(wnlist,nrvalue,'g-',lable ='intergrate')
        plt.xlabel('wavenumber')
        plt.ylabel('nr')
        plt.legend()
        plt.title('wavenumber-nr polyfit')
        plt.show()

        # 建立ni 和wn 的多项式关系
        plt.plot(wn, ni, 'rx', label='original values')
        coeffi = polyfit(wn, ni, 5)
        pi = poly1d(coeffi)
        print(pi)
        # 另一种拟合y值的方法
        # yvals = np.polyval(coeff,nr)
        yvalsi = pi(wn)
        plt.plot(wn, yvalsi, 'b-', label='polyfit values')
        # 利用多项式插值
        nivalue = coeffi[0] * wnlist ** 5 + coeffi[1] * wnlist ** 4 + coeffi[2] * wnlist**3 + coeffi[3]*wnlist**2+coeffi[4]*wnlist+coeffi[5]
        plt.plot(wnlist,nivalue,'g-')
        plt.xlabel('wavenumber')
        plt.ylabel('ni')
        plt.legend()
        plt.title('wavenumber-ni polyfit')
        plt.show()

    def band6(self, wn1, wn2, wn3, wn4, wn5, wn6, wn7, wn8,wn9, dv):
        # 给定波数区间和该区间已有的折射指数（不是需要的wn），根据dv和nv，对每个子区间进行插值，得到每个子区间特定wn的折射指数
        nvl = self.nvl
        nv = self.nv

        nr = [1.132, 1.135, 1.139, 1.143, 1.149, 1.156, 1.162, 1.168, 1.174, 1.181]
        ni = [0.159, 0.144, 0.130, 0.118, 0.107, 0.0973, 0.0898, 0.0828, 0.0764, 0.0707]
        wn = [860, 870, 880, 890, 900, 910, 920, 930, 940, 950]
        wl = [ 10.526,10.638,10.753, 10.870,10.989,11.111,11.236,11.364,11.494,11.628]
        # 从cy划分的每个区间波数中按照间隔nvl取波数
        w1s = np.linspace(wn1[0], wn1[-1], self.w1n + 1)
        w2s = np.linspace(wn2[0], wn2[-1], self.w2n + 1)
        w3s = np.linspace(wn3[0], wn3[-1], self.w3n + 1)
        w4s = np.linspace(wn4[0], wn4[-1], self.w4n + 1)
        w5s = np.linspace(wn5[0], wn5[-1], self.w5n + 1)
        w6s = np.linspace(wn6[0], wn6[-1], self.w6n + 1)
        w7s = np.linspace(wn7[0], wn7[-1], self.w7n + 1)
        w8s = np.linspace(wn8[0], wn8[-1], self.w8n + 1)
        w9s = np.linspace(wn9[0], wn9[-1], self.w9n + 1)
        # 读取排序区间每个子区间波数最大值
        w1 = w1s[-1]
        w2 = w2s[-1]
        w3 = w3s[-1]
        w4 = w4s[-1]
        w5 = w5s[-1]
        w6 = w6s[-1]
        w7 = w7s[-1]
        w8 = w8s[-1]
        w9 = w9s[-1]

        # 看波数最大值在一直折射指数中波数所在位置
        wnloc1 = np.where(wn <= w1)[0][-1]
        wnloc2 = np.where(wn <= w2)[0][-1]
        wnloc3 = np.where(wn <= w3)[0][-1]
        wnloc4 = np.where(wn <= w4)[0][-1]
        wnloc5 = np.where(wn <= w5)[0][-1]
        wnloc6 = np.where(wn <= w6)[0][-1]
        wnloc7 = np.where(wn <= w7)[0][-1]

        # 合并要插值折射指数对应的wn
        wz1 = np.append(w1s, w2s, axis=0)
        wz2 = np.append(wz1, w3s, axis=0)
        wz3 = np.append(wz2, w4s, axis=0)
        wz4 = np.append(wz3, w5s, axis=0)
        wz5 = np.append(wz4, w6s, axis=0)
        wz6 = np.append(wz5, w7s, axis=0)
        wz7 = np.append(wz6, w8s, axis=0)
        wzong = np.append(wz7, w9s, axis=0)
        ##################################
        # 插值wns 内的折射指数nr ni
        nrr = []
        nii = []
        wll = []
        m = 0
        for i in range(len(wn) - 1):
            # 如果不是最后一截（不完整的那截）
            if i != len(wn) - 2:
                # 查找已有折射指数的波数在每个区间内的位置
                local1 = np.abs(np.subtract.outer(wzong, wn[i + 1])).argmin(0)
                wntem = wzong[m:local1]
                nrtem = np.linspace(nr[i], nr[i + 1], local1 - m + 1)
                nitem = np.linspace(ni[i], ni[i + 1], local1 - m + 1)
                wltem = np.linspace(wl[i], wl[i + 1], local1 - m + 1)

                nrr.extend(nrtem[0:- 1])
                nii.extend(nitem[0:- 1])
                wll.extend(wltem[0:- 1])
                m = local1
            else:
                local1 = np.abs(np.subtract.outer(wzong, wn[i + 1])).argmin(0)
                wntem = w1s[m:local1]
                nrtem = np.linspace(nr[i], nr[i + 1], local1 - m + 1)
                nitem = np.linspace(ni[i], ni[i + 1], local1 - m + 1)
                wltem = np.linspace(wl[i], wl[i + 1], local1 - m + 1)
                nrr.extend(nrtem)
                nii.extend(nitem)
                wll.extend(wltem)
        # 倒序
        wll2 = wll[::-1]
        plt.plot(wn, nr, 'o')
        plt.plot(wzong, np.array(nrr, dtype=np.float64), 'g-')
        plt.show()
        plt.plot(wn, ni, 'o')
        plt.plot(wzong, np.array(nii, dtype=np.float64), 'g-')
        plt.show()
        return wzong, nrr, nii, wll2

    def band9(self,wn1,wn2,wn3,wn4,wn5,wn6,wn7,dv):
        # 给定波数区间和该区间已有的折射指数（不是需要的wn），根据dv和nv，对每个子区间进行插值，得到每个子区间特定wn的折射指数
        nvl = self.nvl
        nv = self.nv

        nr = [1.302,1.303,1.305,1.306,1.307,1.308,1.309,1.310,1.311,1.313]
        ni = [0.0342,0.0342,0.0342,0.0342,0.0343,0.0342,0.0342,0.0342,0.0343,0.0346]
        wn = [1330,1340,1350,1360,1370,1380,1390,1400,1410,1420]
        wl = [7.042,7.092,7.143,7.194,7.246,7.299,7.353,7.407,7.463,7.519]
        # 从cy划分的每个区间波数中按照间隔nvl取波数
        w1s = np.linspace(wn1[0],wn1[-1],self.w1n+1)
        w2s = np.linspace(wn2[0], wn2[-1], self.w2n+1)
        w3s = np.linspace(wn3[0], wn3[-1], self.w3n+1)
        w4s = np.linspace(wn4[0], wn4[-1], self.w4n+1)
        w5s = np.linspace(wn5[0], wn5[-1], self.w5n+1)
        w6s = np.linspace(wn6[0], wn6[-1], self.w6n+1)
        w7s = np.linspace(wn7[0], wn7[-1], self.w7n+1)
        # 读取排序区间每个子区间波数最大值
        w1 = w1s[-1]
        w2 = w2s[-1]
        w3 = w3s[-1]
        w4 = w4s[-1]
        w5 = w5s[-1]
        w6 = w6s[-1]
        w7 = w7s[-1]

        # 看波数最大值在一直折射指数中波数所在位置
        wnloc1 = np.where(wn<=w1)[0][-1]
        wnloc2 = np.where(wn<=w2)[0][-1]
        wnloc3 = np.where(wn<=w3)[0][-1]
        wnloc4 = np.where(wn<=w4)[0][-1]
        wnloc5 = np.where(wn<=w5)[0][-1]
        wnloc6 = np.where(wn<=w6)[0][-1]
        wnloc7 = np.where(wn<=w7)[0][-1]

        # 合并要插值折射指数对应的wn
        wz1 = np.append(w1s, w2s, axis=0)
        wz2 = np.append(wz1, w3s, axis=0)
        wz3 = np.append(wz2, w4s, axis=0)
        wz4 = np.append(wz3, w5s, axis=0)
        wz5 = np.append(wz4, w6s, axis=0)
        wzong = np.append(wz5, w7s, axis=0)
##################################
        # 插值wns 内的折射指数nr ni
        nrr = []
        nii = []
        wll = []
        m =0
        for i in range(len(wn)-1):
            # 如果不是最后一截（不完整的那截）
            if i !=len(wn)-2:
                # 查找已有折射指数的波数在每个区间内的位置
                local1 = np.abs(np.subtract.outer(wzong, wn[i+1])).argmin(0)
                wntem = wzong[m:local1]
                nrtem = np.linspace(nr[i],nr[i+1],local1-m+1)
                nitem = np.linspace(ni[i], ni[i + 1], local1-m+1)
                wltem = np.linspace(wl[i], wl[i + 1], local1-m+1)

                nrr.extend(nrtem[0:- 1])
                nii.extend(nitem[0:- 1])
                wll.extend(wltem[0:- 1])
                m = local1
            else:
                local1 = np.abs(np.subtract.outer(wzong, wn[i + 1])).argmin(0)
                wntem = w1s[m:local1]
                nrtem = np.linspace(nr[i],nr[i+1],local1-m+1)
                nitem = np.linspace(ni[i], ni[i + 1], local1-m+1)
                wltem = np.linspace(wl[i], wl[i + 1], local1 - m + 1)
                nrr.extend(nrtem)
                nii.extend(nitem)
                wll.extend(wltem)
        # 倒序
        wll2 = wll[::-1]
        plt.plot(wn, nr, 'o')
        plt.plot(wzong, np.array(nrr, dtype=np.float64), 'g-')
        plt.show()
        plt.plot(wn, ni, 'o')
        plt.plot(wzong, np.array(nii, dtype=np.float64), 'g-')
        plt.show()
        return wzong,nrr,nii,wll2

    def band9_sort(self,wn,wn1,wn2,wn3,wn4,wn5,wn6,wn7):
        nr = [1.302,1.303,1.305,1.306,1.307,1.308,1.309,1.310,1.311,1.313]
        ni = [0.0342,0.0342,0.0342,0.0342,0.0343,0.0342,0.0342,0.0342,0.0343,0.0346]
        wnindex = [1330,1340,1350,1360,1370,1380,1390,1400,1410,1420]
        path = self.path
        nv = self.nv
        # 插值 折射指数
        wnloc1 = np.abs(np.subtract.outer(wn,wnindex[1])).argmin(0)
        wnloc2 = np.abs(np.subtract.outer(wn, wnindex[2])).argmin(0)
        wnloc3 = np.abs(np.subtract.outer(wn, wnindex[3])).argmin(0)
        wnloc4 = np.abs(np.subtract.outer(wn, wnindex[4])).argmin(0)
        wnloc5 = np.abs(np.subtract.outer(wn, wnindex[5])).argmin(0)
        wnloc6 = np.abs(np.subtract.outer(wn, wnindex[6])).argmin(0)
        wnloc7 = np.abs(np.subtract.outer(wn, wnindex[7])).argmin(0)
        wnloc8 = np.abs(np.subtract.outer(wn, wnindex[8])).argmin(0)
        wnloc9 = np.abs(np.subtract.outer(wn, wnindex[9])).argmin(0)
        wn1tem = wn[0:wnloc1+1]
        wn2tem = wn[wnloc1+1:wnloc2+1]
        wn3tem = wn[wnloc2+1:wnloc3+1]
        wn4tem = wn[wnloc3+1:wnloc4+1]
        wn5tem = wn[wnloc4+1:wnloc5+1]
        wn6tem = wn[wnloc5+1:wnloc6+1]
        wn7tem = wn[wnloc6+1:wnloc7+1]
        wn8tem = wn[wnloc7+1:wnloc8+1]
        wn9tem = wn[wnloc8+1:wnloc9+1]

        nrr = []
        nii = []
        lenwn = [np.shape(wn1tem)[0], np.shape(wn2tem)[0], np.shape(wn3tem)[0], np.shape(wn4tem)[0],
                 np.shape(wn5tem)[0], np.shape(wn6tem)[0], np.shape(wn7tem)[0], np.shape(wn8tem)[0],
                 np.shape(wn9tem)[0]]

        for i in range(len(wnindex)-1):
            nrtem = np.linspace(nr[i],nr[i+1],lenwn[i])
            nitem = np.linspace(ni[i],ni[i+1],lenwn[i])
            #
            nrr.extend(nrtem)
            nii.extend(nitem)
        # 插值完的nr,ni画图
        plt.plot(wnindex,nr,'o')
        plt.plot(wn,np.array(nrr,dtype=np.float64),'g-')
        plt.show()
        plt.plot(wnindex,ni,'o')
        plt.plot(wn,np.array(nii,dtype=np.float64),'g-')
        plt.show()

        # 读取。dat排序的序号
        file = glob.glob(os.path.join(path,"*.dat"))
        f = open(str(file[0]), encoding='utf-8')
        list = []
        for line in f:
            s = line.strip().split('\t')
            list.append(s)
        f.close()
        df_data = pd.DataFrame(list,columns=['data'])
        tem0 =[]
        # index = np.zeros([nv],dtype = np.int)
        for i in range(3,len(df_data),1):
            tem = np.array(df_data)[i][0].split(' ')
            a = [i for i in tem if i != ' ']
            aa = [i for i in a if i != '']
            aaa = [i for i in aa if i != ',']
            tem0.extend(np.array(aaa))
        idx = np.array(tem0,dtype=int)-1
        wnsort = wn[idx]
        # 排序每个子区间de波数
        wnn1 = wnsort[0:len(wn1)]
        wnn2 = wnsort[len(wn1):len(wn1)+len(wn2)]
        wnn3 = wnsort[len(wn1)+len(wn2):len(wn1)+len(wn2)+len(wn3)]
        wnn4 = wnsort[len(wn1)+len(wn2)+len(wn3):len(wn1)+len(wn2)+len(wn3)+len(wn4)]
        wnn5 = wnsort[len(wn1)+len(wn2)+len(wn3)+len(wn4):len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5)]
        wnn6 = wnsort[len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5):len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5)+len(wn6)]
        wnn7 = wnsort[len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5)+len(wn6):]
        # 排序插值的折射指数
        nr0 = np.array(nrr, dtype=np.float64)
        nrsort = nr0[idx]
        nrr1 = nrsort[0:len(wn1)]
        nrr2 = nrsort[len(wn1):len(wn1)+len(wn2)]
        nrr3 = nrsort[len(wn1)+len(wn2):len(wn1)+len(wn2)+len(wn3)]
        nrr4 = nrsort[len(wn1)+len(wn2)+len(wn3):len(wn1)+len(wn2)+len(wn3)+len(wn4)]
        nrr5 = nrsort[len(wn1)+len(wn2)+len(wn3)+len(wn4):len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5)]
        nrr6 = nrsort[len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5):len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5)+len(wn6)]
        nrr7 = nrsort[len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5)+len(wn6):]
        ni0 = np.array(nii, dtype=np.float64)
        nisort = ni0[idx]
        nii1 = nisort[0:len(wn1)]
        nii2 = nisort[len(wn1):len(wn1)+len(wn2)]
        nii3 = nisort[len(wn1)+len(wn2):len(wn1)+len(wn2)+len(wn3)]
        nii4 = nisort[len(wn1)+len(wn2)+len(wn3):len(wn1)+len(wn2)+len(wn3)+len(wn4)]
        nii5 = nisort[len(wn1)+len(wn2)+len(wn3)+len(wn4):len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5)]
        nii6 = nisort[len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5):len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5)+len(wn6)]
        nii7 = nisort[len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5)+len(wn6):]
        # 按给定的想要的间隔取nr,ni,wn
        wn1re = self.selec_idx(wnn1,np.shape(wnn1)[0]/self.w1n)
        wn2re = self.selec_idx(wnn2, np.shape(wnn2)[0] / self.w2n)
        wn3re = self.selec_idx(wnn3, np.shape(wnn3)[0] / self.w3n)
        wn4re = self.selec_idx(wnn4, np.shape(wnn4)[0] / self.w4n)
        wn5re = self.selec_idx(wnn5, np.shape(wnn5)[0] / self.w5n)
        wn6re = self.selec_idx(wnn6, np.shape(wnn6)[0] / self.w6n)
        wn7re = self.selec_idx(wnn7, np.shape(wnn7)[0] / self.w7n)
        wnz1 = np.append(wn1re , wn2re, axis=0)
        wnz2 = np.append(wnz1,wn3re, axis=0)
        wnz3 = np.append(wnz2, wn4re, axis=0)
        wnz4 = np.append(wnz3, wn5re, axis=0)
        wnz5 = np.append(wnz4, wn6re, axis=0)
        wnzong = np.append(wnz5, wn7re, axis=0)
        # 间隔取nr,ni,wn
        nr1re = self.selec_idx(nrr1,np.shape(nrr1)[0]/self.w1n)
        nr2re = self.selec_idx(nrr2, np.shape(nrr2)[0] / self.w2n)
        nr3re = self.selec_idx(nrr3, np.shape(nrr3)[0] / self.w3n)
        nr4re = self.selec_idx(nrr4, np.shape(nrr4)[0] / self.w4n)
        nr5re = self.selec_idx(nrr5, np.shape(nrr5)[0] / self.w5n)
        nr6re = self.selec_idx(nrr6, np.shape(nrr6)[0] / self.w6n)
        nr7re = self.selec_idx(nrr7, np.shape(nrr7)[0] / self.w7n)
        nrz1 = np.append(nr1re, nr2re, axis=0)
        nrz2 = np.append(nrz1, nr3re, axis=0)
        nrz3 = np.append(nrz2, nr4re, axis=0)
        nrz4 = np.append(nrz3, nr5re, axis=0)
        nrz5 = np.append(nrz4, nr6re, axis=0)
        nrzong = np.append(nrz5, nr7re, axis=0)
        ni1re = self.selec_idx(nii1,np.shape(nii1)[0]/self.w1n)
        ni2re = self.selec_idx(nii2, np.shape(nii2)[0] / self.w2n)
        ni3re = self.selec_idx(nii3, np.shape(nii3)[0] / self.w3n)
        ni4re = self.selec_idx(nii4, np.shape(nii4)[0] / self.w4n)
        ni5re = self.selec_idx(nii5, np.shape(nii5)[0] / self.w5n)
        ni6re = self.selec_idx(nii6, np.shape(nii6)[0] / self.w6n)
        ni7re = self.selec_idx(nii7, np.shape(nii7)[0] / self.w7n)
        niz1 = np.append(ni1re, ni2re, axis=0)
        niz2 = np.append(niz1, ni3re, axis=0)
        niz3 = np.append(niz2, ni4re, axis=0)
        niz4 = np.append(niz3, ni5re, axis=0)
        niz5 = np.append(niz4, ni6re, axis=0)
        nizong = np.append(niz5, ni7re, axis=0)
        return wnzong,nrzong,nizong


    def band6_sort(self,wn,wn1,wn2,wn3,wn4,wn5,wn6,wn7,wn8,wn9):

        nr = [1.132, 1.135, 1.139, 1.143, 1.149, 1.156, 1.162, 1.168, 1.174, 1.181]
        ni = [0.159, 0.144, 0.130, 0.118, 0.107, 0.0973, 0.0898, 0.0828, 0.0764, 0.0707]
        wnindex = [860, 870, 880, 890, 900, 910, 920, 930, 940, 950]

        path = self.path
        nv = self.nv
        # 插值 折射指数
        # 查找已有折射指数波数在总波数区间内位置
        wnloc1 = np.abs(np.subtract.outer(wn,wnindex[1])).argmin(0)
        wnloc2 = np.abs(np.subtract.outer(wn, wnindex[2])).argmin(0)
        wnloc3 = np.abs(np.subtract.outer(wn, wnindex[3])).argmin(0)
        wnloc4 = np.abs(np.subtract.outer(wn, wnindex[4])).argmin(0)
        wnloc5 = np.abs(np.subtract.outer(wn, wnindex[5])).argmin(0)
        wnloc6 = np.abs(np.subtract.outer(wn, wnindex[6])).argmin(0)
        wnloc7 = np.abs(np.subtract.outer(wn, wnindex[7])).argmin(0)
        wnloc8 = np.abs(np.subtract.outer(wn, wnindex[8])).argmin(0)
        wnloc9 = np.abs(np.subtract.outer(wn, wnindex[9])).argmin(0)
        # 折射指数插值 分区插
        wn1tem = wn[0:wnloc1+1]
        wn2tem = wn[wnloc1+1:wnloc2+1]
        wn3tem = wn[wnloc2+1:wnloc3+1]
        wn4tem = wn[wnloc3+1:wnloc4+1]
        wn5tem = wn[wnloc4+1:wnloc5+1]
        wn6tem = wn[wnloc5+1:wnloc6+1]
        wn7tem = wn[wnloc6+1:wnloc7+1]
        wn8tem = wn[wnloc7+1:wnloc8+1]
        wn9tem = wn[wnloc8+1:wnloc9+1]

        nrr = []
        nii = []
        lenwn = [np.shape(wn1tem)[0], np.shape(wn2tem)[0], np.shape(wn3tem)[0], np.shape(wn4tem)[0],
                 np.shape(wn5tem)[0], np.shape(wn6tem)[0], np.shape(wn7tem)[0], np.shape(wn8tem)[0],
                 np.shape(wn9tem)[0]]

        for i in range(len(wnindex)-1):
            nrtem = np.linspace(nr[i],nr[i+1],lenwn[i])
            nitem = np.linspace(ni[i],ni[i+1],lenwn[i])
            #
            nrr.extend(nrtem)
            nii.extend(nitem)
        # 插值完的nr,ni画图
        plt.plot(wnindex,nr,'o')
        plt.plot(wn,np.array(nrr,dtype=np.float64),'g-')
        plt.show()
        plt.plot(wnindex,ni,'o')
        plt.plot(wn,np.array(nii,dtype=np.float64),'g-')
        plt.show()

        # 读取。dat排序的序号
        file = glob.glob(os.path.join(path,"*.dat"))
        f = open(str(file[0]), encoding='utf-8')
        list = []
        for line in f:
            s = line.strip().split('\t')
            list.append(s)
        f.close()
        df_data = pd.DataFrame(list,columns=['data'])
        tem0 =[]
        # index = np.zeros([nv],dtype = np.int)
        for i in range(3,len(df_data),1):
            tem = np.array(df_data)[i][0].split(' ')
            a = [i for i in tem if i != ' ']
            aa = [i for i in a if i != '']
            aaa = [i for i in aa if i != ',']
            tem0.extend(np.array(aaa))

        idx = np.array(tem0,dtype=int)-1
        wnsort = wn[idx]
        # 排序每个子区间de波数
        wnn1 = wnsort[0:len(wn1)]
        wnn2 = wnsort[len(wn1):len(wn1)+len(wn2)]
        wnn3 = wnsort[len(wn1)+len(wn2):len(wn1)+len(wn2)+len(wn3)]
        wnn4 = wnsort[len(wn1)+len(wn2)+len(wn3):len(wn1)+len(wn2)+len(wn3)+len(wn4)]
        wnn5 = wnsort[len(wn1)+len(wn2)+len(wn3)+len(wn4):len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5)]
        wnn6 = wnsort[len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5):len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5)+len(wn6)]
        wnn7 = wnsort[len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5)+len(wn6):\
                      len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5)+len(wn6)+len(wn7)]
        wnn8 = wnsort[len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5)+len(wn6)+len(wn7):\
                    len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5)+len(wn6)+len(wn7)+len(wn8)]
        wnn9 = wnsort[len(wn1) + len(wn2) + len(wn3) + len(wn4) + len(wn5) + len(wn6) + len(wn7) + len(wn8):]
        # 排序插值的折射指数
        nr0 = np.array(nrr, dtype=np.float64)
        nrsort = nr0[idx]
        nrr1 = nrsort[0:len(wn1)]
        nrr2 = nrsort[len(wn1):len(wn1)+len(wn2)]
        nrr3 = nrsort[len(wn1)+len(wn2):len(wn1)+len(wn2)+len(wn3)]
        nrr4 = nrsort[len(wn1)+len(wn2)+len(wn3):len(wn1)+len(wn2)+len(wn3)+len(wn4)]
        nrr5 = nrsort[len(wn1)+len(wn2)+len(wn3)+len(wn4):len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5)]
        nrr6 = nrsort[len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5):len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5)+len(wn6)]
        nrr7 = nrsort[len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5)+len(wn6):\
               len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5)+len(wn6)+len(wn7)]
        nrr8 = nrsort[len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5)+len(wn6)+len(wn7):\
               len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5)+len(wn6)+len(wn7)+len(wn8)]
        nrr9 = nrsort[len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5)+len(wn6)+len(wn7)+len(wn8):]
        ni0 = np.array(nii, dtype=np.float64)
        nisort = ni0[idx]
        nii1 = nisort[0:len(wn1)]
        nii2 = nisort[len(wn1):len(wn1)+len(wn2)]
        nii3 = nisort[len(wn1)+len(wn2):len(wn1)+len(wn2)+len(wn3)]
        nii4 = nisort[len(wn1)+len(wn2)+len(wn3):len(wn1)+len(wn2)+len(wn3)+len(wn4)]
        nii5 = nisort[len(wn1)+len(wn2)+len(wn3)+len(wn4):len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5)]
        nii6 = nisort[len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5):len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5)+len(wn6)]
        nii7 = nisort[len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5)+len(wn6):\
               len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5)+len(wn6)+len(wn7)]
        nii8 = nisort[len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5)+len(wn6)+len(wn7):\
               len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5)+len(wn6)+len(wn7)+len(wn8)]
        nii9 = nisort[len(wn1)+len(wn2)+len(wn3)+len(wn4)+len(wn5)+len(wn6)+len(wn7)+len(wn8):]
        # 按给定的想要的间隔取nr,ni,wn
        wn1re = self.selec_idx(wnn1,np.shape(wnn1)[0]/self.w1n)
        wn2re = self.selec_idx(wnn2, np.shape(wnn2)[0] / self.w2n)
        wn3re = self.selec_idx(wnn3, np.shape(wnn3)[0] / self.w3n)
        wn4re = self.selec_idx(wnn4, np.shape(wnn4)[0] / self.w4n)
        wn5re = self.selec_idx(wnn5, np.shape(wnn5)[0] / self.w5n)
        wn6re = self.selec_idx(wnn6, np.shape(wnn6)[0] / self.w6n)
        wn7re = self.selec_idx(wnn7, np.shape(wnn7)[0] / self.w7n)
        wn8re = self.selec_idx(wnn8, np.shape(wnn8)[0] / self.w8n)
        wn9re = self.selec_idx(wnn9, np.shape(wnn9)[0] / self.w9n)
        wnz1 = np.append(wn1re , wn2re, axis=0)
        wnz2 = np.append(wnz1,wn3re, axis=0)
        wnz3 = np.append(wnz2, wn4re, axis=0)
        wnz4 = np.append(wnz3, wn5re, axis=0)
        wnz5 = np.append(wnz4, wn6re, axis=0)
        wnz6 = np.append(wnz5, wn7re, axis=0)
        wnz7 = np.append(wnz6, wn8re, axis=0)
        wnzong = np.append(wnz7, wn9re, axis=0)
        # 间隔取nr,ni,wn
        nr1re = self.selec_idx(nrr1,np.shape(nrr1)[0]/self.w1n)
        nr2re = self.selec_idx(nrr2, np.shape(nrr2)[0] / self.w2n)
        nr3re = self.selec_idx(nrr3, np.shape(nrr3)[0] / self.w3n)
        nr4re = self.selec_idx(nrr4, np.shape(nrr4)[0] / self.w4n)
        nr5re = self.selec_idx(nrr5, np.shape(nrr5)[0] / self.w5n)
        nr6re = self.selec_idx(nrr6, np.shape(nrr6)[0] / self.w6n)
        nr7re = self.selec_idx(nrr7, np.shape(nrr7)[0] / self.w7n)
        nr8re = self.selec_idx(nrr8, np.shape(nrr8)[0] / self.w8n)
        nr9re = self.selec_idx(nrr9, np.shape(nrr9)[0] / self.w9n)
        nrz1 = np.append(nr1re, nr2re, axis=0)
        nrz2 = np.append(nrz1, nr3re, axis=0)
        nrz3 = np.append(nrz2, nr4re, axis=0)
        nrz4 = np.append(nrz3, nr5re, axis=0)
        nrz5 = np.append(nrz4, nr6re, axis=0)
        nrz6 = np.append(nrz5, nr7re, axis=0)
        nrz7 = np.append(nrz6, nr8re, axis=0)
        nrzong = np.append(nrz7, nr9re, axis=0)
        ni1re = self.selec_idx(nii1,np.shape(nii1)[0]/self.w1n)
        ni2re = self.selec_idx(nii2, np.shape(nii2)[0] / self.w2n)
        ni3re = self.selec_idx(nii3, np.shape(nii3)[0] / self.w3n)
        ni4re = self.selec_idx(nii4, np.shape(nii4)[0] / self.w4n)
        ni5re = self.selec_idx(nii5, np.shape(nii5)[0] / self.w5n)
        ni6re = self.selec_idx(nii6, np.shape(nii6)[0] / self.w6n)
        ni7re = self.selec_idx(nii7, np.shape(nii7)[0] / self.w7n)
        ni8re = self.selec_idx(nii8, np.shape(nii8)[0] / self.w8n)
        ni9re = self.selec_idx(nii9, np.shape(nii9)[0] / self.w9n)
        niz1 = np.append(ni1re, ni2re, axis=0)
        niz2 = np.append(niz1, ni3re, axis=0)
        niz3 = np.append(niz2, ni4re, axis=0)
        niz4 = np.append(niz3, ni5re, axis=0)
        niz5 = np.append(niz4, ni6re, axis=0)
        niz6 = np.append(niz5, ni7re, axis=0)
        niz7 = np.append(niz6, ni8re, axis=0)
        nizong = np.append(niz7, ni9re, axis=0)
        return wnzong,nrzong,nizong

class savetxt(object):
    def __init__(self,wn,nr,ni,path):
        self.wn = wn
        self.nr =nr
        self.ni = ni
        self.path =path

    def saveA(self,wl):
        # 保存不排序的波数指定的区间每个子区间的nr,ni,wn
        path = self.path
        nr = self.nr
        ni = self.ni
        wn = self.wn


        with open(path+'\\wl.txt',"w") as f:
            for i in range(len(wl)):
                f.write(str(wl[i])+'\n')
        with open(path + '\\nr.txt',"w") as f:
            for i in range(len(wn)):
                f.write(str(nr[i]) + '\n')
        with open(path+'\\ni.txt',"w") as f:
            for i in range(len(wl)):
                f.write(str(ni[i])+'\n')
        with open(path+'\\wn.txt',"w") as f:
            for i in range(len(wl)):
                f.write(str(wn[i])+'\n')


if __name__=='__main__':
    bandname = 6
    maindata = dataku(bandname)
    # 区间wave number划分
    # wn1,wn2,wn3,wn4,wn5,wn6,wn7,wl,nvv,dv,wn=maindata.band9()
    wn1, wn2, wn3, wn4, wn5, wn6, wn7, wn8, wn9, wl, nv, dv, wn = maindata.band6()

    nvl = 1 #已有折射指数的指定区间的每个子区间内划分个数
    nv = 600000
    path = r'D:\work\data\CLOUD\index\band6'
    ######### 方案A不排序插值折射指数
    w1inter = 18
    w2inter = 18
    w3inter = 18
    w4inter = 17
    w5inter = 8
    w6inter = 2
    w7inter = 2
    w8inter = 1
    w9inter = 1
    # mainindexa = index_re(bandname,nvl,nv,path,w1inter,w2inter,w3inter,w4inter,w5inter,w6inter,w7inter)
    # wnn,nrr,nii,wlns = mainindexa.band9(wn1,wn2,wn3,wn4,wn5,wn6,wn7,dv)
    mainindexa = index_re(bandname,nvl,nv,path,w1inter,w2inter,w3inter,w4inter,w5inter,w6inter,w7inter,w8inter,w9inter)
    wnn,nrr,nii,wlns = mainindexa.band6(wn1,wn2,wn3,wn4,wn5,wn6,wn7,wn8,wn9,dv)

    # # 保存方案A结果
    mainsave= savetxt(wnn, nrr, nii, path)
    wlns = 10000 / np.array(wnn)
    mainsave.saveA(wlns)
    # 方案B排序插值折射函数
    dat = 'index_p3_t3.dat'
    pathsort = r'D:\work\data\CLOUD\index\band6sort'
    # nvlsort =
    # mainindexb = index_re(bandname,nvl,nv,pathsort,w1inter,w2inter,w3inter,w4inter,w5inter,w6inter,w7inter)
    # wnzong,nrzong,nizong = mainindexb.band9_sort(wn,wn1,wn2,wn3,wn4,wn5,wn6,wn7)
    mainindexb = index_re(bandname,nvl,nv,pathsort,w1inter,w2inter,w3inter,w4inter,w5inter,w6inter,w7inter,w8inter,w9inter)
    wnzong,nrzong,nizong = mainindexb.band6_sort(wn,wn1,wn2,wn3,wn4,wn5,wn6,wn7,wn8,wn9)
    # 保存方案B结果
    mainsave= savetxt(wnzong,nrzong,nizong, pathsort)
    wlsort = 10000/np.array(wnzong)
    mainsave.saveA(wlsort)

