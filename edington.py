"""
edington-beijing result
true result
date:2021-12-03
author:ida
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

def rf(omg0, g0, miu0, fo,tao0):
    # 相函数截断误差
    f = g0**2
    g = g0/(1+g0)
    tao = tao0*(1-f*omg0)
    omg = ((1-f)*omg0)/(1-omg0*f)


    miu1 = 0.5773503
    k = np.sqrt(3*(1-omg)*(1-omg*g))

    r = (1/2*omg*(1+3*g*(1-omg)*(miu0**2)))/(1-(miu0**2)*(k**2))
    P = k/(1-omg*g)
    q = (1/2+1/3*P)
    # D = ((1/2*miu0*r*np.exp(-tao/miu0))-(((np.exp(-k*tao))*miu0*r)/(4*q))-((miu0*np.exp(-k*tao))/(4*q))-((miu0*r*P*np.exp(-k*tao))/(6*q))-((miu0*P*np.exp(-k*tao))/(6*q))\
    #     +((1/2)*miu0*np.exp(tao/miu0)))/(((P*np.exp(-k*tao))/(6*q))-((np.exp(-k*tao))/(4*q))+(1/2*np.exp(k*tao))-(1/3*P*np.exp(k*tao))-((P*np.exp(-k*tao))/(6*q))+((P**2)*np.exp(-k*tao))/(9*q))
    # C = ((1/2)*miu0*r-1/2*D+1/2*miu0+1/3*P*D)/(1/2+1/3*P)
    D = ((P*(miu0**2)*np.exp(tao/miu0))-(3/2*miu0*r*np.exp(-tao/miu0)))/(2*P*np.exp(k*tao))
    C = (((miu0**2)*np.exp(tao/miu0))-(D*np.exp(k*tao)))/(np.exp(-k*tao))
    I0 = C*np.exp(-k*tao)+D*np.exp(k*tao)-miu0*(np.exp(tao/miu0))
    I1 = P*C*np.exp(-k*tao)-P*D*np.exp(k*tao)-(3/2)*miu0*r*np.exp(-tao/miu0)

    # 计算积分验证结果
    fx = lambda miu: ((C+D-miu0)+miu*(P*C-P*D-(3/2)*miu0*r))*miu
    y1 = scipy.integrate.quad(fx,0,1)
    fy = lambda miu: (I0+miu*I1)*miu
    y2 = scipy.integrate.quad(fy, 0, -1)
    FUP0 = 2*np.pi*miu1*(y1[0])
    FDOWN0 = 2*np.pi*miu1*y2[0]
    # FUP0 = 2*np.pi*np.pi*(I0+(2/3*I1))
    # FDOWN0 = 2*np.pi*np.pi*(I0-(2/3*I1))
    FUP = FUP0/(miu0*fo)
    FDOWN = FDOWN0/(miu0*fo)+np.exp(-tao/miu0)

    e = ((3/4)*omg*miu0*(1+g*(1-omg)))/(1-(miu0**2)*(k**2))
    pgang = 2/3*P
    N = ((1+pgang)**2)*(np.exp(k*tao))-(((1-pgang)**2)*(np.exp(-k*tao)))
    t = 4*pgang/N
    rgang = (1/N)*(1-pgang**2)*(np.exp(k*tao)-np.exp(-k*tao))
    R = (e-r)*((t*np.exp(-tao/miu0))-1)+(e+r)*rgang
    T = (e+r)*(t-np.exp(-tao/miu0))+((e-r)*rgang*np.exp(-tao/miu0))+np.exp(-tao/miu0)
    return I0,I1,R,T

if __name__ == '__main__':
    omg = 0.999999
    g = 0.837
    # tht = 10
    # miu0 = np.cos(tht*np.pi/180)
    # miu0 = 0.8
    # tao = 1

    fo = 15
    miu0s = np.arange(0.1,1,0.01)
    taos = [0.05,0.1,0.15,0.2,0.25,0.3]
    # print(miu0)

    I0 = np.zeros([len(taos),len(miu0s)],dtype=np.float64)
    I1 = np.zeros([len(taos), len(miu0s)], dtype=np.float64)
    FUP = np.zeros([len(taos), len(miu0s)], dtype=np.float64)
    FDOWN = np.zeros([len(taos), len(miu0s)], dtype=np.float64)
    i = 0

    for tao in taos:
        j = 0
        for miu0 in miu0s:
            I0[i,j],I1[i,j],FUP[i,j],FDOWN[i,j]=rf(omg, g, miu0, fo,tao)
            j = j+1
        i = i+1


    ln1 = plt.plot(miu0s,(FUP+FDOWN)[0,:],color = 'red',linewidth =1.0,linestyle = '-')
    ln2 = plt.plot(miu0s, (FUP + FDOWN)[1, :], color='blue', linewidth=1.0, linestyle='-')
    ln3 = plt.plot(miu0s, (FUP + FDOWN)[2, :], color='green', linewidth=1.0, linestyle='-')
    ln4 = plt.plot(miu0s, (FUP + FDOWN)[3, :], color='yellow', linewidth=1.2, linestyle='-')
    ln5 = plt.plot(miu0s, (FUP + FDOWN)[4, :], color='cyan', linewidth=1.0, linestyle='-')
    ln6 = plt.plot(miu0s, (FUP + FDOWN)[5, :], color='magenta', linewidth=1.0, linestyle='-')

    plt.legend([r'$\tau=%.3f$'% taos[0],r'$\tau=%.3f$'% taos[1],r'$\tau=%.3f$'% taos[2],\
                r'$\tau=%.3f$'% taos[3],r'$\tau=%.3f$'% taos[4],r'$\tau=%.3f$'% taos[5]])
    # plt.legend([r'$\tau=%.3f$' % taos[0], r'$\tau=%.3f$' % taos[1], r'$\tau=%.3f$' % taos[2], \
    #             r'$\tau=%.3f$'% taos[5]])
    my_x_ticks = np.arange(0,1.0,0.1)
    plt.xticks(my_x_ticks)
    plt.xlabel('solar azimuth angle')
    my_y_ticks = np.arange(0,1.2,0.1)
    plt.yticks(my_y_ticks)
    plt.ylabel('F')
    plt.show()

    fig, ax1 = plt.subplots()
    ax1.plot(miu0s,FUP[0,:],color = 'red',linewidth =1.0,linestyle = '-')
    ax1.plot(miu0s, FUP[1,:] , color='blue', linewidth=1.0, linestyle='-')
    ax1.plot(miu0s, FUP[2,:], color='green', linewidth=1.0, linestyle='-')
    ax1.plot(miu0s, FUP[3,:] , color='yellow', linewidth=1.2, linestyle='-')
    ax1.plot(miu0s, FUP[4,:] , color='cyan', linewidth=1.0, linestyle='-')
    ax1.plot(miu0s, FUP[5,:] , color='magenta', linewidth=1.0, linestyle='-')
    plt.legend([r'$\tau=%.3f$'% taos[0],r'$\tau=%.3f$'% taos[1],r'$\tau=%.3f$'% taos[2],\
                r'$\tau=%.3f$'% taos[3],r'$\tau=%.3f$'% taos[4],r'$\tau=%.3f$'% taos[5]])
    # plt.legend([r'$\tau=%.3f$' % taos[0], r'$\tau=%.3f$' % taos[1], r'$\tau=%.3f$' % taos[2], \
    #             r'$\tau=%.3f$'% taos[5]])
    my_x_ticks = np.arange(0,1,0.1)
    plt.xticks(my_x_ticks)
    plt.xlabel('solar azimuth angle')
    my_y_ticks = np.arange(0,1.1,0.1)
    plt.yticks(my_y_ticks)
    plt.ylabel('Fup')
    plt.show()

    ln11 = plt.plot(miu0s,(FDOWN)[0,:],color = 'red',linewidth =1.0,linestyle = '-')
    ln22 = plt.plot(miu0s, (FDOWN)[1, :], color='blue', linewidth=1.0, linestyle='-')
    ln33 = plt.plot(miu0s, (FDOWN)[2, :], color='green', linewidth=1.0, linestyle='-')
    ln44 = plt.plot(miu0s, (FDOWN)[3, :], color='yellow', linewidth=1.2, linestyle='-')
    ln55 = plt.plot(miu0s, (FDOWN)[4, :], color='cyan', linewidth=1.0, linestyle='-')
    ln66 = plt.plot(miu0s, (FDOWN)[5, :], color='magenta', linewidth=1.0, linestyle='-')
    plt.legend([r'$\tau=%.3f$'% taos[0],r'$\tau=%.3f$'% taos[1],r'$\tau=%.3f$'% taos[2],\
                r'$\tau=%.3f$'% taos[3],r'$\tau=%.3f$'% taos[4],r'$\tau=%.3f$'% taos[5]])
    # plt.legend([r'$\tau=%.3f$'% taos[0],r'$\tau=%.3f$'% taos[1],r'$\tau=%.3f$'% taos[2],\
    #            r'$\tau=%.3f$'% taos[5]])

    my_x_ticks = np.arange(0,1,0.1)
    plt.xticks(my_x_ticks)
    plt.xlabel('solar azimuth angle')
    my_y_ticks = np.arange(0,1.1,0.1)
    plt.yticks(my_y_ticks)
    plt.ylabel('Fdown')
    plt.show()