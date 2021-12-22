

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate


def muly(r1gang, t1gang, r2gang, t2gang,r1,t1,r2,t2, tao1, tao2, miu1, miu2,miu0):
    # liou
    miu11 = 1/(np.sqrt(3))
    r2u0 = r2
    r1u0 = r1
    t1u0 = t1
    t2u0 = t2
    print(r1,t1,r2,t2)
    # umiu0 = ((r2u0*np.exp(-tao1/miu0))+(t1u0*r2gang))/(1-r1gang*r2gang)
    # dmiu0 = (t1u0+r2u0*r1gang*np.exp(-tao1/miu0))/(1-r1gang*r2gang)
    umiu0 = r2*(1/(1-r1*r2))*t1
    dmiu0 = (1/(1-r1*r2))*t1
    ###liou easy###
    # r12 = r1+t1*r2*(1/(1-r1*r2))*t1
    # t12 = t2*(1/(1-r1*r2))*t1
    ###liou complex###
    r12 = r1+t1*umiu0
    t12 = t2*dmiu0
    return r12,t12

def muly2(r1gang, t1gang, r2gang, t2gang,r1,t1,r2,t2, tao01, tao2,miu0,g0,omg0):
    # yangwen
    f = g0**2
    tao1 = tao01*(1-f*omg0)
    miu11 = 1/(np.sqrt(3))
    r2u0 = r2
    r1u0 = r1
    t1u0 = t1
    t2u0 = t2
    print(r1,t1,r2,t2)

    umiu0 = ((r2u0*np.exp(-tao1/miu0))+((t1u0-np.exp(-tao1/miu0))*r2gang))/(1-r1gang*r2gang)
    dmiu0 = (((t1u0-np.exp(-tao1/miu0))+(r2u0*r1gang*np.exp(-tao1/miu0)))/(1-r1gang*r2gang))

    ###yangwen#####
    t1u1b = t1gang
    t2u1b = t2gang
    # #
    r12 = r1u0+t1u1b*umiu0
    t12 = (t2u0*np.exp(-tao1/miu0))+t2u1b*dmiu0

    return r12,t12

def el0(omg0, g0, miu0, fo,tao0):
    # 单层二流近似 老师论文的
    f = g0**2
    g = g0/(1+g0)
    tao = tao0*(1-f*omg0)
    omg = ((1-f)*omg0)/(1-omg0*f)
    miu1 = 0.5773503
    k= np.sqrt(((1-omg)*(1-omg*g))/(miu1**2))
    a = -((fo*omg)/(2*np.pi))*(((1-omg*g)/(miu1**2))+3*g)*((miu0**2)/(1-(miu0**2)*(k**2)))
    b = ((fo*omg)/(2*np.pi))*((((1-omg)*3*g*miu0)/miu1)+(1/(miu0*miu1)))*((miu0**2)/(1-(miu0**2)*(k**2)))
    af = np.sqrt(1-omg*g)+np.sqrt(1-omg)
    bt = np.sqrt(1-omg*g)-np.sqrt(1-omg)
    r = 0.5*(a-b)
    e = 0.5*(a+b)
    D = ((e*bt*np.exp(-tao/miu0))-(r*af*np.exp(k*tao)))/(((af**2)*np.exp(k*tao))-((bt**2)*np.exp(-k*tao)))
    C = -(D*af+r)/bt
    tao2 =0
    IUP = C*af*np.exp(k*tao2)+D*bt*np.exp(-k*tao2)+e*np.exp(-tao2/miu0)
    IDOWN = C*bt*np.exp(k*tao)+D*af*np.exp(-k*tao)+r*np.exp(-tao/miu0)

    FUP = 2*np.pi*miu1*IUP
    FDOWN = 2*np.pi*miu1*IDOWN
    R = FUP/(fo*miu0)
    T = FDOWN/(fo*miu0)+np.exp(-tao/miu0)
    return R,T

def elshu(omg0, g0, miu0, fo,tao0):
    # 二流近似
    # 相函数截断误差 书上
    f = g0**2
    g = g0/(1+g0)
    tao = tao0*(1-f*omg0)
    omg = ((1-f)*omg0)/(1-omg0*f)

    miu1 = 0.5773503
    a = np.sqrt((1-omg)/(1-omg*g))
    k= np.sqrt(((1-omg)*(1-omg*g))/(miu1**2))
    sz = (fo*omg*(1+3*g*miu0*miu1))/(4*np.pi)
    sf = (fo * omg * (1 - 3 * g * miu0 * miu1)) / (4 * np.pi)
    z1 = -(((1-omg*g)*(sz+sf))/(miu1**2))+((sf-sz)/(miu0*miu1))
    z2 = -(((1-omg)*(sf-sz))/(miu1**2))+((sz+sf)/(miu0*miu1))
    af = (z1*(miu0**2))/(1-(miu0**2)*(k**2))
    bt = (z2*(miu0**2))/(1-(miu0**2)*(k**2))
    r = 0.5*(af-bt)
    e = 0.5*(af+bt)
    v = (1+a)/2
    u = (1-a)/2

    H = ((e*u*np.exp(-tao/miu0))-(r*v*np.exp(k*tao)))/((v**2*np.exp(k*tao))-(u**2*np.exp(-k*tao)))
    K = -((e*v*np.exp(-tao/miu0))-(r*u*np.exp(-k*tao)))/((v**2*np.exp(k*tao))-(u**2*np.exp(-k*tao)))
    IUP = K*v*np.exp(k*0)+H*u*np.exp(-k*0)+e*np.exp(-0/miu0)
    IDOWN = K*u*np.exp(k*tao)+H*v*np.exp(-k*tao)+r*np.exp(-tao/miu0)
    print('Iup,Idown'+str(IUP)+str(IDOWN))
    FUP = 2*np.pi*miu1*IUP
    FDOWN = 2*np.pi*miu1*IDOWN
    R = FUP/(fo*miu0)
    T = FDOWN/(fo*miu0)+np.exp(-tao/miu0)

    return R,T

if __name__ == '__main__':

###########累加法######
# 层数
    number =10
    omg1 = 0.9
    omg2 = 0.9
    g1 = 0.837
    g2 = 0.837
    miu1 = 1 / np.sqrt(3)
    miu2 = 1 / np.sqrt(3)
    miu0 = 0.5
    fo = 1
    tao = 0.1
    tao1 = tao/number
    tao2 =tao/number
    r1gang,t1gang= elshu(omg1, g1, miu1, fo, tao1)
    r2gang,t2gang= elshu(omg2, g2, miu2, fo, tao2)
    r1,t1 = elshu(omg1, g1, miu0, fo, tao1)
    r2,t2= elshu(omg2, g2, miu0, fo, tao2)

    ## yangwen
    if number == 2:
        # 单层
        r, t = elshu(omg2, g2, miu0, fo, tao)
        R12s,T12s = muly2(r1gang, t1gang, r2gang, t2gang,r1,t1,r2,t2, tao1, tao2,miu0,g1,omg1)

    else:
        # 多层
        R12gang = np.zeros([number-1],dtype=np.float64)
        T12gang = np.zeros([number - 1], dtype=np.float64)
        R12s = np.zeros([number - 1], dtype=np.float64)
        T12s = np.zeros([number - 1], dtype=np.float64)
        R12s[0], T12s[0] = muly2(r1gang, t1gang, r2gang, t2gang, r1, t1, r2, t2, tao1, tao2, miu0, g1, omg1)
        j = number-2
        for i in range(1,number-1,1):
            ## 多层
            R12gang[i], T12gang[i] = elshu(omg2, g2, miu1, fo, (tao-tao1*j))
            R12s[i], T12s[i] = muly2(R12gang[i], T12gang[i], r2gang, t2gang, R12s[i-1], T12s[i-1], r2, t2, tao-tao1*j, tao2, miu0,g1,omg1)
            print(i)
            j = j-1
            # R12gang4, T12gang4 = elshu(omg2, g2, miu1, fo, (tao - tao1))
            # R12ss, T12ss = muly2(R12gang4, T12gang4, r2gang, t2gang, R12s, T12s, r2, t2,tao - tao1, tao2, miu0, g1, omg1)
            # print('四层：' + str(R12ss) + str(T12ss))

            r, t = elshu(omg2, g2, miu0, fo, tao)
        print('单层：' + str(r) + str(t))
        print('多层：' + str(R12s[-1]) + str(T12s[-1]))
