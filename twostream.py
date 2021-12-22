"""
Two-Stream scheme
date:2021.12.03
author:ida
"""
import numpy as np
def el0(omg0, g0, miu0, fo,tao0):
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
    # 相函数截断误差
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

    # H = ((e*u*np.exp(-tao/miu0))-(r*v*np.exp(-tao/miu0)))/(((v**2)*np.exp(-k*tao))-((u**2)*np.exp(-k*tao)))
    # K = ((-e*np.exp(-tao/miu0))-H*u*np.exp(-k*tao))/(v*np.exp(k*tao))
    # H = ((r*v*np.exp(k*tao))-(e*u*np.exp(-tao/miu0)))/(((u**2)*np.exp(-k*tao))-((v**2)*np.exp(k*tao)))
    # K = (-r-H*v)/u
    # K = (-e-H*u)/v
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
    omg = 0.999999
    g = 0.837
    miu0 = 0.5
    taos = [0.1,0.5,1.0,2.0]
    # print(miu0)
    fo = 15

    I0 = np.zeros([len(taos)],dtype=np.float64)
    I1 = np.zeros([len(taos)], dtype=np.float64)
    FUP = np.zeros([len(taos)], dtype=np.float64)
    FDOWN = np.zeros([len(taos)], dtype=np.float64)
    i = 0

    for tao in taos:
        FUP[i],FDOWN[i]=elshu(omg, g, miu0, fo, tao)
        i = i+1

    print(FUP,FDOWN,FUP+FDOWN)