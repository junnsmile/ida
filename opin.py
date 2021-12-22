"""
Interpolation of cloud optical properties
sort wavenumbers and optical properties
date:2021-12-22
"""

import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os

def readfile(df_data,wldatas,name):
    dataku = np.zeros([len(df_data),6999],dtype=np.float64)
    for i in range(len(df_data)):
        f = open(inputpath + '\\' + "{:.6f}".format(wldatas[i]) + '\\'+name, encoding='utf-8')
        datalist = []
        for line in f:
            s = line.strip().split('\t')
            datalist.append(s)
        df = pd.DataFrame(datalist,columns=['data'])
        dataku[i,:] = np.array(df,dtype=np.float64)[:,0]
    return dataku

def readcn(df_data,wldatas,name):
    c1ku = np.zeros([len(df_data),6999],dtype=np.float64)
    c2ku = np.zeros([len(df_data), 6999], dtype=np.float64)
    c3ku = np.zeros([len(df_data), 6999], dtype=np.float64)
    c4ku = np.zeros([len(df_data), 6999], dtype=np.float64)
    for i in range(len(df_data)):
        f = open(inputpath + '\\' + "{:.6f}".format(wldatas[i]) + '\\'+name, encoding='utf-8')
        c1list = []
        c2list =[]
        c3list = []
        c4list = []
        for line in f:
            s = line.strip().split('\t')
            tem0 = np.array(s)[0].split(' ')
            tem1 = [i for i in tem0 if i!='']
            tem = np.array(tem1,dtype=np.float64)
            c1list.append(tem[0])
            c2list.append(tem[1])
            c3list.append(tem[2])
            c4list.append(tem[3])
        # df = pd.DataFrame(datalist,columns=['data'])
        c1ku[i,:] = np.array(c1list,dtype=np.float64)
        c2ku[i, :] = np.array(c2list, dtype=np.float64)
        c3ku[i, :] = np.array(c3list, dtype=np.float64)
        c4ku[i, :] = np.array(c4list, dtype=np.float64)
    return c1ku,c2ku,c3ku,c4ku

def draw(x,y):
    plot.plot(x,y[:,0],linewidth = 1.0,color = 'red',linestyle='-')
    plot.plot(x, y[:, 10], linewidth=1.0, color='red', linestyle='-')
    plot.plot(x, y[:, 100], linewidth=1.0, color='red', linestyle='-')
    plot.plot(x, y[:, 1000], linewidth=1.0, color='red', linestyle='-')
    x_ticks = np.arange(7,8,0.1)
    y_ticks = np.arange(0,1,0.1)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.show()

def savet(data,name,len):
    datas = np.linspace(data[0, :], data[-1, :],len)
    np.savetxt(name, datas)
    print(name+'save complete!')

if __name__=='__main__':
    inputpath = r'D:\work\data\CLOUD\index\band9\mie'
    wlfile = r'D:\work\data\CLOUD\index\band9\wl.txt'
    wnfile = [1330,1420]
    sortpath = r'D:\work\data\CLOUD\index\band9sort'
    # 读取wavelength
    f = open(wlfile,encoding='utf-8')
    wllist = []
    for line in f:
        s1 = line.strip().split('\t')
        wllist.append(s1)
    df_data = pd.DataFrame(wllist,columns=['data'])

    wldata = []
    for i in range(len(df_data)):
        tem = np.array(df_data)[i][0]
        wldata.append(tem)

    wldatas = np.array(wldata,dtype = np.float64)
    # 读取wavenumber
    # fn = open(wnfile,encoding='utf-8')
    # wnlist = []
    # for line in fn:
    #     s1 = line.strip().split('\t')
    #     wnlist.append(s1)
    # df_datan = pd.DataFrame(wnlist,columns=['data'])
    #
    # wndata = []
    # for i in range(len(df_datan)):
    #     temn = np.array(df_datan)[i][0]
    #     wndata.append(temn)
    #
    # wndatas = np.array(wndata,dtype = np.float64)
    # 读取不同波长下云光学参数
    afadata = readfile(df_data,wldatas,'afa.txt')
    afedata = readfile(df_data, wldatas, 'afe.txt')
    gdata = readfile(df_data, wldatas, 'g.txt')
    omgdata = readfile(df_data, wldatas, 'omg.txt')
    c1,c2,c3,c4 = readcn(df_data, wldatas, 'cn.txt')

    # draw(wldatas,gdata)
    # draw(wldatas, afadata)
    # draw(wldatas,afedata)
    # draw(wldatas, omgdata)
    # draw(wldatas, c1)
    # draw(wldatas, c2)
    # draw(wldatas, c3)
    # draw(wldatas, c4)

    wls = np.linspace(wldatas[0],wldatas[-1],360000)
    wns = np.linspace(wnfile[0], wnfile[-1], 360000)
    np.savetxt(inputpath+"\\wls.txt",wls)
    np.savetxt(inputpath + "\\wns.txt", wns)
    # afas = np.linspace(afadata[0,:],afadata[-1,:],360000)
    # np.savetxt(inputpath+"\\afa.txt",afas)
    # savet(afedata,inputpath + "\\afe.txt",360000)
    # savet(gdata, inputpath + "\\g.txt", 360000)
    # savet(omgdata, inputpath + "\\omg.txt", 360000)
    # savet(c1, inputpath + "\\c1.txt", 360000)
    # savet(c2, inputpath + "\\c2.txt", 360000)
    # savet(c3, inputpath + "\\c3.txt", 360000)
    # savet(c4, inputpath + "\\c4.txt", 360000)
    # savet(afadata, inputpath + "\\afa.txt", 360000)

    # 读取。dat排序的序号
    file = glob.glob(os.path.join(sortpath, "*.dat"))
    f = open(str(file[0]), encoding='utf-8')
    list = []
    for line in f:
        s = line.strip().split('\t')
        list.append(s)
    f.close()
    df_data = pd.DataFrame(list, columns=['data'])
    tem0 = []
    # index = np.zeros([nv],dtype = np.int)
    for i in range(3, len(df_data), 1):
        tem = np.array(df_data)[i][0].split(' ')
        a = [i for i in tem if i != ' ']
        aa = [i for i in a if i != '']
        aaa = [i for i in aa if i != ',']
        tem0.extend(np.array(aaa))

    idx = np.array(tem0, dtype=int) - 1
    wnsort = wns[idx]
    wlsort = wls[idx]
    # gsort =
#