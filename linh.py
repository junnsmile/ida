import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

ds = nc.Dataset('D:\work\code\Mie\MIE_lh\water_group_extend_64.nc')
GG = ds.variables['GG'][:].data
Ld= ds.variables['Legendre-degrees'][:].data
Phase_fuc = ds.variables['Phase_fuc'][:].data
WWn_Legendre = ds.variables['WWn_Legendre'][:].data
wavelenths = ds.variables['wavelenths'][:].data
sizes = ds.variables['sizes'][:].data
angles = ds.variables['angles'][:].data
print(ds.variables.keys())
xtht = np.cos(angles*np.pi/180)
# 波长选择
wavenumber = 11
wave = wavelenths[wavenumber]

# 尺度选择
size = 2
mx = 2*np.pi*sizes[size]/wave

measurement = 2*np.pi*sizes/wave
print('尺度参数是：'+str(mx))

# c = np.zeros([104,257],dtype=np.float64)
c = WWn_Legendre[wavenumber,size,:]


p1 = xtht
p2 = 1/2*(3*xtht**2-1)
p3 = 1/2*(5*xtht**3-3*xtht)
p4 = 1/8*(35*xtht**4-30*xtht**2+3)
p5 = 1/8*(63*xtht**5-70*xtht**3+15*xtht)
p6 = 1/16*(231*xtht**6-315*xtht**4+105*xtht**2-5)
# p7 = (13*x*p6-6*p5)/7
# p8 = (15*x*p7-7*p6)/8
# p9 = (17*x*p8-8*p7)/9
# p10 = (19*x*p9-9*p8)/10
# p11= (21*x*p10-10*p9)/11
# p12 = (23*x*p11-11*p10)/12
pxh = np.zeros([257,len(xtht)],dtype=np.float64)
pxh[0,:] = 1
pxh[1,:] = p1
pxh[2,:] = p2
pxh[3,:] = p3
pxh[4,:] = p4
pxh[5,:] = p5
pxh[6,:] = p6
for n in range(7,257,1):
    pxh[n,:] = ((2*n-1)*xtht*pxh[n-1,:]-(n-1)*pxh[n-2,:])/n


# 读取勒让德展开系数
ctest = np.zeros(257,dtype=np.float64)
ctest[0] = 1
for j in range(1,257,1):
    ctest[j] = c[j-1]
    # ctem = c[j-1]
    # ctest[j] = np.float64(ctem)*(2*j+1)
    print(j)

p11s = 1
for k in range(1,257,1):
    p11s = p11s+ctest[k]*pxh[k,:]

plin = Phase_fuc[wavenumber,size,:]
# 单个粒子相函数画图
my_x_ticks = np.arange(0,190,10)
plt.xticks(my_x_ticks)

fig,ax = plt.subplots()
ax.set_yscale("log")
ax.set_adjustable("datalim")
ax.plot(angles,p11s,linewidth= 1.0,color='blue',linestyle = '-')
ax.plot(angles,plin,linewidth= 1.0,color='red',linestyle = '--')
# '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
ax.set_ylim(1e-3, 1e4)
plt.legend(['OURS','LIN'])
plt.xlabel("scattering angle")
plt.ylabel("phase function")
#  折射指数
# plt.legend(['α=1'])
af = 2*np.pi*sizes[size]/wave
plt.text(0.5,0.01,r'$\alpha=%.3f$'% mx,
         fontdict={'size':12,'color':'r'})
plt.show()