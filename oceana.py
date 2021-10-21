"""
OCEAN SURFACE ALBEDO CALCULATE

AUTHOR:IDA
DATE:2021-09-22
"""
import numpy as np
import math
import netCDF4 as nc

class ocean():
    # def __init__(self):
    #     erfc_R = self.erfc_R
    #     erfc_Cheb = self.erfc_Cheb

    def ocnref__NN_roughness_R(self,u10):

        res = 0.00534 * u10
        return res


    def ocnref__NN_BRDF_ang_R(self,cr, ci, sigma2, mu0, phi0, mu1, phi1):
        # Direction cosines
        s0 = np.sqrt(1.0 - mu0 ** 2)
        s1 = np.sqrt(1.0 - mu1 ** 2)
        vec0 = np.zeros([3])
        vec1 = np.zeros([3])
        vec0[0] = s0 * np.cos(phi0)
        vec0[1] = s0 * np.sin(phi0)
        vec0[2] = mu0
        vec1[0] = s1 * np.cos(phi1)
        vec1[1] = s1 * np.sin(phi1)
        vec1[2] = mu1

    # Compute the BRDF
        res = self.ocnref__NN_BRDF_vec_R(cr, ci, sigma2, vec0, vec1)
        return res

    def ocnref__NN_BRDF_vec_R(self,cr, ci, sigma2, vec0, vec1):
        REPS = math.pow(10,-16)

        RMAX = 1.0e+35
        FPI = 0.56418958
        if (vec0[2] * vec1[2] > REPS):
            res = 0.0

   #Shadowing factors
        s2 = max(1.0e-4, sigma2)
        uz0 = -vec0[2]
        uz1 = vec1[2]
        vv0 = max(REPS, abs(uz0) /max(REPS, np.sqrt(s2 * max(0.0, 1.0 - uz0 ** 2))))
        vv1 = max(REPS, abs(uz1) /max(REPS, np.sqrt(s2 * max(0.0, 1.0 - uz1 ** 2))))
        fshad0 = 0.0
        fshad1 = 0.0
        if vv0 < 10.0:
            fshad0 = max(0.0, 0.5 * (np.exp(-vv0 ** 2) / vv0 * FPI - (self.erfc_R(vv0))))
        if vv1 < 10.0:
            fshad1 = max(0.0, 0.5 * (np.exp(-vv1 ** 2) / vv1 * FPI - (self.erfc_R(vv1))))

        # Facet normal vector
        vecn = np.zeros([3])
        vecn[0] = vec1[0] - vec0[0]        #facet normal vector(upward)
        vecn[1] = vec1[1] - vec0[1]
        vecn[2] = vec1[2] - vec0[2]
        vecn[:] = vecn[:] / np.sqrt(sum(vecn[:] ** 2))   # normalize

        # Other functions
        uza = vecn[0] * vec0[0] + vecn[1] * vec0[1] + vecn[2] * vec0[2]
        uzt, rho1, rho2, rho= self.fresnelRef1(cr, ci, -uza) # rho = Fresnel reflectance
        fshad = 1.0 / (1.0 + fshad0 + fshad1) # shadowing factor  for the bidirectional geometry
        funcs = rho * fshad / vecn[2] ** 4            # function S

        # BRDF
        uzn2 = vecn[2] ** 2
        rnum = funcs / (math.pi * s2) * np.exp(-(1.0 - uzn2) / (s2 * uzn2))

        deno = 4.0 * (-vec0[2]) * vec1[2]
        if (deno * RMAX > rnum): # to avoid too large BRDF values
            res = rnum / deno
        else:
            res = RMAX
        return res

    def fresnelRef1(self,rr, ri, uzi):
        RSML = 3*math.pow(10,-16)
        if (uzi <= 0.0) | (rr <= 0.0):
            uzt  =  0.0
            rhov = -1.0
            rhoh = -1.0
            rho  = -1.0

        else:
            uzi2 = uzi * uzi
            rr2 = rr * rr
            ri2 = ri * ri
            g2 = rr2 + uzi2 - 1.0

        #Partial reflection
        if (g2 > RSML):
            uzt = np.sqrt(g2) / rr
            w1 = rr2 - ri2
            w2 = 2.0 * rr * abs(ri)
            w3 = g2 - ri2
            wa = np.sqrt(w3 * w3 + w2 * w2)
            u = np.sqrt(0.5 * abs(wa + w3))
            v = np.sqrt(0.5 * abs(wa - w3))
            rhov = ((uzi - u)**2 + v*v) / ((uzi + u)**2 + v*v)
            rhoh = ((w1 * uzi - u)**2 + (w2 * uzi - v)**2) / ((w1 * uzi + u)**2 + (w2 * uzi + v)**2)
            rho = 0.5 * (rhov + rhoh)

              # 100% reflection
        else:
            uzt  = 0.0
            rhov = 1.0
            rhoh = 1.0
            rho  = 1.0
        return  uzt, rhov, rhoh, rho


    def erfc_R(self,x):
        if (x > 0.0):
            y = self.erfcCheb(x)
        else:
            y0 = self.erfcCheb(-x)
            y = 2.0 - y0
        return y

    def erfcCheb(self,z):

        t = 2.0 / (2.0 + z)
        y = -z**2 - 1.26551223 + t*(1.00002368 +t*(0.37409196 + t*(0.09678418 /
        + t*(-0.18628806 + t*(0.2788680 /
        + t*(-1.13520398 + t*(1.4851587 /
        + t*(-0.82215223 + t*0.17087277))))))))
        y = t * np.exp(y)
        return y



ds =nc.Dataset(r'D:\work\data\ertm\uvw\2016030507.nc')
u10 =ds.variables['u10'][:].data
# 讀取H8
h8file = r'D:\work\data\ertm\NC_H08_20160305_0700_R21_FLDK.02401_02401.nc'
ds = nc.Dataset(h8file)
print(ds.variables.keys())
SOZ = ds.variables['SOZ'][:].data[400:641,640:881]
SAZ = ds.variables['SAZ'][:].data[400:641,640:881]
# BRDF
#  real part of refractive index
cr = 1.16
# imaginary part of refractive index
ci =0
# incident direction azimuth angle (radian)
phi0=(180-SOZ)*math.pi/180
# incident direction cosine (should be < 0 for downward)
mu0 = np.cos(phi0)
# reflection direction azimuth angle (radian)
phi1 = SAZ*math.pi/180
# reflection direction cosine (should be > 0 for upward)
mu1 = np.cos(phi1)

main = ocean()
# sigma^2, surface roughness, variance of tangent
sigma2 = main.ocnref__NN_roughness_R(u10)

res = np.zeros([241,241],dtype=np.float64)
for i in range(241):
    for j in range(241):
        res[i,j] = main.ocnref__NN_BRDF_ang_R(cr, ci, sigma2[i,j], mu0[i,j], phi0[i,j], mu1[i,j], phi1[i,j])
        print('BRDF IS' +str(res))

print('finish!')
