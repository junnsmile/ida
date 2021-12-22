"""
creat gamma distribution n(r)
author:ida
date:2021-10-26
"""

import numpy as np
from scipy.special import gamma,factorial
import matplotlib.pyplot as plt
import matplotlib as mpl

print(gamma(1.5))
N=5
x = np.linspace(0,N,50)
y= gamma(x+1)
plt.figure(facecolor='w')
plt.plot(x, y, 'r-', x, y, 'ro', linewidth=2, markersize=6, mec='k')
z = np.arange(0, N+1)

f = factorial(z, exact=True) # 阶乘

print(f)

plt.plot(z, f, 'go', markersize=9, markeredgecolor='k')

plt.grid(b=True, ls=':', color='#404040')

plt.xlim(-0.1,N+0.1)

plt.ylim(0.5, np.max(y)*1.05)

plt.xlabel('X', fontsize=15)

plt.ylabel('Gamma(X) - 阶乘', fontsize=12)

plt.title('阶乘和Gamma函数', fontsize=14)

plt.show()