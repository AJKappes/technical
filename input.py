import numpy as np
from scipy import stats


x = np.array([2, 3, 4])
factor = 2
print('Multiplication is:', factor * x)
print()

s_factor = float(input('Enter RV scaling factor: '))

rv = stats.norm.rvs(size=3)
scaled_rvs = rv * s_factor
print('Scaled normal RVs:', scaled_rvs)






