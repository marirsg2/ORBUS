
import numpy as np

from scipy import stats
import statsmodels.api as sm

import numpy as np

# sp_d = sm.datasets.spector.load(as_pandas=False)
#
# sp_d.exog = sm.add_constant(sp_d.exog,prepend=False)
#
# mod = sm.OLS(sp_d.endog,sp_d.exog)
# res = mod.fit()
# print(res.summary())
#
# print(res.bse)
# print(res.params)
# print(mod.predict([1,0,0,0]))


import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std

np.random.seed(9876789)

nsample = 100
x = np.linspace(0, 10, 100)
X = np.column_stack((x, x**2))
beta = np.array([1, 0.1, 10])
e = np.random.normal(size=nsample)

X = sm.add_constant(X)
y = np.dot(X, beta) + e


model = sm.OLS(y, X)
# results = model.fit_regularized(method='elastic_net',alpha=0.005,L1_wt=0)
results = model._fit_ridge(alpha=0.005)
print(results)


print(results.predict([1,1,0]))