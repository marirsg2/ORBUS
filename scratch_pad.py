
import numpy as np

from scipy import stats


precision_variance = 10 #norm by variance of the dataset. THIS way the value is not small and hurts the behavior of F beta function
recall_gain = 100

#when gain == variance, then they transition. to gain not mattering much. So norm by variance may help.
#higher the variance, more the effect of gain. GREAT PROPERTY
#less variance, less effect of gain.
#BETA tells you how much you want gain to matter. Could get a lot of early gains, but much slower later

beta = 0.0 # Beta penalizes precision, is chosen such that recall is considered β times as important as precision
F_beta = (1+beta**2) * precision_variance * recall_gain / (beta ** 2 * precision_variance + recall_gain)
print(F_beta)

"""
When beta is small, near 0. Then variance is king. as beta increases, we are giving gain more value
"""


#
# #Studnt, n=999, p<0.05, 2-tail
# #equivalent to Excel TINV(0.05,999)
# print ( stats.t.ppf(1-0.025, 999))
#
# #Studnt, n=999, p<0.05%, Single tail
# #equivalent to Excel TINV(2*0.05,999)
# print( stats.t.ppf(1-0.0015, 999))
# print( stats.t.ppf(1-0.0015, 22))
# #
# # SCALER = 1
# # source  = np.concatenate((np.random.uniform(0.8,0.99,50),np.random.uniform(1.3,1.5,10)))
# # #==============================
# #
# # a = SCALER*np.power(source,1)
# # # print(a)
# # print(np.mean(a))
# # print(np.var(a))
# # print(len([x for x in a if x<1.0]))
# # #==============================
# #
# # a = SCALER*np.power(source,2)
# # # print(a)
# # print(np.mean(a))
# # print(np.var(a))
# # print(len([x for x in a if x<1.0]))
# # #==============================
# #
# # a = SCALER*np.power(source,5)
# # # print(a)
# # print(np.mean(a))
# # print(np.var(a))
# # print(len([x for x in a if x<1.0]))
# # print(sorted([x for x in a if x>1.0],reverse=True))
# # print(sorted([x for x in a if x<1.0],reverse=True))
