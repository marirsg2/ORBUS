
import numpy as np

SCALER = 1
source  = np.concatenate((np.random.uniform(0.8,0.99,50),np.random.uniform(1.3,1.5,10)))
#==============================

a = SCALER*np.power(source,1)
# print(a)
print(np.mean(a))
print(np.var(a))
print(len([x for x in a if x<1.0]))
#==============================

a = SCALER*np.power(source,2)
# print(a)
print(np.mean(a))
print(np.var(a))
print(len([x for x in a if x<1.0]))
#==============================

a = SCALER*np.power(source,5)
# print(a)
print(np.mean(a))
print(np.var(a))
print(len([x for x in a if x<1.0]))
print(sorted([x for x in a if x>1.0],reverse=True))
print(sorted([x for x in a if x<1.0],reverse=True))
