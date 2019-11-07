
import itertools

a = [True,False]
b = [a]*5
c = itertools.product(*b)
print(list(c))
