import numpy as np
from scipy import integrate
from scipy.stats import norm
def get_gain_function(min_value,max_value):
    """
    :summary: abstract gain function
    :param x_value:
    :return:
    """
    range = max_value-min_value
    lower_fifth = min_value + range*0.2
    upper_fifth = min_value + range*0.8
    scaled_std_dev = range*0.1
    return get_biModal_gaussian_gain_function(lower_fifth,upper_fifth,scaled_std_dev)


#====================================================
def get_biModal_gaussian_gain_function(a= 0.2, b =0.8, sd = 0.1):
    """
    :summary: It averages two gaussians at means "a" and "b" who have the same std deviation.
    There is no special property with probabilities and summing to 1 in this application. Only relative gain values
    that match what we need.
    :param x_value:
    :param a:
    :param b:
    :return: the function
    """
    gain_a = norm(a,sd)
    gain_b = norm(b,sd)
    return lambda x: (gain_a.pdf(x) + gain_b.pdf(x))/2

#====================================================
def NthPoly_Ushape_gain_function(x_value, poly_n = 6, a= 0, b =1):
    """
    :summary: Extension of the U-quadratic distribution idea (see Quadratic gain function)
    https://en.wikipedia.org/wiki/U-quadratic_distribution
    Derived simply by ensuring the integral between [0,1]  = 1

    solve  Integral ( C*(x-0.5)^n ) between [a,b] = 1. For us [a,b] = [0,1]

    This gives C * ( 2*(0.5)^(n+1)/(n+1)) = 1 and thus we can solve for C

    For n = 6, Coeff = 448

    :param x_value:
    :return:
    """
    if poly_n%2 != 0:
        raise Exception("Cannot have odd number polynomial order for U shape polynomial distribution, saw n="+str(poly_n))

    #TODO optimize this so the coefficient is not calculated everytime !!
    # print("IMPORTANT, OPTIMIZE THIS SO THE COEFF IS NOT CALCULATED EVERYTIME, intensive float point arithmetic")

    middle = (a+b)/2
    range = abs(b-a)#abs is just in case people give the range in the wrong order
    #NOTE: Exponent operator is **, not ^ in python
    coeff = (2**poly_n)*(poly_n+1)/range**(poly_n+1)

    #TODO print logging /debug info of what the probability is btw top 20% and bottom 20% of the range
    return coeff*(x_value-middle)**poly_n

#====================================================
def LINEAR_gain_function(x_value):
    """
    :summary: return the y value
    :param x_value:
    :return:
    """
    #todo replace with two exponential distributions, that integrate to 1.
    # for NOW it is a trivial "v" shaped function that is piece-wise linear and integrates to 1.
    # if x< 0.5, y = -8x + 4, and if x>=0.5 y = 8x -4
    if x_value < 0.5 :
        return -8*x_value +4
    else:
        return 8*x_value - 4
#====================================================
def compute_expected_gain(Input_samples):
    """
    :summary : multiply the input function with the gain function and then integrate over [0,1] = domain of x values
    The input is a list of tuples of form (x,y) , the y value is multiplied with the gain function's y value at the associated
    x value.
    :param Input_samples:
    :return: Expected gain
    """
    expected_gain = 0.0
    multiplied_function_samples = [(sample[0],sample[1]*gain_function(sample[0])) for sample in Input_samples]
    #now integrate over the range [0,1]\
    #the integration function requires the values to be sorted...expected. 
    multiplied_function_samples = sorted(multiplied_function_samples, key = lambda x:x[0])
    x_vals = [sample[0] for sample in multiplied_function_samples]
    y_vals = [sample[1] for sample in multiplied_function_samples]
    expected_gain = integrate.trapz(y_vals,x_vals)
    return expected_gain


