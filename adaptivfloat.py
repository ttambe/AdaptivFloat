#Reference paper: https://arxiv.org/pdf/1909.13271.pdf (T. Tambe et al.)

import numpy as np

def quantize_adaptivfloat(float_arr, n_bits=8, n_exp=4, bias = None):
    n_mant = n_bits-1-n_exp
    
    # 1. store sign value and do the following part as unsigned value
    sign = np.sign(float_arr)
    float_arr = abs(float_arr)
    
    # 1.5  if bias not determined, auto set exponent bias by the maximum input 
    if (bias == None):
       bias_temp = np.frexp(float_arr.max())[1]-1
       bias = bias_temp - (2**n_exp - 1)	
    
    # 2. limits the range of output float point
    min_exp = 0+bias
    max_exp = 2**(n_exp)-1+bias 
    
    ## min and max values of adaptivfloat
    min_value = 2.0**min_exp*(1+2.0**(-n_mant))
    max_value = (2.0**max_exp)*(2.0-2.0**(-n_mant))
    
    #print(min_value, max_value)
    ## 2.1. reduce too small values to zero
            
    float_arr[float_arr < 0.5*min_value] = 0
    float_arr[(float_arr > 0.5*min_value)*(float_arr < min_value)] = min_value
    
    ## 2.2. reduce too large values to max value of output format
    float_arr[float_arr > max_value] = max_value
    
    # 3. get mant, exp (the format is different from IEEE float)
    mant, exp = np.frexp(float_arr)
    
    # 3.1 change mant, and exp format to IEEE float format
    # no effect for exponent of 0 outputs
    mant = 2*mant
    exp = exp-1
    power_exp = np.exp2(exp)
    ## 4. quantize mantissa
    scale = 2**(-n_mant) ## e.g. 2 bit, scale = 0.25
    mant = ((mant/scale).round())*scale
    
    float_out = sign*power_exp*mant
   		
    float_out = float_out.astype("float32")
    return float_out
