#Reference paper: https://arxiv.org/pdf/1909.13271.pdf (T. Tambe et al.)

import torch 
import torch.nn.functional as F

def quantize_adaptivfloat(float_arr, n_bits=8, n_exp=4, bias = None):
    
    n_mant = n_bits-1-n_exp
    # 1. store sign value and do the following part as unsigned value
    sign = torch.sign(float_arr)
    float_arr = torch.abs(float_arr)
    
    if (bias == None):
       bias_temp = torch.frexp(float_arr.max())[1]-1
       bias = (2**(n_exp-1) - 1) - bias_temp
        
    # 2. limits the range of output float point
    min_exp = -2**(n_exp-1)+2-bias 
    max_exp = 2**(n_exp-1)-1-bias 
    
    min_value = 2.**min_exp
    max_value = (2.**max_exp)*(2-2**(-n_mant))

    # Non denormal part 
    float_arr[float_arr < min_value] = 0
    
    ## 2.2. reduce too large values to max value of output format
    float_arr[float_arr > max_value] = max_value
    
    # 3. get mant, exp (the format is different from IEEE float)
    mant, exp = torch.frexp(float_arr)
    
    # 3.1 change mant, and exp format to IEEE float format
    # no effect for exponent of 0 outputs
    mant = 2*mant
    exp = exp-1
   
    power_exp = torch.exp2(exp)
    ## 4. quantize mantissa
    scale = 2**(-n_mant) ## e.g. 2 bit, scale = 0.25
    mant = ((mant/scale).round())*scale
    
    float_out = sign*power_exp*mant
    
    return float_out
