
import math

import numpy as np
from scipy.signal import convolve


def sinc(scale=1):
    assert(scale >= 1)
    xs = np.arange(-int(5*scale),int(5*scale)+0.5)
           
    f = []    
    for x in xs:
        if x == 0:
            f.append(1.0)
        else:
            v = math.sin(math.pi*x/scale)/(math.pi*x/scale)*(1+math.cos(math.pi*x/5/scale))/2
            f.append(v) 
    
    taps = np.array(f)
    taps = taps/np.sum(taps)
    
    assert(len(taps)%2==1)
    assert(taps[int((len(taps)-1)/2)] == np.max(taps))

    return taps


def sinc_derivative(scale=1):
    assert(scale >= 1)
    xs = np.arange(-int(5*scale),int(5*scale)+0.5)
        
    f = []    
    for x in xs:
        if x == 0:
            f.append(0.0)
        else:
            v = ( math.cos(math.pi*x/scale)/(x/scale) - math.sin(math.pi*x/scale)/(math.pi*x/scale*x/scale) ) * ( 1+math.cos(math.pi*x/5/scale) )/2
            f.append(v) 
    
    taps = np.array(f)
    taps = taps/(scale*scale)

    assert(len(taps)%2==1)
    assert(taps[int((len(taps)-1)/2)] == 0)

    return taps







def sinc_filter2(x, scale=1):
    taps = sinc(scale)
    b = (len(taps)+1)//2        

    #Extend front and back
    N = len(x)
    assert(N > b)    
    x_extended = np.zeros((2*b+N,), dtype='float')
    x_extended[b:b+N] = x
    #front:    
    for i in range(1,b+1):
        x_extended[b-i] = x[0]
        #x_extended[b-i] = x[i]
        #x_extended[b-i] = x[0] - x[i]
    #back:
    for i in range(b):
        x_extended[b+N+i] = x[N-1]
        #x_extended[b+N+i] = x[N-1-i-1]
        #x_extended[b+N+i] = x[N-1] - x[N-1-i-1]
    

    #Filter
    y_extended = convolve(x_extended, taps, mode='same', method='direct')
    

    #Chop off the boundary extension
    y = np.array(y_extended[b:b+N])


    return y



def sinc_derivative_filter(x, scale=1):
    taps = sinc_derivative(scale)
    b = (len(taps)+1)//2        

    #Extend front and back
    N = len(x)
    assert(N > b)    
    x_extended = np.zeros((2*b+N,), dtype='float')
    x_extended[b:b+N] = x
    #front:    
    for i in range(1,b+1):
        x_extended[b-i] = x[0]
        #x_extended[b-i] = x[i]
        #x_extended[b-i] = x[0] - x[i]
    #back:
    for i in range(b):
        x_extended[b+N+i] = x[N-1]
        #x_extended[b+N+i] = x[N-1-i-1]
        #x_extended[b+N+i] = x[N-1] - x[N-1-i-1]
    

    #Filter
    y_extended = convolve(x_extended, taps, mode='same', method='direct')
    

    #Chop off the boundary extension
    y = np.array(y_extended[b:b+N])


    return y



#This is a special convolution where you can specify the amount of zero padded boundary extenstion on both side
def convolve_noname(x, y, ext=0):
    
    #Extend front and back
    N = len(x)    
    x_extended = np.zeros((2*ext+N,), dtype='float')
    x_extended[ext:ext+N] = x
    #front:    
    for i in range(1,ext+1):
        x_extended[ext-i] = 0        
        #x_extended[b-i] = x[0]
        #x_extended[b-i] = x[i]
        #x_extended[b-i] = x[0] - x[i]
    #back:
    for i in range(ext):
        x_extended[ext+N+i] = 0
        #x_extended[b+N+i] = x[N-1]
        #x_extended[b+N+i] = x[N-1-i-1]
        #x_extended[b+N+i] = x[N-1] - x[N-1-i-1]


    #Filter
    #y_extended = 
    return convolve(x_extended, y, mode='valid', method='direct')


#stride 0.5 convolution    
#upsample by 2 by inserting zeros and then convolve
def upconv(x, h, extra_end_zero = False):
    #This depends on whether you want your result to have one extra value or not to be consistent with the downsampling path    
    if extra_end_zero:
        x = np.insert(x, range(1,len(x)+1), 0)
    else:
        x = np.insert(x, range(1,len(x)), 0)    
    
    return convolve(x, h, mode='same')


#stride 2 convolution
#convolve and then downsample by 2
def downconv(x, h):
    y = convolve(x, h, mode='same')
    return y[0::2]










