


import numpy as np
from scipy.signal import convolve
import filters
import math
import matplotlib.pyplot as plt


#generate the training data for the target filter of h
def training_data(h):
    L = 100    

    t = np.arange(0,L)
    v = []
    for tt in t:
        v.append(math.sin(math.pi/100*tt))
    v = np.array(v)
    x = np.random.uniform(low=-0.5, high=0.5, size=L) + v
    #y = filters.sinc_filter2(x, scale=50)
    y = convolve(x, h, mode='same')

    return x,y

    




x = np.array([1,2,3,4,5])
h = np.array([5,4,3,2,1])

z = convolve(x, h, mode='full')



print(z)


z2 = filters.convolve_noname(x,h,ext=1)
print(z2)




#I'll need a special convolution operation when the input x and y are of the same size and I only need a small number of points in the output z

target_h = filters.sinc(1.5)
#target_h = filters.sinc_derivative(1.5)
#target_h = filters.sinc_derivative(5)
print(target_h)
print(target_h.size)


#Generate the training data
x, y = training_data(target_h)
plt.plot(x)
plt.plot(y)
plt.show()




#Stochastic gradient descent with a batch size of 1
#initialize h to be a zero array or with random values:
#half_N = 7
half_N = 25
h = np.zeros(2*half_N+1, dtype='float')
print(h)

track_L = []

alpha=0.000001
#alpha=0.00001
prev_dh = 0
beta=0.95
for i in range(50000):
    if i%1000 == 0:    
        print('-----------------iteration {}----------------'.format(i))
    #generate the test data
    x, y = training_data(target_h)
    
    #forward
    s = convolve(x, h, mode='same')
    L = ((y - s)**2).sum()

    #backward
    ds = 2*(s-y)
    dh = filters.convolve_noname(ds, np.flip(x), ext=half_N)
    
    #SGD with momentum:
    dh = beta*prev_dh + (1-beta)*dh
    h = h - alpha*dh
    prev_dh = dh
    track_L.append(L)


#print(track_L)
plt.plot(track_L)
plt.show()





#See how well it works
x, y = training_data(target_h)
s = convolve(x, h, mode='same')

plt.plot(x, label='input')
plt.plot(y, label='target filter')
plt.plot(s, label='trained filter')
plt.legend()
plt.show()




