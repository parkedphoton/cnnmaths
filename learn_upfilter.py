
import math

import numpy as np
import matplotlib.pyplot as plt


import filters


#generate the training data for the target filter of h
def training_data(h):
    L = 30 
    #L = 200    

    t = np.arange(0,L)
    v = []
    for tt in t:
        v.append(math.sin(math.pi/100*tt))
    v = np.array(v)
    #x = v    
    x = np.random.uniform(low=-0.5, high=0.5, size=L) + v
    #y = filters.sinc_filter2(x, scale=50)
    y = filters.upconv(x, h)

    return x,y



target_h = 2*filters.sinc(2.2)   #This may not be a very good upsample by 2 filter
print(target_h)
print(target_h.size)
x,y = training_data(target_h)


plt.plot(np.arange(0,len(x))*2, x, label='x')
plt.plot(y, label='y')
plt.legend()
plt.show()




#Stochastic gradient descent with a batch size of 1
#initialize h to be a zero array or with random values:
half_N = 11
#half_N = 25
#h = np.zeros(2*half_N+1, dtype='float')
h = np.random.uniform(size=(2*half_N+1,))/10
print(h)

track_L = []

alpha=0.00001
#alpha=0.00001
prev_dh = 0
beta=0.95

for i in range(50000):
    if i%1000 == 0:    
        print('-----------------iteration {}----------------'.format(i))
   
    #generate the test data
    x, y = training_data(target_h)

     
    #forward
    s = filters.upconv(x, h)
    L = ((y - s)**2).sum()/y.size

    #backward
    ds = 2*(s-y)
    xx = np.flip(x)    
    xx = np.insert(xx, range(1,len(xx)), 0)    
    dh = filters.convolve_noname(ds, xx, ext=half_N)
    
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
s = filters.upconv(x, h)

plt.plot(np.arange(0,len(x))*2, x, label='input')
plt.plot(y, label='target filter')
plt.plot(s, label='trained filter')
plt.legend()
plt.show()








