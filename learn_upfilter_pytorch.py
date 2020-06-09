


import math

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # kernel
        self.conv1 = nn.ConvTranspose1d(1, 1, 23, padding=11, stride=2)


    def forward(self, x):
        x = self.conv1(x)
        return x




net = Net()
net = net.float()


print(net)


'''
print(x)
xx = torch.from_numpy(x)
xx = xx.unsqueeze(0)  #1 channel
xx = xx.unsqueeze(0)  #1 example
s = net.forward(xx.float())
print(s)
print(s.shape)
print(y.shape)
'''



#---------------------------The training----------------------------------

criterion = nn.MSELoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.95)  #better
#optimizer = optim.Adam(net.parameters(), lr=0.001)   %also better


track_L = []


for i in range(5000):
    if i%1000 == 0:    
        print('-----------------iteration {}----------------'.format(i))
    #get the training data
    x, y = training_data(target_h)
    
    x = torch.from_numpy(x)
    x = x.unsqueeze(0)  #1 channel
    x = x.unsqueeze(0)  #1 example
    y = torch.from_numpy(y)

    optimizer.zero_grad()    


    #forward + backward + optimize
    s = net(x.float())
    s = s.squeeze()    
    L = criterion(s, y.float())
    L.backward()
    optimizer.step()

    
    
    track_L.append(L)


#print(track_L)
plt.plot(track_L)
plt.show()


#See how well it works
xx, y = training_data(target_h)

with torch.no_grad():
    x = torch.from_numpy(xx)
    x = x.unsqueeze(0)  #1 channel
    x = x.unsqueeze(0)
    
    s = net(x.float())
    s = s.squeeze()

plt.plot(np.arange(0,len(xx))*2, xx, label='input')
plt.plot(y, label='target filter')
plt.plot(s.numpy(), 'g--', label='trained filter')
plt.legend()
plt.show()
















