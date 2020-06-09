

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # kernel
        self.conv1 = nn.Conv1d(1, 1, 3, padding=1)
        self.conv2 = nn.Conv1d(1, 1, 3, padding=1)
        self.conv3 = nn.Conv1d(1, 1, 3, padding=1)
        self.conv4 = nn.Conv1d(1, 1, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x




net = Net()
net = net.float()
print(net)





#---------------------------The training----------------------------------
#target_h = filters.sinc(1.5)
#target_h = filters.sinc_derivative(1.5)
target_h = filters.sinc_derivative(5)
print(target_h)
#print(target_h.size)


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

plt.plot(xx, label='input')
plt.plot(y, label='target filter')
plt.plot(s.numpy(), label='trained filter')
plt.legend()
plt.show()











