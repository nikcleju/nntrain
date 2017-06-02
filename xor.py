#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 23:21:53 2017

@author: nic
"""

# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.utils
import torch.utils.data
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

############
# Simulation parameters
############
numdata = 100
noisevar = 0.1
batchsize = 20
epochs = 10000

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Define a hidden lineaoutputs.datar layer
        # The relu function will be applied in forward()
        self.hidden1 =  nn.Linear(2, 2)
        # Define the output layer
        # The sigmoid function will be applied in forward()
        self.output =  nn.Linear(2, 1)

    def forward(self, x):
        # Propagate through hidden linear layer together with relu activation
        #out1 = F.relu(self.hidden1(x))
        out1 = F.sigmoid(self.hidden1(x))
        #out1 = F.tanh(self.hidden1(x))
        # Propagate through output linear layer together with thr activation
        out2 = F.sigmoid(self.output(out1))
        return out2

def plot_separating_curve(net):
	points = np.array([(i, j) for i in np.linspace(0,1,100) for j in np.linspace(0,1,100)])
	outputs = net(Variable(torch.FloatTensor(points)))
	outlabels = outputs > 0.5
	plt.scatter(points[:,0], points[:,1], c=outlabels.data.numpy(), alpha=0.5)
	plt.title('Decision areas')
	plt.show()

# Create the network object
net = Net()

# Let's use a Classification Cross-Entropy loss and SGD optimization
import torch.optim as optim
#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

#===================
# Generate the training data
#===================
import numpy as np
import torch

train00 = np.tile( np.array([0, 0]), (numdata,1) ) + noisevar * np.random.randn(numdata,2)
train01 = np.tile( np.array([0, 1]), (numdata,1) ) + noisevar * np.random.randn(numdata,2)
train10 = np.tile( np.array([1, 0]), (numdata,1) ) + noisevar * np.random.randn(numdata,2)
train11 = np.tile( np.array([1, 1]), (numdata,1) ) + noisevar * np.random.randn(numdata,2)

test00 = np.tile( np.array([0, 0]), (numdata,1) ) + noisevar * np.random.randn(numdata,2)
test01 = np.tile( np.array([0, 1]), (numdata,1) ) + noisevar * np.random.randn(numdata,2)
test10 = np.tile( np.array([1, 0]), (numdata,1) ) + noisevar * np.random.randn(numdata,2)
test11 = np.tile( np.array([1, 1]), (numdata,1) ) + noisevar * np.random.randn(numdata,2)

#label00 = np.zeros((100,1), dtype=int)
#label01 = np.ones((100,1), dtype=int)
#label10 = np.ones((100,1), dtype=int)
#label11 = np.zeros((100,1), dtype=int)
label00 = np.zeros((numdata,1))
label01 = np.ones((numdata,1))
label10 = np.ones((numdata,1))
label11 = np.zeros((numdata,1))

# Concatenate and change datatype to float
trainset = np.array(np.vstack((train00, train01, train10, train11)), dtype=np.float32)
testset = np.array(np.vstack((test00, test01, test10, test11)), dtype=np.float32)
labels = np.vstack((label00, label01, label10, label11))

#dataset = torch.utils.data.TensorDataset(torch.FloatTensor(trainset), torch.LongTensor(labels))
dataset = torch.utils.data.TensorDataset(torch.FloatTensor(trainset), torch.FloatTensor(labels))
trainloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=1)


#============
# Plot the training set
#============
plt.scatter(trainset[:,0], trainset[:,1], c=labels)
plt.show()

#============
# Plot the classification decisions before the training
#============
plot_separating_curve(net)


#============
# Training
#============

for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        
        tensor_inputs, tensor_labels = data
        
        # wrap inputs in Variable
        varinputs, varlabels = Variable(tensor_inputs), Variable(tensor_labels)
        # convert to flat to avoid errors
        #tensor_inputs = tensor_inputs.float()
        #tensor_labels = labels.float()
        
    
        # zero the parameter gradients
        optimizer.zero_grad()
    
        # forward + backward + optimize
        outputs = net(varinputs)

        #plt.scatter(varinputs.data.numpy()[:,0], varinputs.data.numpy()[:,1], c=outputs.data.numpy())
        #plt.show()

        loss = criterion(outputs, varlabels)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.data[0]
        numiter = 4*numdata/batchsize
        if i % numiter == (numiter-1):    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


#============
# Plot the classification decisions after the training
#============
plot_separating_curve(net)
