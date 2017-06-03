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
import torch.optim as optim

#========================
# Simulation parameters
#========================
numdata = 100
noisevar = 0.2
batchsize = 50
epochs = 1000

#========================
# Define the network
# Can also be defined as a sequence of predefined layers from torch.nn
#========================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Define the data for 1 hidden layer and 1 output layer
        self.hidden1 =  nn.Linear(2, 2)
        self.output =  nn.Linear(2, 1)

    def forward(self, x):
		# This is where the actual computations are done
        # We apply the activation functions ourselves here
        
        # Propagate inputs through hidden layer and apply activation function
        x = F.relu(self.hidden1(x))
        #x = F.sigmoid(self.hidden1(x))
        #x = F.tanh(self.hidden1(x))
        
        # Propagate through output layer and apply activation function
        x = F.sigmoid(self.output(x))
        
        return x

#===================
# Create the network object
#===================
net = Net()

#===================
# Auxiliary function to visualize the decision areas
#===================
def plot_separating_curve(net):
	"""
	Function to visualize the decision areas
	"""
	
	points = np.array([(i, j) for i in np.linspace(0,1,100) for j in np.linspace(0,1,100)])
	outputs = net(Variable(torch.FloatTensor(points)))
	outlabels = outputs > 0.5
	plt.scatter(points[:,0], points[:,1], c=outlabels.data.numpy(), alpha=0.5)
	plt.title('Decision areas')
	plt.show()

#===================
# Define the loss criterion and optimization algorithm (SGD)
#===================
#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss(size_average=False)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#===================
# Generate the training data
#===================
import numpy as np
import torch

# Data is like [0,0] + noise, [0,1] + noise etc
train00 = np.tile( np.array([0, 0]), (numdata,1) ) + noisevar * np.random.randn(numdata,2)
train01 = np.tile( np.array([0, 1]), (numdata,1) ) + noisevar * np.random.randn(numdata,2)
train10 = np.tile( np.array([1, 0]), (numdata,1) ) + noisevar * np.random.randn(numdata,2)
train11 = np.tile( np.array([1, 1]), (numdata,1) ) + noisevar * np.random.randn(numdata,2)

test00 = np.tile( np.array([0, 0]), (numdata,1) ) + noisevar * np.random.randn(numdata,2)
test01 = np.tile( np.array([0, 1]), (numdata,1) ) + noisevar * np.random.randn(numdata,2)
test10 = np.tile( np.array([1, 0]), (numdata,1) ) + noisevar * np.random.randn(numdata,2)
test11 = np.tile( np.array([1, 1]), (numdata,1) ) + noisevar * np.random.randn(numdata,2)

# Labels must be of type float if using MSELoss, type int if using SoftMax (classification)
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

# pytorch specifics: make data as Tensors, make a DataSet, make a DataLoader for easy batch selection
#dataset = torch.utils.data.TensorDataset(torch.FloatTensor(trainset), torch.LongTensor(labels))
dataset = torch.utils.data.TensorDataset(torch.FloatTensor(trainset), torch.FloatTensor(labels))
trainloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=1)

#===================
# Plot the training set
#===================
plt.scatter(trainset[:,0], trainset[:,1], c=labels)
plt.show()

#===================
# Plot the decisions areas before the training
# - optional
#===================
#plot_separating_curve(net)

#===================
# Training
#===================
for epoch in range(epochs):

    running_loss = 0.0
    
    # DataLoader will provide the batches
    for i, data in enumerate(trainloader, 0):
        
        # Current batch, split inputs and labels
        batch_inputs, batch_labels = data
        
        # Wrap inputs in Variable
        varinputs, varlabels = Variable(batch_inputs), Variable(batch_labels)
    
        # Zero the parameter gradients
        optimizer.zero_grad()
    
        # Forward pass + backward pass + optimize
        outputs = net(varinputs)
        loss = criterion(outputs, varlabels)
        loss.backward()
        optimizer.step()
        
        # Print statistics
        running_loss += loss.data[0]
        totalnumiter = 4*numdata/batchsize
        if i % totalnumiter == (totalnumiter-1):    # print after full set
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

#===================
# Plot the decisions areas after the training
#===================
plot_separating_curve(net)
