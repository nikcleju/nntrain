import keras.optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt	

#===================
# Define the network
#===================
# Define a sequential model
model = Sequential()

# Add the hidden layer
model.add(Dense(2, input_dim=2))
model.add(Activation('relu'))

# Add the output layer
model.add(Dense(1)) # automatically knows that it has 2 inputs, from previous layer
model.add(Activation('sigmoid'))

# Define optimizer and loss
# - optimizer = RMSprop
# - loss = MSE
#model.compile(optimizer='rmsprop', loss='mse')\
optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(optimizer=optimizer, loss='mse')


#========================
# Simulation parameters
#========================
numdata = 100
noisevar = 0.2
batchsize = 50
epochs = 200

#===================
# Generate the training data
#===================
import numpy as np

# Data is like [0,0] + noise, [0,1] + noise etc
train00 = np.tile( np.array([0, 0]), (numdata,1) ) + noisevar * np.random.randn(numdata,2)
train01 = np.tile( np.array([0, 1]), (numdata,1) ) + noisevar * np.random.randn(numdata,2)
train10 = np.tile( np.array([1, 0]), (numdata,1) ) + noisevar * np.random.randn(numdata,2)
train11 = np.tile( np.array([1, 1]), (numdata,1) ) + noisevar * np.random.randn(numdata,2)

test00 = np.tile( np.array([0, 0]), (numdata,1) ) + noisevar * np.random.randn(numdata,2)
test01 = np.tile( np.array([0, 1]), (numdata,1) ) + noisevar * np.random.randn(numdata,2)
test10 = np.tile( np.array([1, 0]), (numdata,1) ) + noisevar * np.random.randn(numdata,2)
test11 = np.tile( np.array([1, 1]), (numdata,1) ) + noisevar * np.random.randn(numdata,2)

label00 = np.zeros((numdata,1))
label01 = np.ones((numdata,1))
label10 = np.ones((numdata,1))
label11 = np.zeros((numdata,1))

# Concatenate and change datatype to float
trainset = np.array(np.vstack((train00, train01, train10, train11)), dtype=np.float32)
testset = np.array(np.vstack((test00, test01, test10, test11)), dtype=np.float32)
labels = np.vstack((label00, label01, label10, label11))

#===================
# Plot the training set
#===================
plt.scatter(trainset[:,0], trainset[:,1], c=labels)
plt.show()

#===================
# Auxiliary function to visualize the decision areas
#===================
def plot_separating_curve(model):
	"""
	Function to visualize the decision areas
	"""
	import matplotlib.pyplot as plt	
	
	points = np.array([(i, j) for i in np.linspace(0,1,100) for j in np.linspace(0,1,100)])
	#outputs = net(Variable(torch.FloatTensor(points)))
	outputs = model.predict(points)
	outlabels = outputs > 0.5
	plt.scatter(points[:,0], points[:,1], c=outlabels, alpha=0.5)
	plt.title('Decision areas')
	plt.show()
	
#===================
# Train
#===================
# Data = plain numpy arrays
model.fit(trainset, labels, epochs=epochs, batch_size=batchsize)

#===================
# Plot the decisions areas after the training
#===================
plot_separating_curve(model)
