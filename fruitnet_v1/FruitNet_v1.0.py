from fruitnet_api import *

#instantiate the network
fruitnet_v1=FruitNet()

#load in data from (my own) Google Drive
fruitnet_v1.loadData('training_set.h5','inputs','labels')

#add a hidden layer of size 128 with ReLU activation to the subnet
fruitnet_v1.addLayer('sub_hidden_1',128,relu)

#adjust each of the relevant hyperparameters
fruitnet_v1.adjustLearningRate(0.05)
fruitnet_v1.adjustEpochs(256)
fruitnet_v1.adjustBatchSize(128)
fruitnet_v1.adjustInitialization('Other')
fruitnet_v1.adjustStrips(4)

#view data
fruitnet_v1.viewData(4)
