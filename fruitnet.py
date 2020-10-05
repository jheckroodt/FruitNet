import numpy as np
import h5py
from compiler import *
from initialization import *

class fruitnet():
    
    def __init__(self):
        '''
        Description:
        This method initializes the hyperparameter dictionary the network() method requires to operate
        properly

        Inputs:
        none

        Outputs:
        Initializes self.__hp 
        '''
        self.__hp={'conv_layers':int(0),
                   'conv_layer_types':[],
                   'padding':[],
                   'stride':[],
                   'channels':[],
                   'f_values':[],
                   'connected_layers':int(0),
                   'connected_layer_dims':[],
                   'activations':[],
                   'layer_names':[],
                   'batch_size':int(0),
                   'epochs':int(0),
                   'learning_rate':float(0),
                   'beta1':float(0),
                   'beta2':float(0),
                   'cost_function':None}
    
    def Input(self,shape):
        '''
        Description:
        This method allows the user to specify the shape of the data being input into the network.

        Inputs:
        shape - the shape of the input, in the form (x,x,y) (that is, the input must be square, per sé)

        Outputs:
        Updates self.__input_shape accordingly
        '''
        try:
            if self.__X_train.shape[1:]!=shape:
                raise ValueError('The proposed input layer shape does not match the dimensions of the data you\'ve loaded in.')
        except AttributeError:
            raise AttributeError('You have not loaded in any data. Please load in some data, then try again.')
        if shape[0]==shape[1]:
            self.__input_shape=shape
        else:
            raise AttributeError('Your specified input shape is of incorrect dimensions.')

    def Conv2D(self,name,channels,filters=3,padding=0,stride=1,activation='relu'):
        '''
        Description:
        This method allows the user to add to the network a convolutional layer
        
        Inputs:
        name - the name of the layer
        channels - the number of channels our filter has
        filter - this layer utilizes a (filter,filter) filter
        padding - the padding applied to the input to this layer
        stride - the stride our filter makes
        activation - the activation function applied to our output volume

        Outputs:
        Updates self.__hp accordingly
        '''
        if self.__hp['connected_layers']==0:
            self.__hp['conv_layers']+=1
            self.__hp['conv_layer_types'].append('conv')

            if activation not in ['relu','sigmoid']:
                raise NameError('Invalid activation function')
            if name in self.__hp['layer_names']:
                raise NameError('Duplicate name')

            self.__hp['layer_names'].append(name)
            self.__hp['channels'].append(channels)
            self.__hp['f_values'].append(filters)
            self.__hp['padding'].append(padding)
            self.__hp['stride'].append(stride)
            self.__hp['activations'].append(activation)
        else:
            raise IndexError('Unable to add a convolutional layer after a fully connected layer.')

    def Maxpool(self,name,filters=3,padding=0,stride=1,activation='relu'):
        '''
        Description:
        This method allows the user to add to the network a convolutional layer
        
        Inputs:
        name - the name of the layer
        channels - the number of channels our filter has
        filter - this layer utilizes a (filter,filter) filter
        padding - the padding applied to the input to this layer
        stride - the stride our filter makes
        activation - the activation function applied to our output volume

        Outputs:
        Updates self.__hp accordingly
        '''

        self.__hp['conv_layers']+=1
        self.__hp['conv_layer_types'].append('maxpool')

        if activation not in ['relu','sigmoid']:
            raise NameError('Invalid activation function')
        if name in self.__hp['layer_names']:
            raise NameError('Duplicate name')

        self.__hp['layer_names'].append(name)
        self.__hp['f_values'].append(filters)
        self.__hp['padding'].append(padding)
        self.__hp['stride'].append(stride)
        self.__hp['activations'].append(activation)
        self.__hp['channels'].append(self.__hp['channels'][-1])

    def Connected(self,name,hidden_units,activation='relu'):
        '''
        Description:
        This method allows the user to add a fully connected layer to the network

        Inputs:
        name - name of the layer
        hidden_units - the number of hidden units found in the layer
        activation - the activation applied to the layer

        Outputs:
        Updates self.__hp accordingly
        '''
        self.__hp['connected_layers']+=1

        if activation not in ['relu','sigmoid']:
            raise NameError('Invalid activation function')
        if name in self.__hp['layer_names']:
            raise NameError('Duplicate name')

        self.__hp['layer_names'].append(name)
        self.__hp['connected_layer_dims'].append(hidden_units)
        self.__hp['activations'].append(activation)

    def Output(self,name,hidden_units,loss='softmax_cost',activation='softmax'):
        '''
        Description:
        This layer allows the user to identify the final (fully connected) layer of the network

        Inputs:
        name - the name of the layer
        hidden_units - the last layer in our network must be fully connected, hence we need the nr of hidden units
        loss - the loss function according to which our network will evaluate the quality of its predictions
        activation - the activation used in this layer

        Outputs:
        Updates self.__hp accordingly
        '''
        self.__hp['connected_layers']+=1

        if loss not in ['softmax_cost','mse_cost']:
          raise AttributeError('Invalid loss function.')
        if activation not in ['sigmoid','relu','softmax']:
            raise NameError('Invalid activation function')
        if name in self.__hp['layer_names']:
            raise NameError('Duplicate name')

        self.__hp['layer_names'].append(name)
        self.__hp['connected_layer_dims'].append(hidden_units)
        self.__hp['cost_function']=loss
        self.__hp['activations'].append(activation)

    def adjustTrainingSettings(self):
        '''
        Description:
        This method allows the user to make changes to the settings that aren't dictated by the
        rest of the methods belonging to this class using a simple command line interface

        Inputs:
        none

        Outputs:
        Updates self.__hp accordingly
        '''
        ops=['mini batch size','number of epochs','learning rate','momentum parameter','RMSprop parameter']
        hp_ops=['batch_size','epochs','learning_rate','beta1','beta2']
        mainloop=True
        print('The current training settings are as follows:\n1. Mini batch size: '+str(self.__hp['batch_size'])+'\n2. Epochs: '+str(self.__hp['epochs'])+'\n3. Learning rate: '+str(self.__hp['learning_rate'])+'\n4. Momentum parameter: '+str(self.__hp['beta1'])+'\n5. RMSprop parameter: '+str(self.__hp['beta2'])+'\nTo adjust any of the aforementioned settings, please input the number of the setting you would like to adjust.\n')
        while mainloop:
            n=int(input())
            assert(n in [1,2,3,4,5])
            print('Please enter the new value you would like to assign to the '+str(ops[n-1])+'.')
            self.__hp[str(hp_ops[n-1])]=input()
            if int(input('Would you like to adjust another setting?\n1 - Yes\n2 - No\n'))==2:
                mainloop=False
            print('The current training settings are as follows:\n1. Mini batch size: '+str(self.__hp['batch_size'])+'\n2. Epochs: '+str(self.__hp['epochs'])+'\n3. Learning rate: '+str(self.__hp['learning_rate'])+'\n4. Momentum parameter: '+str(self.__hp['beta1'])+'\n5. RMSprop parameter: '+str(self.__hp['beta2']))
            if mainloop:
                print('To adjust any of the aforementioned settings, please input the number of the setting you would like to adjust.')

    def loadData(self,filename,input_dataset,label_dataset,test=False):
        '''
        Description:
        This method loads the training data, given a .h5 file with name filename, an input dataset of name input_data
        and a label_dataset of name label_dataset

        Inputs:
        filename - the name of the file containing the training set
        input_dataset - the name of the dataset containing the input features to our network
        label_dataset - the name of the dataset containing the labels of each input example in input_dataset
        test - whether or not it is the test set being loaded

        Outputs:
        Updates either self.__X_train and self.__Y_train or self.__X_test and self.__Y_test accordingly
        '''
        dataset=h5py.File(filename,'r')
        X=np.array(dataset[str(input_dataset)])
        if len(X.shape)<3:
          raise AttributeError('You have loaded in data of an incorrect format. Please load in different data, and try again.')
        Y=np.array(dataset[label_dataset])
        if X.shape[0]==Y.shape[0]:
            Y=(Y.T).copy()
        elif X.shape[0]!=Y.shape[1]:
            raise AttributeError('Your input dataset and label dataset appear to have conflicting sizes.')
        dataset.close()
        if test:
            self.__X_test=X
            self.__Y_test=Y
        else:
            self.__X_train=X
            self.__Y_train=Y

    def saveModel(self,filename,inference=False,epsilon=1e-8):
        '''
        Description:
        Saves the model either mid training or as a ready-for-inference model in a .h5 file called filename

        Inputs:
        filename - the name of the .h5 file in which our network will be stored
        inference - if True, the network's batch normalization parameters are adjusted to prepare them for inference

        Outputs:
        Saves the model to a file called filename.h5
        '''
        try:
            normalize_params=self.__normalize_params
            cache_list=self.__cache_list
            parameters=self.__parameters
        except AttributeError:
            raise AttributeError('Your network has not been trained. Please train your network, then try again.')
        net=h5py.File(filename,'w')
        (mu,var)=normalize_params
        net.create_dataset('mean',data=mu)
        net.create_dataset('variance',data=var)
        (conv_layers,conv_layer_types,connected_layers)=(self.__hp['conv_layers'],self.__hp['conv_layer_types'],self.__hp['connected_layers'])
        if inference:
            for l in range(conv_layers):
                if conv_layer_types[l]=='conv':
                    (conv_cache,batch_norm_cache,activation_cache)=cache_list[l]
                    (gamma,Zhat,mu,Z,var)=batch_norm_cache
                    net.create_dataset('W'+str(l+1),data=parameters['W'+str(l+1)])
                    net.create_dataset('gamma'+str(l+1),data=parameters['gamma'+str(l+1)]/np.sqrt(var+epsilon))
                    net.create_dataset('beta'+str(l+1),data=parameters['beta'+str(l+1)]-mu*(parameters['gamma'+str(l+1)]/np.sqrt(var+epsilon)))
            for l in range(conv_layers,conv_layers+connected_layers):
                (linear_cache,batch_norm_cache,activation_cache)=cache_list[l+1]
                (gamma,Zhat,mu,Z,var)=batch_norm_cache
                net.create_dataset('W'+str(l+1),data=parameters['W'+str(l+1)])
                net.create_dataset('gamma'+str(l+1),data=parameters['gamma'+str(l+1)]/np.sqrt(var+epsilon))
                net.create_dataset('beta'+str(l+1),data=parameters['beta'+str(l+1)]-mu*(parameters['gamma'+str(l+1)]/np.sqrt(var+epsilon)))
        else:
            for l in range(conv_layers):
                if conv_layer_types[l]=='conv':
                    net.create_dataset('W'+str(l+1),data=parameters['W'+str(l+1)])
                    net.create_dataset('gamma'+str(l+1),data=parameters['gamma'+str(l+1)])
                    net.create_dataset('beta'+str(l+1),data=parameters['beta'+str(l+1)])
            for l in range(conv_layers,conv_layers+connected_layers):
                net.create_dataset('W'+str(l+1),data=parameters['W'+str(l+1)])
                net.create_dataset('gamma'+str(l+1),data=parameters['gamma'+str(l+1)])
                net.create_dataset('beta'+str(l+1),data=parameters['beta'+str(l+1)])
        net.close()

    def loadModel(self,filename):
        '''
        Description:
        Loads a model (which isn't ready for inference) in accordance with the specified model architecture.

        Inputs:
        filename - the name of the .h5 file in which our network will be stored
        inference - if True, it is assumed the network is no longer to be trained, and hence running .train on
                    loaded model will raise a TypeError

        Outputs:
        Updates self.__parameters accordingly
        '''
        self.__parameters={}
        net=h5py.File(filename,'r')
        normalize_params=(np.array(net['mean']),np.array(net['variance']))
        (conv_layers,conv_layer_types,connected_layers)=(self.__hp['conv_layers'],self.__hp['conv_layer_types'],self.__hp['connected_layers'])
        for l in range(conv_layers):
            if conv_layer_types[l]=='conv':
                self.__parameters['W'+str(l+1)]=np.array(net['W'+str(l+1)])
                self.__parameters['gamma'+str(l+1)]=np.array(net['gamma'+str(l+1)])
                self.__parameters['beta'+str(l+1)]=np.array(net['beta'+str(l+1)])
        for l in range(conv_layers,conv_layers+connected_layers):
            self.__parameters['W'+str(l+1)]=np.array(net['W'+str(l+1)])
            self.__parameters['gamma'+str(l+1)]=np.array(net['gamma'+str(l+1)])
            self.__parameters['beta'+str(l+1)]=np.array(net['beta'+str(l+1)])
        net.close()

    def modelSummary(self):
        '''
        Description:
        This method allows the user to view a summary of the model they have loaded in.

        Inputs:
        none

        Outputs:
        Prints a summary of each of the layers in the network, given self.__parameters has been appropriately loaded.
        '''
        try:
            parameters=self.__parameters
        except AttributeError:
            try:
                parameters,v,s=initialize(self.__input_shape,self.__hp)
            except AttributeError:
                raise AttributeError('Your network does not have an input layer. Please create an input layer and try again.')
        if self.__hp['cost_function']==None:
            raise AttributeError('Your network does not an output layer. Please create an output layer and try again.')
        total_trainable_params=0
        for i in range(self.__hp['conv_layers']):
            print('--------------------------------------------------')
            print('Layer name: '+str(self.__hp['layer_names'][i]))
            if self.__hp['conv_layer_types'][i]=='conv':
                print('Layer type: \'conv\'/Conv2D')
                print('Parameters:')
                (W,gamma,beta)=(parameters['W'+str(i+1)],parameters['gamma'+str(i+1)],parameters['beta'+str(i+1)])
                print('- W'+str(i+1)+', with shape '+str(W.shape))
                print('- gamma'+str(i+1)+', with shape '+str(gamma.shape))
                print('- beta'+str(i+1)+', with shape '+str(beta.shape))

                trainable_params=(W.shape[0]*W.shape[1]*W.shape[2]*W.shape[3])+(gamma.shape[0]*gamma.shape[1]*gamma.shape[2])+(beta.shape[0]*beta.shape[1]*beta.shape[2])
                total_trainable_params+=trainable_params

                print('Trainable parameters in this layer: '+str(trainable_params))
            else:
                print('Layer type: \'maxpool\'/Maxpool')
                print('Parameters:\n- None')
                print('Total trainable parameters: 0')
            print('Activation function: '+str(self.__hp['activations'][i]))
        n_H_prev=gamma.shape[0]*gamma.shape[1]*gamma.shape[2]
        for i in range(self.__hp['connected_layers']):
            print('--------------------------------------------------')
            print('Layer name: '+str(self.__hp['layer_names'][self.__hp['conv_layers']+i]))
            print('Hidden units: '+str(self.__hp['connected_layer_dims'][i]))
            print('Parameters:')
            (W,gamma,beta)=(parameters['W'+str(i+self.__hp['conv_layers']+1)],parameters['gamma'+str(i+self.__hp['conv_layers']+1)],parameters['beta'+str(i+self.__hp['conv_layers']+1)])
            print('- W'+str(self.__hp['conv_layers']+i+1)+', with shape '+str(W.shape))
            print('- gamma'+str(self.__hp['conv_layers']+i+1)+', with shape '+str(gamma.shape))
            print('- beta'+str(self.__hp['conv_layers']+i+1)+', with shape '+str(beta.shape))

            trainable_params=n_H_prev*self.__hp['connected_layer_dims'][i]
            n_H_prev=self.__hp['connected_layer_dims'][i]
            total_trainable_params+=trainable_params

            print('Trainable parameters in this layer: '+str(trainable_params))
            print('Activation function: '+str(self.__hp['activations'][self.__hp['conv_layers']+i]))
        print('--------------------------------------------------')
        print('Total model layers: '+str(self.__hp['conv_layers']+self.__hp['connected_layers']))
        print('Total trainable parameters: '+str(total_trainable_params))
        print('Loss function: '+str(self.__hp['cost_function']))
        print('Optimizer: Adam Optimizer with momentum parameter and RMSprop parameter '+str(self.__hp['beta1'])+' and '+str(self.__hp['beta2'])+', respectively.')
        print('--------------------------------------------------')

    def __adjustHpDataTypes(self):
      '''
      Description:
      This private method ensures that each of the hyperparameters stored in self.__hp are of the correct dtype

      Inputs:
      none

      Outputs:
      Updates self.__hp with the correct datatypes
      '''
      self.__hp['padding']=np.array(self.__hp['padding'],dtype=np.uint8)
      self.__hp['stride']=np.array(self.__hp['stride'],dtype=np.uint8)
      self.__hp['channels']=np.array(self.__hp['channels'],dtype=np.uint16)
      self.__hp['f_values']=np.array(self.__hp['f_values'],dtype=np.uint8)
      self.__hp['connected_layer_dims']=np.array(self.__hp['connected_layer_dims'],dtype=np.uint16)
      self.__hp['batch_size']=int(self.__hp['batch_size'])
      self.__hp['epochs']=int(self.__hp['epochs'])
      self.__hp['learning_rate']=float(self.__hp['learning_rate'])
      self.__hp['beta1']=float(self.__hp['beta1'])
      self.__hp['beta2']=float(self.__hp['beta2'])

    def compile(self,parameters=False):
        '''
        Description:
        This method allows the user to train their network by optimizing the selected cost function using the
        Adam optimizer with the chosen beta1 and beta2 values.

        Inputs:
        none

        Outputs:
        Trains the values stored in self.__parameters in accordance with self.__hp. Updates other class attributes accordingly.
        '''
        try:
            X_tr=self.__X_train
            Y_tr=self.__Y_train
        except AttributeError:
            raise AttributeError('You have not loaded in a training set. Please load in a training set and try again.')
        try:
            X_te=self.__X_test
            Y_te=self.__Y_test
        except AttributeError:
            raise AttributeError('You have not loaded in a test set. Please load in a test set and try again.')
        self.__adjustHpDataTypes()
        if parameters:
            self.__normalize_params,self.__parameters,self.__cache_list=network(X_tr,Y_tr,X_te,Y_te,self.__hp,parameters=self.__parameters)
        else:
            self.__normalize_params,self.__parameters,self.__cache_list=network(X_tr,Y_tr,X_te,Y_te,self.__hp)
