import numpy as np
from dataset_operations import *
from initialization_operations import *
from network_operations import *
from compiler import *
from inference_operations import *

class FruitNet():

    def __init__(self):

        #instantiate hyperparameters
        self.__hp={'learning_rate':0,
                   'epochs':0,
                   'mini_batch_size':0,
                   'initialization':None,
                   'strips':0,
                   'subnet_names':[],
                   'subnet_hidden':[],
                   'subnet_activations':[],
                   'supnet_names':[],
                   'supnet_hidden':[],
                   'supnet_activations':[]}
    
    def __raw_shuffle__(self,X,Y):

        #permute [1,2,...,m]
        permutation=np.random.permutation(X.shape[0])

        #shuffle X and Y in accordance with the above permutation
        shuffled_X=X[permutation,:,:,:]
        shuffled_Y=Y[:,permutation]

        return shuffled_X,shuffled_Y

    def loadData(self,filename,features,labels,split=0.05):

        #retrieve data
        try:
            training_set=h5py.File(filename,'r')
            X=training_set[features][:]
            Y=training_set[labels][:]
            training_set.close()
        except:
            raise NameError('you\'ve entered an incorrect h5py filename, feature dataset name, or label dataset name, when loading in your data.')

        #confirm validity of X dataset shape
        assert len(X.shape)==4
        assert X.shape[3]==1

        #confirm validity of Y dataset shape
        assert len(Y.shape)==2
        assert X.shape[0]==Y.shape[1]

        #shuffle datasets (maintaining correspondence)
        shuffled_X,shuffled_Y=self.__raw_shuffle__(X,Y)

        #establish cross-validation test set (and, therefore, training set) sizes
        m=X.shape[0]
        self.__test_quant=int(split*m)
        self.__train_quant=m-self.__test_quant

        #create cross-validation test set and training set using the above sizes
        self.__X_train=shuffled_X[self.__test_quant:self.__test_quant+self.__train_quant,:,:,:]
        self.__Y_train=shuffled_Y[:,self.__test_quant:self.__test_quant+self.__train_quant]
        self.__X_test=shuffled_X[:self.__test_quant,:,:,:]
        self.__Y_test=shuffled_Y[:,:self.__test_quant]

    def __retrieve_single_split_example__(self,strips,view_data=True):

        #select a random sample number
        try:
            sample=np.random.randint(0,self.__test_quant)
        except:
            raise UnboundLocalError('self.__test_quant has referenced before assignment, meaning you have not yet loaded in any data. please load in data and try again.')

        #select the sample from the test set and split into regions
        X=np.zeros((1,self.__X_test.shape[1],self.__X_test.shape[2],1))
        Y=np.zeros((self.__Y_test.shape[0],1))
        X[0,:,:,0]=self.__X_test[sample,:,:,0]
        Y[:,0]=self.__Y_test[:,sample]
        [(Xt,Yt)]=reshuffle_split_mini_batches(X,Y,{'mini_batch_size':1,'strips':strips})

        #check why the function has been called and return accordingly
        if view_data:
            return X,Xt,Y
        else:
            return (Xt,Yt)


    def viewData(self,strips):

        #confirm the validity of the number of strips
        assert type(strips)==int
        assert strips>1

        #retrieve (split) item of data (from test set) corresponding to random_sample
        X,Xt,Y=self.__retrieve_single_split_example__(strips)

        #plot the sample, as well as the strips into which we've split it, and the corresponding label
        fig=plt.figure()
        axes=fig.subplots(strips+1,1)
        axes[0].imshow(X[0,:,:,0])
        for i in range(strips):
            axes[i+1].imshow(Xt[i].reshape(X.shape[1]//strips,X.shape[2]))
        print('The label corresponding to the above item of data (read from left to right) is given by:\n'+str(Y.reshape(Y.shape[0],)))
    
    def addLayer(self,name,n_H,activation,net='sub'):
        
        #confirm the validity of name, n_H, activation, and supnet
        assert type(name)==str
        assert type(n_H)==int
        assert n_H>0
        assert activation in [relu,sigmoid]
        assert type(net)==str

        #append the hyperparameters in accordance with the function parameters
        if net=='sup':
            assert name not in self.__hp['supnet_names']
            self.__hp['supnet_names'].append(name)
            self.__hp['supnet_hidden'].append(n_H)
            self.__hp['supnet_activations'].append(activation)
        elif net=='sub':
            assert name not in self.__hp['subnet_names']
            self.__hp['subnet_names'].append(name)
            self.__hp['subnet_hidden'].append(n_H)
            self.__hp['subnet_activations'].append(activation)
        else:
            raise ValueError('you have not selected a valid network to add a layer to. please re-enter the net string and try again.')
    
    def __decoy_parameters__(self):

        #extract an example from the test set to preserve computational efficiency
        single_example_mini_batch=self.__retrieve_single_split_example__(self.__hp['strips'],view_data=False)

        #generate and return relevant parameters (NOT as class attributes, though)
        return build_nets(single_example_mini_batch,self.__hp)

    def modelSummary(self):

        #confirm that enough hyperparameters have been specified in order for a summary to be generated
        assert self.__hp['strips']>0

        #retrieve the parameters whose summary we wish to produce
        try:
            subnet=self.__subnet_parameters['subnet1']
            supnet=self.__supnet_parameters
        except:
            subnets,supnet=self.__decoy_parameters__()
            subnet=subnets['subnet1']
        
        #produce model summary header
        print('==================================================')
        print('NETWORK ARCHITECTURE SUMMARY')
        print('==================================================\n\n')

        #produce subnet summary
        print('SUBNET SUMMARY')
        print('--------------------------------------------------')
        for i in range(len(subnet)//3+1):
            if i==0:
                print('Input Layer:')
                print('- Name: N/A')
                print('- No. of Input Nodes: '+str(subnet['W1'].shape[1]))
                print('- Activation Function: N/A')
                print('--------------------------------------------------')
            else:
                if i==len(subnet)//3:
                    print('Output Layer:')
                    print('- Name: N/A')
                    print('- No. of Output Nodes: '+str(subnet['W'+str(i)].shape[0]))
                    print('Activation Function: softmax')
                    print('--------------------------------------------------')
                else:
                    print('Hidden Layer '+str(i)+':')
                    print('- Name: '+str(self.__hp['subnet_names'][i-1]))
                    print('- No. of Hidden Nodes: '+str(subnet['W'+str(i)].shape[0]))
                    if self.__hp['subnet_activations'][i-1]==relu:
                        print('- Activation Function: relu')
                    else:
                        print('- Activation Function: sigmoid')
                    print('--------------------------------------------------')
        print('\n')

        #produce supnet summary
        print('SUPNET SUMMARY')
        print('--------------------------------------------------')
        for i in range(len(supnet)//3+1):
            if i==0:
                print('Input Layer:')
                print('- Name: N/A')
                print('- No. of Input Nodes: '+str(supnet['W1'].shape[1]))
                print('- Activation Function: N/A')
                print('--------------------------------------------------')
            else:
                if i==len(supnet)//3:
                    print('Output Layer:')
                    print('- Name: N/A')
                    print('- No. of Output Nodes: '+str(supnet['W'+str(i)].shape[0]))
                    print('Activation Function: softmax')
                    print('--------------------------------------------------')
                else:
                    print('Hidden Layer '+str(i)+':')
                    print('- Name: '+str(self.__hp['supnet_names'][i-1]))
                    print('- No. of Hidden Nodes: '+str(supnet['W'+str(i)].shape[0]))
                    if self.__hp['supnet_activations'][i-1]==relu:
                        print('- Activation Function: relu')
                    else:
                        print('- Activation Function: sigmoid')
                    print('--------------------------------------------------')
        print('\n')

    def adjustLearningRate(self,learning_rate):

        #confirm the validity of the proposed learning_rate
        assert type(learning_rate)==float
        assert learning_rate>0

        #inform the user of the current learning rate, as well as the updated learning rate
        print('The current learning rate is '+str(self.__hp['learning_rate'])+'.')
        self.__hp['learning_rate']=learning_rate
        print('And the new learning rate is '+str(self.__hp['learning_rate'])+'.')

    def adjustEpochs(self,epochs):

        #confirm the validity of the proposed number of epochs
        assert type(epochs)==int
        assert epochs>0

        #inform the user of the current number of epochs, as well as the updated number of epochs
        print('The current number of epochs is '+str(self.__hp['epochs'])+'.')
        self.__hp['epochs']=epochs
        print('And the new number of epochs is '+str(self.__hp['epochs'])+'.')

    def adjustBatchSize(self,size):

        #confirm the validity of the proposed number of epochs
        assert type(size)==int
        assert size>0

        #inform the user of the current mini batch size, as well as the updated mini batch size
        print('The current mini batch size is '+str(self.__hp['mini_batch_size'])+'.')
        self.__hp['mini_batch_size']=size
        print('And the new mini batch size is '+str(self.__hp['mini_batch_size'])+'.')
    
    def adjustInitialization(self,init):

        #confirm the validity of the proposed intialization technique
        assert init in ['None','He','Xavier','Other']

        #inform the user of the current intialization technique, as well as the updated initialization technique
        print('The current intialization tehcnique is '+str(self.__hp['initialization'])+'.')
        self.__hp['initialization']=init
        print('And the new initialization technique is '+str(self.__hp['initialization'])+'.')

    def adjustStrips(self,strips):

        #confirm the validity of the proposed numbwe of strips
        assert type(strips)==int
        assert strips>1

        #inform the user of the current number of strips, as well as the updated number of strips
        print('The current number of strips into which training examples are split is '+str(self.__hp['strips'])+'.')
        self.__hp['strips']=strips
        print('And the new number of strips into which training examples are split is '+str(self.__hp['strips'])+'.')
    
    def compile(self):

        #check the validity of the relevant hyperparameters before initiating training
        assert self.__hp['learning_rate']>0
        assert self.__hp['epochs']>0
        assert self.__hp['mini_batch_size']>0
        assert self.__hp['strips']>1

        #expand the activation function hyperparameters for both the sub- and the supnet
        self.__hp['subnet_activations'].append(softmax)
        self.__hp['supnet_activations'].append(softmax)

        #exectute the relevant training
        try:
            self.__subnet_parameters,self.__supnet_parameters,inf_caches=train(self.__X_train,self.__Y_train,self.__hp)
        except:
            raise UnboundLocalError('self.__X_train and self.__Y_train referenced before assignment, meaning you have not loaded in any data. please load in a dataset and try again.')

        #instantiate the networks inference parameters
        self.__subnet_inf_parameters={}
        for i in range(len(self.__subnet_parameters)):
            self.__subnet_inf_parameters['subnet'+str(i+1)]=to_inf_params(self.__subnet_parameters['subnet'+str(i+1)],inf_caches['subnet'+str(i+1)])
        self.__supnet_inf_parameters=to_inf_params(self.__supnet_parameters,inf_caches['supnet'])

        #check the network's performance on the test set
        print('The network\'s performance on the test set yields an error of roughly '+str(perc_error(self.__X_test,self.__Y_test,self.__subnet_inf_parameters,self.__supnet_inf_parameters,self.__hp)))

        #rectify the activation function hyperparameters for both the sub- and the supnet once training is complete
        self.__hp['subnet_activations'].pop()
        self.__hp['supnet_activations'].pop()

    def saveModel(self):
        pass
