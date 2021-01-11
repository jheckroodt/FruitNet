import numpy as np

def initialize_parameters(layer_dims,initialization):
    '''
    Description: initializes the parameters of a fully connected network whose architecture is specified by layer_dims

    Inputs:
    - layer_dims: a list of the number of nodes in each of the layers of the network whose parameters we wish to initialize
    - initialization: the method in accordance with which our parameters are initialized

    Returns:
    - parameters: a dictionary containing all of the parameters relevant to our network
    '''

    #establish the number of layers in the network
    L=len(layer_dims)

    #instantiate our parameter dictionary
    parameters={}

    #in accordance with our preferred initialization method, initialize each of the relevant parameters
    if initialization=='None':
        for l in range(1,L):
            parameters['W'+str(l)]=0.01*np.random.rand(layer_dims[l],layer_dims[l-1])
            parameters['gamma'+str(l)]=np.ones((layer_dims[l],1))
            parameters['beta'+str(l)]=np.zeros((layer_dims[l],1))
    elif initialization=='He':
        for l in range(1,L):
            const=np.sqrt(2/layer_dims[l-1])
            parameters['W'+str(l)]=const*np.random.rand(layer_dims[l],layer_dims[l-1])
            parameters['gamma'+str(l)]=np.ones((layer_dims[l],1))
            parameters['beta'+str(l)]=np.zeros((layer_dims[l],1))
    elif initialization=='Xavier':
        for l in range(1,L):
            const=np.sqrt(1/layer_dims[l-1])
            parameters['W'+str(l)]=const*np.random.rand(layer_dims[l],layer_dims[l-1])
            parameters['gamma'+str(l)]=np.ones((layer_dims[l],1))
            parameters['beta'+str(l)]=np.zeros((layer_dims[l],1))
    elif initialization=='Other':
        for l in range(1,L):
            const=np.sqrt(2/(layer_dims[l]+layer_dims[l-1]))
            parameters['W'+str(l)]=const*np.random.rand(layer_dims[l],layer_dims[l-1])
            parameters['gamma'+str(l)]=np.ones((layer_dims[l],1))
            parameters['beta'+str(l)]=np.zeros((layer_dims[l],1))
    
    return parameters

def build_nets(split_mini_batch,hp):
    '''
    Description: initializes the parameters necessary for all of the subnets, as well as all of the supernets in our model

    Inputs:
    - split_mini_batch: a single mini batch from the dataset whose contents we wish to model (used to establish the input layer shape of our subnets)
    - hp: the hyperparameters in accordance with which the generation of the subnets and supernet takes place

    Returns:
    - subnet_parameters: a dictionary of dictionaries, each (nested) dictionary defining the parameters for a different subnet
    - supnet parameters: a dictionary defining the parameters for the supernet
    '''

    #extract a single subset from the given mini batch (these subsets are what is ultimately fed into our subnets as inputs), as well as its corresponding labels
    X=split_mini_batch[0][0]
    Y=split_mini_batch[1]

    #instantiate our subnet parameters dictionary
    subnet_parameters={}

    #define the dimensions of our subnets, in accordance with hp['subnet_hidden'], which contained the number of hidden nodes in a hidden layer, if there are any hidden layers
    if hp['subnet_hidden']==[]:
        layer_dims=[X.shape[0],Y.shape[0]]
    else:
        layer_dims=[X.shape[0]]+hp['subnet_hidden']+[Y.shape[0]]

    #in accordance with the dimensions of our subnets, define their parameters
    for i in range(hp['strips']):
        subnet_parameters['subnet'+str(i+1)]=initialize_parameters(layer_dims,hp['initialization'])

    #repeat the above process for the supernet
    if hp['supnet_hidden']==[]:
        layer_dims=[hp['strips']*Y.shape[0],Y.shape[0]]
    else:
        layer_dims=[hp['strips']*Y.shape[0]]+hp['supnet_hidden']+[Y.shape[0]]
    supnet_parameters=initialize_parameters(layer_dims,hp['initialization'])

    return subnet_parameters,supnet_parameters

def normalize_inputs(X):
    '''
    Description: normalize the dataset X by dividing each of its elements by 255 (since the range of pixel intensities of an image is given by [0,255])

    Inputs:
    - X: the dataset to be noramlized

    Returns:
    - X divided by 255
    '''
    
    return X/255
