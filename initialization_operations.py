import numpy as np

def initialize_parameters(layer_dims,initialization):
    L=len(layer_dims)
    parameters={}
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
    X=split_mini_batch[0][0]
    Y=split_mini_batch[1]
    subnet_parameters={}
    if hp['subnet_hidden']==[]:
        layer_dims=[X.shape[0],Y.shape[0]]
    else:
        layer_dims=[X.shape[0]]+hp['subnet_hidden']+[Y.shape[0]]
    for i in range(hp['strips']):
        subnet_parameters['subnet'+str(i+1)]=initialize_parameters(layer_dims,hp['initialization'])
    if hp['supnet_hidden']==[]:
        layer_dims=[hp['strips']*Y.shape[0],Y.shape[0]]
    else:
        layer_dims=[hp['strips']*Y.shape[0]]+hp['supnet_hidden']+[Y.shape[0]]
    supnet_parameters=initialize_parameters(layer_dims,hp['initialization'])
    return subnet_parameters,supnet_parameters

def normalize_inputs(X,epsilon=1e-8):
    return X/255
