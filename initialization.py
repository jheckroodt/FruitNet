import numpy as np

def build_mini_batches(X,Y,size):
    '''
    Description:
    This method takes in a set of data and divides it into mini batches

    Inputs:
    X - input labels to build mini batches out ot
    Y - output labels to build corresponding mini batches out of
    size - size of the mini batches to be build

    Outputs:
    mini_batches - a python list containing tuples, each of which is a mini batch
    '''
    (m,n_H,n_W,n_C)=X.shape
    (c,m)=Y.shape
    mini_batches=[]
    permutation=np.random.permutation(m)
    X_shuffled=X[permutation,:,:,:]
    Y_shuffled=Y[:,permutation]
    complete_batches=m//size
    for t in range(complete_batches):
        X_mini_batch=X_shuffled[size*t:size*(t+1),:,:,:]
        Y_mini_batch=Y_shuffled[:,size*t:size*(t+1)]
        mini_batches.append((X_mini_batch,Y_mini_batch))
    if (m%size!=0):
        X_mini_batch=X_shuffled[size*complete_batches:m-1,:,:,:]
        Y_mini_batch=Y_shuffled[:,size*complete_batches:m-1]
        mini_batches.append((X_mini_batch,Y_mini_batch))
    return mini_batches

def normalize(X,mu=False,var=False,epsilon=1e-8):
    '''
    Description:
    This method normalizes a set of data

    Inputs:
    X - set to be normalized
    epsilon - avoids division by 0
    mu - predetermined mean
    var - predetermined variance

    Outputs:
    mean - (if mu and var are bools) the mean of the elements in the set of data to be normalized
    variance - (if neither mu nor var are bools) the variance of the elements in the set of data to be normalized
    normalized X - the data set X, normalized
    '''
    if type(mu)==type(var)==bool:
        mean=np.mean(X,axis=0,keepdims=True)
        variance=np.var(X,axis=0,keepdims=True)
        return (mean,variance),(X-mean)/np.sqrt(variance+epsilon)
    else:
        return (X-mu)/np.sqrt(var+epsilon)

def initialize(shape,hp,return_parameters=True,const=0.05):
    '''
    Description:
    This method initializes the parameters our network will use, along with the momentum
    and RMSprop dictionaries necessary for Adam optimization

    Inputs:
    shape - shape of the inputs to the network
    hp - hyperparameters in accordance with which our parameters will be
         initialized
    return_parameters - dictates whether or not the parameters generated here are returned

    Outputs:
    parameters - (if return_parameters is True) the parameters our network will use
    v - momentum dictionary for Adam optimization
    s - RMSprop dictionary for Adam optimization
    '''
    parameters={}
    v={}
    s={}
    (n_H,n_H,n_C)=shape
    for l in range(hp['conv_layers']):
        n_H=int(1+(n_H+2*hp['padding'][l]-hp['f_values'][l])/hp['stride'][l])
        n_C_prev=n_C
        n_C=hp['channels'][l]
        if hp['conv_layer_types'][l]=='conv':
            v['dW'+str(l+1)]=np.zeros((hp['f_values'][l],hp['f_values'][l],n_C_prev,n_C))
            s['dW'+str(l+1)]=np.zeros((hp['f_values'][l],hp['f_values'][l],n_C_prev,n_C))
            v['dgamma'+str(l+1)]=np.zeros((1,n_H,n_H,n_C))
            s['dgamma'+str(l+1)]=np.zeros((1,n_H,n_H,n_C))
            v['dbeta'+str(l+1)]=np.zeros((1,n_H,n_H,n_C))
            s['dbeta'+str(l+1)]=np.zeros((1,n_H,n_H,n_C))
            parameters['W'+str(l+1)]=const*np.random.randn(hp['f_values'][l],hp['f_values'][l],n_C_prev,n_C)
            parameters['gamma'+str(l+1)]=np.ones((1,n_H,n_H,n_C))
            parameters['beta'+str(l+1)]=np.zeros((1,n_H,n_H,n_C))
    for l in range(hp['connected_layers']):
        v['dgamma'+str(hp['conv_layers']+l+1)]=np.zeros((hp['connected_layer_dims'][l],1))
        s['dgamma'+str(hp['conv_layers']+l+1)]=np.zeros((hp['connected_layer_dims'][l],1))
        v['dbeta'+str(hp['conv_layers']+l+1)]=np.zeros((hp['connected_layer_dims'][l],1))
        s['dbeta'+str(hp['conv_layers']+l+1)]=np.zeros((hp['connected_layer_dims'][l],1))
        if l==0:
            v['dW'+str(hp['conv_layers']+l+1)]=np.zeros((hp['connected_layer_dims'][l],n_H*n_H*hp['channels'][-1]))
            s['dW'+str(hp['conv_layers']+l+1)]=np.zeros((hp['connected_layer_dims'][l],n_H*n_H*hp['channels'][-1]))
            parameters['W'+str(hp['conv_layers']+l+1)]=const*np.random.randn(hp['connected_layer_dims'][l],n_H*n_H*hp['channels'][-1])
        else:
            v['dW'+str(hp['conv_layers']+l+1)]=np.zeros((hp['connected_layer_dims'][l],hp['connected_layer_dims'][l-1]))
            s['dW'+str(hp['conv_layers']+l+1)]=np.zeros((hp['connected_layer_dims'][l],hp['connected_layer_dims'][l-1]))
            parameters['W'+str(hp['conv_layers']+l+1)]=const*np.random.randn(hp['connected_layer_dims'][l],hp['connected_layer_dims'][l-1])
        parameters['gamma'+str(hp['conv_layers']+l+1)]=np.ones((hp['connected_layer_dims'][l],1))
        parameters['beta'+str(hp['conv_layers']+l+1)]=np.zeros((hp['connected_layer_dims'][l],1))
    if return_parameters:
        return parameters,v,s
    else:
        return v,s
