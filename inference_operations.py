import numpy as np
from operations import flatten,zero_pad

def to_inference_parameters(params,cache_list,hparams,epsilon=1e-8):
    '''
    Description:
    Converts the parameters in params to an inference-ready counterpart

    Inputs:
    params - the parameters to adjust
    cache_list - the cache list containing the values necessary to adjust the parameters in params appropriately
    hparams - hyperparameters of the network

    Outputs:
    inf_params - a dictionary containing the inference-ready variations of the parameters in params
    '''
    inf_params={}
    (conv_layers,conv_layer_types,connected_layers)=(hparams['conv_layers'],hparams['conv_layer_types'],hparams['connected_layers'])
    for l in range(conv_layers):
        if conv_layer_types[l]=='conv':
            (conv_cache,batch_norm_cache,activation_cache)=cache_list[l]
            (gamma,Zhat,mu,Z,var)=batch_norm_cache
            inf_params['W'+str(l+1)]=params['W'+str(l+1)]
            inf_params['gamma'+str(l+1)]=params['gamma'+str(l+1)]/np.sqrt(var+epsilon)
            inf_params['beta'+str(l+1)]=params['beta'+str(l+1)]-mu*(params['gamma'+str(l+1)]/np.sqrt(var+epsilon))
    for l in range(conv_layers,conv_layers+connected_layers):
        (linear_cache,batch_norm_cache,activation_cache)=cache_list[l+1]
        (gamma,Zhat,mu,Z,var)=batch_norm_cache
        inf_params['W'+str(l+1)]=params['W'+str(l+1)]
        inf_params['gamma'+str(l+1)]=params['gamma'+str(l+1)]/np.sqrt(var+epsilon)
        inf_params['beta'+str(l+1)]=params['beta'+str(l+1)]-mu*(params['gamma'+str(l+1)]/np.sqrt(var+epsilon))
    return inf_params

def linear_inf(A_prev,W):
    '''
    Description:
    The inference-ready variation of linear_forward

    Inputs:
    A_prev - previous activation
    W - relevant parameter

    Outputs:
    - W dotted with A_prev
    '''
    return np.dot(W,A_prev)

def activation_inf(Z,activation):
    '''
    Description:
    The inference ready variation of activation_forward

    Inputs:
    Z - value to be activated
    activation - activation function to be applied

    Outputs:
    - A_prev with the activation activation applied to it
    '''
    if activation=='relu':
        return np.where(Z<0,0,Z)
    elif activation=='sigmoid':
        return 1/(1+np.exp(-Z))
    elif activation=='softmax':
        return np.exp(Z)/np.sum(np.exp(Z),axis=0,keepdims=True)

def convolve_inf(A_prev,W,stride,pad):
    '''
    Description:
    The inference-ready variation of convolve_forward

    Inputs:
    A_prev - activation from previous layer, shape (m,n_H_prev,n_W_prev,n_C_prev)
    W - filters of current layer, shape (f,f,n_C_prev,n_C)
    stride - the stride of the convolution
    pad - the padding to be applied to A_prev

    Outputs:
    - A_prev, convolved with W in accordance with stride and pad
    '''
    (m,n_H_prev,n_W_prev,n_C_prev)=A_prev.shape
    (f,f,n_C_prev,n_C)=W.shape
    n_H=((n_H_prev+2*pad-f)//stride)+1
    n_W=((n_W_prev+2*pad-f)//stride)+1
    Z=np.zeros((m,n_H,n_W,n_C))
    A_prev_padded=zero_pad(A_prev,pad)
    for h in range(n_H):
        vert_start=stride*h
        vert_end=stride*h+f
        for w in range(n_W):
            horiz_start=stride*w
            horiz_end=stride*w+f
            for c in range(n_C):
                Z[:,h,w,c]=np.sum(W[:,:,:,c]*A_prev_padded[:,vert_start:vert_end,horiz_start:horiz_end,:],axis=(1,2,3))
    return Z

def batch_norm_inf(Z,gamma,beta,fc=False,epsilon=1e-8):
    '''
    Description:
    The inference-ready variation of batch_norm_forward

    Inputs:
    Z - the activations to be batch normalized
    gamma - the mutliplicative parameter of this layer's batch normalization
    beta - the additive parameter of this layer's batch normalization
    fc - dictates whether or not this process is occurring in a fully connected layer

    Outputs:
    - Z, batch normalized
    '''
    return gamma*Z+beta

def maxpool_inf(A_prev,f,stride,pad):
    '''
    Description:
    The inference-ready variation of maxpool_forward

    Inputs:
    A_prev - activation from previous layer, shape (m,n_H_prev,n_W_prev,n_C_prev)
    f - regions from which maximum value will be selected is of shape (f,f)
    stride - the stride with which we adjust the position of our region

    Outputs:
    Z - A_prev after max pooling has been applied in accordance with f, stride, and pad
    '''
    (m,n_H_prev,n_W_prev,n_C)=A_prev.shape
    n_H=((n_H_prev+2*pad-f)//stride)+1
    n_W=((n_W_prev+2*pad-f)//stride)+1
    A_prev_padded=zero_pad(A_prev,pad)
    Z=np.zeros((m,n_H,n_W,n_C))
    for h in range(n_H):
        vert_start=stride*h
        vert_end=stride*h+f
        for w in range(n_W):
            horiz_start=stride*w
            horiz_end=stride*w+f
            for c in range(n_C):
                Z[:,h,w,c]=np.max(A_prev[:,vert_start:vert_end,horiz_start:horiz_end,c],axis=(1,2))
    return Z

def infer(X,parameters,hp):
    '''
    Description:
    The inference-ready variation of propagate_forward (seen below)

    Inputs:
    X - the input for which we want a corresponding prediction (already normalized)
    parameters - the parameters with which our network will make a prediction
    hp - the network's hyperparameters

    Outputs:
    A_prev - the prediction our network makes on X
    '''
    A_prev=X
    for l in range(hp['conv_layers']):
        (pad,stride,f,activation)=(hp['padding'][l],hp['stride'][l],hp['f_values'][l],hp['activations'][l])
        if hp['conv_layer_types'][l]=='conv':
            (W,gamma,beta)=(parameters['W'+str(l+1)],parameters['gamma'+str(l+1)],parameters['beta'+str(l+1)])
            Z_conv=convolve_inf(A_prev,W,stride,pad)
            Z_bn=batch_norm_inf(Z_conv,gamma,beta)
            A_prev=activation_inf(Z_bn,activation)
        else:
            Z=maxpool_inf(A_prev,f,stride,pad)
            A_prev=activation_inf(Z,activation)
    A_prev=flatten(A_prev)
    conv_layers=hp['conv_layers']
    for l in range(hp['connected_layers']):
        activation=hp['activations'][conv_layers+l]
        (W,gamma,beta)=(parameters['W'+str(conv_layers+l+1)],parameters['gamma'+str(conv_layers+l+1)],parameters['beta'+str(conv_layers+l+1)])
        Z=linear_inf(A_prev,W)
        Z_bn=batch_norm_inf(Z,gamma,beta,fc=True)
        A_prev=activation_inf(Z_bn,activation)
    return A_prev

def top_two_accuracy(A_pred,Y):
    '''
    Description:
    This method yields the top-2 accuracy of the predictions A_pred, given the true labels Y

    Inputs:
    A_pred - the predictions whose accuracy we wish to evaluate
    Y - ground truth labels

    Outputs:
    - top-2 accuracy of A_pred given Y
    '''
    assert(A_pred.shape[1]==Y.shape[1])
    
    def find_penultimate_max(X,max_remove,dim):
        for i in range(dim):
            X[max_remove[i],i]=-np.inf
        penultimate_max=np.argmax(X,axis=0).reshape(1,dim)
        return penultimate_max

    dim=Y.shape[1]
    max1=np.argmax(A_pred,axis=0)
    max2=find_penultimate_max(A_pred,max1,dim)
    max_all=np.concatenate((max1.reshape(1,dim),max2),axis=0)
    max_actual=np.argmax(Y,axis=0).reshape(1,dim)
    top_two=0
    for i in range(dim):
        if int(max_actual[0,i]) in list(max_all[:,i]):
            top_two+=1
    return 100*(top_two/dim)
