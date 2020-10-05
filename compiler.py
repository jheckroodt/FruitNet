import numpy as np
from operations import *
from initialization import *
from inference_operations import *

def propagate_forward(X,parameters,hp):
    '''
    Description:
    This method propagates X forward along a network given by hp, using parameters

    Inputs:
    X - the set of data to be propagated forward
    parameters - the parameters with which our network will propagate X forward
    hp - the hyperparameters in accordance with which X will be propagated forward

    Outputs:
    cache_list - a list of all the values caches as the forward pass occurred
    A_prev - the output activation of our network, given X
    '''
    cache_list=[]
    A_prev=X
    for l in range(hp['conv_layers']):
        (pad,stride,f,activation)=(hp['padding'][l],hp['stride'][l],hp['f_values'][l],hp['activations'][l])
        if hp['conv_layer_types'][l]=='conv':
            (W,gamma,beta)=(parameters['W'+str(l+1)],parameters['gamma'+str(l+1)],parameters['beta'+str(l+1)])
            conv_cache,Z_conv=convolve_forward(A_prev,W,stride,pad)
            batch_norm_cache,Z_bn=batch_norm_forward(Z_conv,gamma,beta)
            activation_cache,A_prev=activation_forward(Z_bn,activation)
            cache_list.append((conv_cache,batch_norm_cache,activation_cache))
        else:
            maxpool_cache,Z=maxpool_forward(A_prev,f,stride,pad)
            activation_cache,A_prev=activation_forward(Z,activation)
            cache_list.append((maxpool_cache,activation_cache))
    cache_list.append(A_prev.shape)
    A_prev=flatten(A_prev)
    conv_layers=hp['conv_layers']
    for l in range(hp['connected_layers']):
        activation=hp['activations'][conv_layers+l]
        (W,gamma,beta)=(parameters['W'+str(conv_layers+l+1)],parameters['gamma'+str(conv_layers+l+1)],parameters['beta'+str(conv_layers+l+1)])
        linear_cache,Z=linear_forward(A_prev,W)
        batch_norm_cache,Z_bn=batch_norm_forward(Z,gamma,beta,fc=True)
        activation_cache,A_prev=activation_forward(Z_bn,activation)
        cache_list.append((linear_cache,batch_norm_cache,activation_cache))
    return cache_list,A_prev

def compute_cost_forward(AL,Y,cost_function):
    '''
    Description:
    This method computes the cost of a particular prediction, given a speicified cost function

    Inputs:
    AL - prediction given the input labels corresponding to the output labels Y
    Y - target labels
    cost_function - the cost function used to evaluate the cost of the given prediction

    Outputs:
    - the cost of the predictions made
    '''
    m=Y.shape[1]
    if cost_function=='softmax_cost':
        return np.sum(-np.sum(Y*np.log(AL),axis=0,keepdims=True))/m
    elif cost_function=='mse_cost':
        return np.squeeze(np.sum(np.sum((Y[0:6,:]-AL[0:6,:])**2,axis=0,keepdims=True)+np.sum(5*(Y[6:10,:]-AL[6:10,:])**2,axis=0,keepdims=True),axis=1,keepdims=True)/m)

def compute_cost_backward(AL,Y,cost_function):
    '''
    Description:
    This method initializes the back propagation process

    Inputs:
    AL - output activation of the network
    Y - ground truth labels for the current mini batch
    cost_function - the cost function used to compute the cost

    Outputs:
    - gradient to be propagated backwards
    '''
    if cost_function=='softmax_cost':
        return AL-Y
    elif cost_function=='mse_cost':
        return np.concatenate((2*(AL[0:6,:]-Y[0:6,:]),10*(AL[6:10,:]-Y[6:10,:])),axis=0)

def propagate_backward(Y,AL,cache_list,hp):
    '''
    Description:
    This method performs the entire back propagation process

    Inputs:
    Y - target labels
    AL - prediction on said labels
    cache_list - cache list output from forward prop
    hp - hyperparameters governing the network

    Outputs:
    grads - the gradients of all of the parameters in the network
    '''
    m=Y.shape[1]
    grads={}
    dA=compute_cost_backward(AL,Y,hp['cost_function'])
    (conv_layers,connected_layers,activations,layer_types)=(hp['conv_layers'],hp['connected_layers'],hp['activations'],hp['conv_layer_types'])
    for l in reversed(range(connected_layers)):
        (linear_cache,batch_norm_cache,activation_cache)=cache_list[conv_layers+l+1]
        dZtilde=activation_backward(dA,activation_cache)
        dZ,dgamma,dbeta=batch_norm_backward(dZtilde,batch_norm_cache,fc=True)
        dA,dW=linear_backward(dZ,linear_cache)
        grads['dW'+str(conv_layers+1+l)]=dW
        grads['dgamma'+str(conv_layers+1+l)]=dgamma
        grads['dbeta'+str(conv_layers+1+l)]=dbeta
    current_cache=cache_list[conv_layers]
    dA=flatten(dA,cache=current_cache)
    for l in reversed(range(conv_layers)):
        current_cache=cache_list[l]
        if layer_types[l]=='conv':
            (conv_cache,batch_norm_cache,activation_cache)=current_cache
            dZtilde=activation_backward(dA,activation_cache)
            dZ,dgamma,dbeta=batch_norm_backward(dZtilde,batch_norm_cache)
            if l==0:
                dW=convolve_backward_filters(dZ,conv_cache)
            else:
                dA,dW=convolve_backward_inputs(dZ,conv_cache)
            grads['dW'+str(l+1)]=dW
            grads['dgamma'+str(l+1)]=dgamma
            grads['dbeta'+str(l+1)]=dbeta
        else:
            (maxpool_cache,activation_cache)=current_cache
            dZ=activation_backward(dA,activation_cache)
            dA=maxpool_backward(dZ,maxpool_cache)
    return grads

def update_parameters(parameters,grads,v,s,hp,epsilon=1e-8):
    '''
    Description:
    This method updates the parameters in parameters, given the gradients in grads

    Inputs:
    parameters - the parameters whose values we want to update
    grads - the gradients we'll use to update the parameters in parameters
    v - the momentum terms used to update the relevant parameters
    s - the RMSprop terms used to update the relevant parameters
    hp - the hyperparameters corresponding to the network

    Outputs:
    parameters - the updates parameters
    v - the updated v terms
    s - the updated s terms
    '''
    (conv_layers,connected_layers,conv_layer_types,beta1,beta2,learning_rate)=(hp['conv_layers'],hp['connected_layers'],hp['conv_layer_types'],hp['beta1'],hp['beta2'],hp['learning_rate'])
    for l in range(conv_layers):
        if conv_layer_types[l]=='conv':
            v['dW'+str(l+1)]=beta1*v['dW'+str(l+1)]+(1-beta1)*grads['dW'+str(l+1)]
            v['dgamma'+str(l+1)]=beta1*v['dgamma'+str(l+1)]+(1-beta1)*grads['dgamma'+str(l+1)]
            v['dbeta'+str(l+1)]=beta1*v['dbeta'+str(l+1)]+(1-beta1)*grads['dbeta'+str(l+1)]
            s['dW'+str(l+1)]=beta2*s['dW'+str(l+1)]+(1-beta2)*(grads['dW'+str(l+1)]**2)
            s['dgamma'+str(l+1)]=beta2*s['dgamma'+str(l+1)]+(1-beta2)*(grads['dgamma'+str(l+1)]**2)
            s['dbeta'+str(l+1)]=beta2*s['dbeta'+str(l+1)]+(1-beta2)*(grads['dbeta'+str(l+1)]**2)
            parameters['W'+str(l+1)]=parameters['W'+str(l+1)]-learning_rate*(v['dW'+str(l+1)]/np.sqrt(s['dW'+str(l+1)]+epsilon))
            parameters['gamma'+str(l+1)]=parameters['gamma'+str(l+1)]-learning_rate*(v['dgamma'+str(l+1)]/np.sqrt(s['dgamma'+str(l+1)]+epsilon))
            parameters['beta'+str(l+1)]=parameters['beta'+str(l+1)]-learning_rate*(v['dbeta'+str(l+1)]/np.sqrt(s['dbeta'+str(l+1)]+epsilon))
    for l in range(conv_layers,conv_layers+connected_layers):
        v['dW'+str(l+1)]=beta1*v['dW'+str(l+1)]+(1-beta1)*grads['dW'+str(l+1)]
        v['dgamma'+str(l+1)]=beta1*v['dgamma'+str(l+1)]+(1-beta1)*grads['dgamma'+str(l+1)]
        v['dbeta'+str(l+1)]=beta1*v['dbeta'+str(l+1)]+(1-beta1)*grads['dbeta'+str(l+1)]
        s['dW'+str(l+1)]=beta2*s['dW'+str(l+1)]+(1-beta2)*(grads['dW'+str(l+1)]**2)
        s['dgamma'+str(l+1)]=beta2*s['dgamma'+str(l+1)]+(1-beta2)*(grads['dgamma'+str(l+1)]**2)
        s['dbeta'+str(l+1)]=beta2*s['dbeta'+str(l+1)]+(1-beta2)*(grads['dbeta'+str(l+1)]**2)
        parameters['W'+str(l+1)]=parameters['W'+str(l+1)]-learning_rate*(v['dW'+str(l+1)]/np.sqrt(s['dW'+str(l+1)]+epsilon))
        parameters['gamma'+str(l+1)]=parameters['gamma'+str(l+1)]-learning_rate*(v['dgamma'+str(l+1)]/np.sqrt(s['dgamma'+str(l+1)]+epsilon))
        parameters['beta'+str(l+1)]=parameters['beta'+str(l+1)]-learning_rate*(v['dbeta'+str(l+1)]/np.sqrt(s['dbeta'+str(l+1)]+epsilon))
    return parameters,v,s

def network(X,Y,X_test,Y_test,hp,parameters=None,epsilon=1e-8):
    '''
    Description:
    This method is responsible for all of the user-specified training

    Inputs:
    X - training inputs
    Y - training ground truth labels
    X_test - test set inputs
    Y_test - test set ground truth labels
    hp - hyperparameters of the network
    parameters - parameters the network has to train

    Outputs:
    normalize_params - the mean and variance of the training set
    parameters - the parameters our network has trained
    cache_list - all of the cache values generated by the final training iteration (for saving purposes)
    '''
    (mean,variance),X_normalized=normalize(X.copy())
    X_test_normalized=normalize(X_test.copy(),mu=mean,var=variance)
    mini_batches=build_mini_batches(X_normalized,Y,hp['batch_size'])
    mini_batches_test=build_mini_batches(X_test_normalized,Y_test,X_test_normalized.shape[0])
    (X_test_normalized,Y_test)=mini_batches_test[0]
    if parameters==None:
        parameters,v,s=initialize(mini_batches[0][0].shape[1:],hp,const=0.01)
    else:
        v,s=initialize(mini_batches[0][0].shape[1:],hp,return_parameters=False,const=0.01)
    for e in range(hp['epochs']):
        for t in range(len(mini_batches)):
            (Xt,Yt)=mini_batches[t]
            cache_list,A_prev=propagate_forward(Xt,parameters,hp)
            print('--------------------------------------------------')
            print('Total cost: '+str(compute_cost_forward(A_prev,Yt,hp['cost_function'])))
            print('Epochs completed: '+str(e))
            print('Training iteration number (for this epoch): '+str(t+1))
            grads=propagate_backward(Yt,A_prev,cache_list,hp)
            parameters,v,s=update_parameters(parameters,grads,v,s,hp)
        print('--------------------------------------------------')
        print('The current epoch has been concluded. First three feature vectors from final batch:\n'+str(A_prev[:,:3]))
        print('First three target labels from final batch:\n'+str(Yt[:,:3])+'\n')
        A_pred=infer(X_test_normalized,to_inference_parameters(parameters,cache_list,hp),hp)
        print('Total cost: '+str(compute_cost_forward(A_prev,Yt,hp['cost_function'])))
        print('Top two accuracy on test set: '+str(top_two_accuracy(A_pred,Y_test)))
    return (mean,variance),parameters,cache_list
