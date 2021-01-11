import numpy as np

def sigmoid(z,backward=False):
    '''
    Description: the sigmoid activation function

    Inputs:
    - z: the np.array() object to which we wish to apply the sigmoid activation function
    - backward: dictates whether or not the function is being called during the forward pass or the backward pass

    Returns:
    - the np.array() object resulting from the sigmoid activation function being applied to z
    '''

    if backward:
        return (1/(1+np.exp(-z)))*(1-(1/(1+np.exp(-z))))
    else:
        return 1/(1+np.exp(-z))

def relu(z,backward=False):
    '''
    Description: the ReLU activation function

    Inputs:
    - z: the np.array() object to which we wish to apply the ReLU activation function
    - backward: dictates whether or not the function is being called during the forward pass or the backward pass

    Returns:
    - the np.array() object resulting from the ReLU activation function being applied to z
    '''
    
    if backward:
        return np.where(z<0,0,1)
    else:
        return np.where(z<0,0,z)

def softmax(z):
    '''
    Description: the softmax activation function (this activation function may only be, and is necessarily, applied to the final layer of both the subnet and the supernet)

    Inputs:
    - z: the np.array() object to which we wish to apply the softmax activation function

    Returns:
    - the np.array() object resulting from the softmax activation function being applied to z
    '''
    
    magnitude=np.sum(np.exp(z),axis=0,keepdims=True)
    return np.exp(z)/magnitude

def concat(z,classes,m=None):
    '''
    Description: concatenates the activations produced by the subnets to form an input for the supernet.

    Inputs:
    - z: a list of activations produced by the subnets, to be concatenated into a supernet input
    - classes: the number of output nodes in a given subnet
    - m: the size of a mini batch on which we are training the subnets

    Returns:
    - new_z: the elements of z, concatenated into an appropriately shaped input to the supernet

    Notes: this function was also used in a previous variation of the project, where subnets operated slightly differently, hence m may be regarded as dictating whether or not this function is being called in the forward pass or the backward pass
    '''

    #check the size of the mini batch (if m==None, then the function is being called during the backward pass, a possibility which is omitted in this iteration of the project, as mentioned by the note above)
    if m==None:
        subnet_dA=[]
        for i in range(z.shape[0]//classes):
            subnet_dA.append(z[i*classes:(i+1)*classes,:])
        return subnet_dA
    else:

        #instantiate our return variable
        new_z=np.zeros((len(z)*classes,m))

        #arrange the elements of z into our return variable
        for i in range(len(z)):
            new_z[i*classes:(i+1)*classes,:]=z[i]

        return new_z

def propagate_forward(X,parameters,activations,epsilon=1e-8):
    '''
    Description: propagates the mini batch X forward across a network with parameters 'parameters'

    Inputs:
    - X: mini batch to be propagated forward
    - parameters: the parameters of the network along which X is propagated
    - activations: the activation function of each of the layers in the network with an activation function
    - epsilon: prevents division by 0

    Returns:
    - cache_list: the list of values caches for the purposes of performing the backward pass
    - A_prev: the final layer activation produced by the network in question
    '''

    #assign X to A_prev so that we may utilize A_prev in our eventual loop through the network layers
    A_prev=X

    #instiate the list in which the caches values from each layer will be stored
    cache_list=[]

    #identify the number of layers in the network
    L=len(parameters)//3

    #loop through the layers in the network
    for l in range(L):

        #apply the weights to the input to the layer
        Z=np.dot(parameters['W'+str(l+1)],A_prev)

        #apply batch normalization to the resulting np.array() object
        mu=np.mean(Z,axis=1,keepdims=True)
        std=np.var(Z,axis=1,keepdims=True)
        Zhat=(Z-mu)/np.sqrt(std+epsilon)
        Ztilde=parameters['gamma'+str(l+1)]*Zhat+parameters['beta'+str(l+1)]

        #cache the relevant values before updating A_prev
        cache_list.append((Ztilde,Zhat,mu,Z,std,A_prev))

        #update A_prev
        A_prev=activations[l](Ztilde)

    return cache_list,A_prev

def compute_cost(AL,Y):
    '''
    Description: computes the (softmax) cost of a network output

    Inputs:
    - AL: the output produced by the network in question
    - Y: the true labels to which the outputs correspond

    Returns:
    - the aforementioned cost
    '''

    #retrieve the size of the mini batch for which the cost is being computed
    m=Y.shape[1]

    #compute the loss of each element
    loss=-np.sum(Y*np.log(AL),axis=0,keepdims=True)

    #average the element-wise loss to obtain the cost
    return np.squeeze(np.sum(loss,axis=1,keepdims=True))/m

def propagate_backward(Y,AL,cache_list,parameters,activations,epsilon=1e-8):
    '''
    Description: performs the backward pass along a network with parameters 'parameters'

    Inputs:
    - Y: the true labels corresponding to the mini batch on which the network is currently being trained
    - AL: the activation produced by the network in question, corresponding to Y
    - cache_list: one of the outputs of propagate_forward, the cache list needed to perform back propagation
    - parameters: the parameters of the network in question
    - activations: the activation functions of each of the layers in the network with an activation function
    - epsilon: avoids division by zero

    Returns:
    - grads: the gradients of each of the parameters in the network
    '''

    #instantiate our gradients dictionary
    grads={}

    #retrieve the size of the mini batch on which the network is currently being trained
    m=Y.shape[1]

    #retrieve the number of layers in the network
    L=len(parameters)//3

    #loop through the entire network
    for l in reversed(range(L)):

        #retrieve the relevant values from the cache corresponding to the current layer
        (Ztilde,Zhat,mu,Z,std,A_prev)=cache_list[l]

        #compute the gradient of the variable to which we apply the activation function of the current layer (during the forward pass)
        if l==L-1:
            dZtilde_loss=AL-Y
            dZtilde=dZtilde_loss/m
        else:
            dZtilde=dA*activations[l](Ztilde,backward=True)

        #propagate backwards through the batch normalization portion of the layer
        dZhat=dZtilde*parameters['gamma'+str(l+1)]
        dgamma=np.sum(dZtilde*Zhat,axis=1,keepdims=True)
        dbeta=np.sum(dZtilde,axis=1,keepdims=True)
        dstd=np.sum(dZhat*((mu-Z)/(2*(std+epsilon)**(3/2))),axis=1,keepdims=True)
        dmu=-np.sum(dZhat*(1/np.sqrt(std+epsilon)),axis=1,keepdims=True)
        dZ=dZhat*(1/np.sqrt(std+epsilon))+(2*dstd*(Z-mu))/m+dmu/m

        #propagate backward through the weighted protion of the layer
        dW=np.dot(m*dZ,A_prev.T)

        #prepare for back propagation through the preceding layer (if there is a preceding layer)
        dA=np.dot(parameters['W'+str(l+1)].T,dZ)

        #amend our gradient dictionary
        grads['dgamma'+str(l+1)]=dgamma
        grads['dbeta'+str(l+1)]=dbeta
        grads['dW'+str(l+1)]=dW
    
    return grads

def update_parameters(parameters,grads,learning_rate,epoch_num):
    '''
    Description: updates the parameters of a network in accordance with the gradients obtained while propagating backward along said network (using learning rate decay)

    Inputs:
    - parameters: the parameters of the network whose parameters we wish to update
    - grads: the gradients corresponding to each of the elements of parameters
    - learning_rate: the initial learning rate
    - epoch_num: the 'how many-th' epoch we're currently executing

    Returns:
    - new_parameters: the updated parameters
    '''

    #perform learning rate decay first
    lrate=learning_rate/(1+(1e-1)*epoch_num)

    #retrieve the number of layers in the network
    L=len(parameters)//3

    #instantiate our return variable
    new_parameters={}

    #populate our return variable with the relevant updated parameters
    for l in range(L):
        new_parameters['W'+str(l+1)]=parameters['W'+str(l+1)]-lrate*grads['dW'+str(l+1)]
        new_parameters['gamma'+str(l+1)]=parameters['gamma'+str(l+1)]-lrate*grads['dgamma'+str(l+1)]
        new_parameters['beta'+str(l+1)]=parameters['beta'+str(l+1)]-lrate*grads['dbeta'+str(l+1)]

    return new_parameters
