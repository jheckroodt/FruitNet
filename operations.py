import numpy as np

def linear_forward(A_prev,W):
    '''
    Description:
    This method dots W with A_prev

    Inputs:
    A_prev - previous activation
    W - relevant parameter

    Outputs:
    - a cache containing A_prev and W for back propagation
    - W dotted with A_prev
    '''
    return (A_prev,W),np.dot(W,A_prev)

def linear_backward(dA,cache):
    '''
    Description:
    This method back propagates the process of dotting W with A_prev

    Inputs:
    dA - input gradients
    cache - tuple of the form (A_prev,W)

    Outputs:
    dA_prev - gradient to be propagated backwards further
    dW - the relevant weight parameter's gradient
    '''
    (A_prev,W)=cache
    dW=np.dot(dA,A_prev.T)
    dA_prev=np.dot(W.T,dA)
    return dA_prev,dW

def activation_forward(Z,activation):
    '''
    Description:
    This method applies the activation function activation to Z

    Inputs:
    Z - value to be activated
    activation - activation function to be applied

    Outputs:
    - a cache containing Z and the activation function
    - the activation function activation applied to Z
    '''
    if activation=='relu':
        return (Z,'relu'),np.where(Z<0,0,Z)
    elif activation=='sigmoid':
        return (Z,'sigmoid'),1/(1+np.exp(-Z))
    elif activation=='softmax':
        return (Z,'softmax'),np.exp(Z)/np.sum(np.exp(Z),axis=0,keepdims=True)

def activation_backward(dA,cache):
    '''
    Description:
    Back propagates along the activation function application point

    Inputs:
    dA - successive gradient
    cache - tuple of the form (Z,activation)

    Outputs:
    - gradient to be propagated backward
    '''
    (Z,activation)=cache
    if activation=='relu':
        dactivation=np.where(Z<0,0,1)
        return dA*dactivation
    elif activation=='sigmoid':
        dactivation=(1/(1+np.exp(-Z)))*(1-(1/(1+np.exp(-Z))))
        return dA*dactivation
    elif activation=='softmax':
        return dA

def convolve_forward(A_prev,W,stride,pad):
    '''
    Description:
    This method convolves A_prev with the filters in W

    Inputs:
    A_prev - activation from previous layer, shape (m,n_H_prev,n_W_prev,n_C_prev)
    W - filters of current layer, shape (f,f,n_C_prev,n_C)
    stride - the stride of the convolution
    pad - the padding applies to the 

    Outputs:
    - a cache containing A_prev, Z, pad, stride, and W
    - A_prev convolved with the filters in W
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
    return (A_prev,Z,pad,stride,W),Z

def convolve_backward_inputs(dA,cache):
    '''
    Description:
    This method propagates backwards along a convolutional point in the network

    Inputs:
    dA - as described above
    cache - tuple of the form (A_prev,Z,pad,stride,filters)

    Outputs:
    dA_prev - gradient to be propagated backwards
    - gradient with respect to the relevant filters
    '''
    (A_prev,Z,pad,stride,filters)=cache
    (m,n_H_prev,n_W_prev,n_C_prev)=A_prev.shape
    (m,n_H,n_W,n_C)=dA.shape
    (f,f,n_C_prev,n_C)=filters.shape
    dW=np.zeros((m,f,f,n_C_prev,n_C))
    A_prev_pad=zero_pad(A_prev,pad)
    dA_prev=np.zeros((m,n_H_prev,n_W_prev,n_C_prev))
    dA_prev_pad=zero_pad(dA_prev,pad)
    for v in range(n_H):
        vert_start=stride*v
        vert_end=stride*v+f
        for h in range(n_W):
            horiz_start=stride*h
            horiz_end=stride*h+f
            for c in range(n_C):
                dA_temp=dA[:,v,h,c].reshape(m,1,1,1)
                dA_prev_pad[:,vert_start:vert_end,horiz_start:horiz_end,:]+=dA_temp*filters[:,:,:,c]
                dW[:,:,:,:,c]+=dA_temp*A_prev_pad[:,vert_start:vert_end,horiz_start:horiz_end,:]
    dA_prev=dA_prev_pad[:,pad:n_H_prev+pad,pad:n_W_prev+pad,:]
    return dA_prev,np.sum(dW,axis=0,keepdims=True)[0,:,:,:,:]

def convolve_backward_filters(dA,cache):
    '''
    Description:
    This method also propagates backwards along a convolutional point in the network, but does not return
    the gradient with respect to the input to the given point in the network

    Inputs:
    dA - as described above
    cache - tuple of the form (A_prev,Z,pad,stride,filters)

    Outputs:
    - gradient with respect to the relevant filters
    '''
    (A_prev,Z,pad,stride,filters)=cache
    (m,n_H_prev,n_W_prev,n_C_prev)=A_prev.shape
    (m,n_H,n_W,n_C)=dA.shape
    (f,f,n_C_prev,n_C)=filters.shape
    dW=np.zeros((m,f,f,n_C_prev,n_C))
    A_prev_pad=zero_pad(A_prev,pad)
    for v in range(n_H):
        vert_start=stride*v
        vert_end=stride*v+f
        for h in range(n_W):
            horiz_start=stride*h
            horiz_end=stride*h+f
            for c in range(n_C):
                dW[:,:,:,:,c]+=dA[:,v,h,c].reshape(m,1,1,1)*A_prev_pad[:,vert_start:vert_end,horiz_start:horiz_end,:]
    return np.sum(dW,axis=0,keepdims=True)[0,:,:,:,:]

def batch_norm_forward(Z,gamma,beta,fc=False,epsilon=1e-8):
    '''
    Description:
    Applies batch normalization to Z using gamma and beta

    Inputs:
    Z - set of data to be batch normalized
    gamma - multiplicative batch normalization parameter
    beta additive batch normalization parameter
    fc - dictates whether or not the layer is fully connected or not

    Outputs:
    - a cache containing gamma, Zhat, mu, Z, and var
    - batch normalized Z
    '''
    if not fc:
        mu=np.mean(Z,axis=0,keepdims=True)
        var=np.var(Z,axis=0,keepdims=True)
        Zhat=(Z-mu)/np.sqrt(var+epsilon)
        Ztilde=gamma*Zhat+beta
    else:
        mu=np.mean(Z,axis=1,keepdims=True)
        var=np.var(Z,axis=1,keepdims=True)
        Zhat=(Z-mu)/np.sqrt(var+epsilon)
        Ztilde=gamma*Zhat+beta
    return (gamma,Zhat,mu,Z,var),Ztilde

def batch_norm_backward(dA,cache,epsilon=1e-8,fc=False):
    '''
    Description:
    Propagates backwards along a batch normalization point in the network

    Inputs:
    dA - as described above
    cache - tuple of the form (Ztilde,Zhat,mu,Z,var)
    fc - determines whether or not the layer is fully connected

    Outputs:
    - gradient to be propagated backwards further
    dgamma - gradient with respect to gamma
    dbeta - gradient with respect to beta
    '''
    (gamma,Zhat,mu,A_prev,var)=cache
    if fc:
        m=Zhat.shape[1]
        dA=dA/m
        dZhat=dA*gamma
        dgamma=np.sum(dA*Zhat,axis=1,keepdims=True)
        dbeta=np.sum(dA,axis=1,keepdims=True)
        dvar=np.sum(dZhat*((mu-A_prev)/(2*(var+epsilon)**(3/2))),axis=1,keepdims=True)
        dmu=-np.sum(dZhat/np.sqrt(var+epsilon),axis=1,keepdims=True)
        dA_prev=dZhat/np.sqrt(var+epsilon)+(2*dvar*(A_prev-mu))/m+dmu/m
    else:
        m=Zhat.shape[0]
        dA=dA/m
        dZhat=dA*gamma
        dgamma=np.sum(dA*Zhat,axis=0,keepdims=True)
        dbeta=np.sum(dA,axis=0,keepdims=True)
        dvar=np.sum(dZhat*((mu-A_prev)/(2*(var+epsilon)**(3/2))),axis=0,keepdims=True)
        dmu=-np.sum(dZhat/np.sqrt(var+epsilon),axis=0,keepdims=True)
        dA_prev=dZhat/np.sqrt(var+epsilon)+(2*dvar*(A_prev-mu))/m+dmu/m
    return m*dA_prev,dgamma,dbeta

def maxpool_forward(A_prev,f,stride,pad):
    '''
    Description:
    Propagates A_prev forward along a max pooling layer in accordance with the value f

    Inputs:
    A_prev - activation from previous layer, shape (m,n_H_prev,n_W_prev,n_C_prev)
    f - regions from which maximum value will be selected is of shape (f,f)
    stride - the stride with which we adjust the position of our region
    pad - usually 0, padding applied to A_prev

    Outputs:
    - a cache containing A_prev, Z, pad, stride, and f (the filters)
    Z - A_prev after max pooling has been applied in accordance with f
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
    return (A_prev,Z,pad,stride,f),Z

def identify_max(X):
    '''
    Description:
    This method identifies the maximal element in X

    Inputs:
    X - matrix whose greatest element we wish to identify

    Outputs:
    - X, where the maximal element is True, and the remaining elements are False
    '''
    return np.where(X==np.max(X,axis=(1,2),keepdims=True),True,False)

def maxpool_backward(dA,cache):
    '''
    Description:
    Propagates backward along a max pooling layer

    Inputs:
    dA - stores the gradient we've received
    cache - (A_prev,Z,pad,stride,f) tuple

    Outputs:
    dA_prev - the gradient to be propagated backwards
    '''
    (A_prev,Z,pad,stride,f)=cache
    (m,n_H_prev,n_W_prev,n_C_prev)=A_prev.shape
    (m,n_H,n_W,n_C)=dA.shape
    dA_prev=np.zeros((m,n_H_prev,n_W_prev,n_C_prev))
    dA_prev_pad=zero_pad(dA_prev,pad)
    A_prev_pad=zero_pad(A_prev,pad)
    for v in range(n_H):
        vert_start=stride*v
        vert_end=stride*v+f
        for h in range(n_W):
            horiz_start=stride*h
            horiz_end=stride*h+f
            for c in range(n_C):
                A_region=A_prev_pad[:,vert_start:vert_end,horiz_start:horiz_end,c]
                A_region_max=identify_max(A_region)
                dA_prev_pad[:,vert_start:vert_end,horiz_start:horiz_end,c]+=dA[:,v,h,c].reshape(m,1,1)*A_region_max
    dA_prev=dA_prev_pad[:,pad:n_H_prev+pad,pad:n_W_prev+pad,:]
    return dA_prev

def flatten(Z,cache=None):
    '''
    Description:
    This method flattens the activation from a convolutional layer to produce a feature vector

    Inputs:
    A_prev - the activation we are passing into our
             fully connected layer
    cache - tuple of the relevant (m,n_H,n_W,n_C) (only for backward pass)

    Outputs:
    - Z, reshaped a feature vector, or a collected of feature vectors (stored as a matrix)
    '''
    if cache==None:
        (m,n_H,n_W,n_C)=Z.shape
        return Z.reshape(n_H*n_W*n_C,m)
    else:
        (m,n_H,n_W,n_C)=cache
        return Z.reshape(m,n_H,n_W,n_C)

def zero_pad(A_prev,pad):
    '''
    Description:
    This method pads A_prev in accordance with the value pad

    Inputs:
    A_prev - activation from previous layer (not padded yet)
             np.array() object of shape (m,n_H_prev,n_W_prev,n_C_prev)
    pad - amount of padding

    Outputs:
    - A_prev_padded - A_prev, padded by pad
    '''
    if pad==0:
        return A_prev
    else:
        A_prev_padded=np.pad(A_prev,((0,0),(pad,pad),(pad,pad),(0,0)),mode='constant',constant_values=(0,0))
        return A_prev_padded
