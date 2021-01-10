import numpy as np

def sigmoid(z,backward=False):
    if backward:
        return (1/(1+np.exp(-z)))*(1-(1/(1+np.exp(-z))))
    else:
        return 1/(1+np.exp(-z))

def relu(z,backward=False):
    if backward:
        return np.where(z<0,0,1)
    else:
        return np.where(z<0,0,z)

def softmax(z):
    magnitude=np.sum(np.exp(z),axis=0,keepdims=True)
    return np.exp(z)/magnitude

def concat(z,classes,m=None):
    if m==None:
        subnet_dA=[]
        for i in range(z.shape[0]//classes):
            subnet_dA.append(z[i*classes:(i+1)*classes,:])
        return subnet_dA
    else:
        new_z=np.zeros((len(z)*classes,m))
        for i in range(len(z)):
            new_z[i*classes:(i+1)*classes,:]=z[i]
        return new_z

def propagate_forward(X,parameters,activations,epsilon=1e-8):
    A_prev=X
    m=X.shape[1]
    cache_list=[]
    L=len(parameters)//3
    for l in range(L):
        Z=np.dot(parameters['W'+str(l+1)],A_prev)
        mu=np.mean(Z,axis=1,keepdims=True)
        std=np.var(Z,axis=1,keepdims=True)
        Zhat=(Z-mu)/np.sqrt(std+epsilon)
        Ztilde=parameters['gamma'+str(l+1)]*Zhat+parameters['beta'+str(l+1)]
        cache_list.append((Ztilde,Zhat,mu,Z,std,A_prev))
        A_prev=activations[l](Ztilde)
    return cache_list,A_prev

def compute_cost(AL,Y):
    m=Y.shape[1]
    loss=-np.sum(Y*np.log(AL),axis=0,keepdims=True)
    return np.squeeze(np.sum(loss,axis=1,keepdims=True))/m

def propagate_backward(Y,AL,cache_list,parameters,activations,epsilon=1e-8):
    grads={}
    m=Y.shape[1]
    L=len(parameters)//3
    for l in reversed(range(L)):
        (Ztilde,Zhat,mu,Z,std,A_prev)=cache_list[l]
        if l==L-1:
            dZtilde_loss=AL-Y
            dZtilde=dZtilde_loss/m
        else:
            dZtilde=dA*activations[l](Ztilde,backward=True)
        dZhat=dZtilde*parameters['gamma'+str(l+1)]
        dgamma=np.sum(dZtilde*Zhat,axis=1,keepdims=True)
        dbeta=np.sum(dZtilde,axis=1,keepdims=True)
        dstd=np.sum(dZhat*((mu-Z)/(2*(std+epsilon)**(3/2))),axis=1,keepdims=True)
        dmu=-np.sum(dZhat*(1/np.sqrt(std+epsilon)),axis=1,keepdims=True)
        dZ=dZhat*(1/np.sqrt(std+epsilon))+(2*dstd*(Z-mu))/m+dmu/m
        dW=np.dot(m*dZ,A_prev.T)
        dA=np.dot(parameters['W'+str(l+1)].T,dZ)
        grads['dgamma'+str(l+1)]=dgamma
        grads['dbeta'+str(l+1)]=dbeta
        grads['dW'+str(l+1)]=dW
    return grads

def update_parameters(parameters,grads,learning_rate,epoch_num):
    lrate=learning_rate/(1+(1e-1)*epoch_num)
    L=len(parameters)//3
    for l in range(L):
        parameters['W'+str(l+1)]=parameters['W'+str(l+1)]-lrate*grads['dW'+str(l+1)]
        parameters['gamma'+str(l+1)]=parameters['gamma'+str(l+1)]-lrate*grads['dgamma'+str(l+1)]
        parameters['beta'+str(l+1)]=parameters['beta'+str(l+1)]-lrate*grads['dbeta'+str(l+1)]
    return parameters
