import numpy as np
from dataset_operations import *
from initialization_operations import *
from network_operations import *

def to_inf_params(parameters,cache_list,epsilon=1e-8):
    inf_params={}
    L=len(parameters)//3
    for l in range(L):
        (Ztilde,Zhat,mu,Z,std,A_prev)=cache_list[l]
        inf_params['W'+str(l+1)]=parameters['W'+str(l+1)]
        inf_params['gamma'+str(l+1)]=parameters['gamma'+str(l+1)]/np.sqrt(std+epsilon)
        inf_params['beta'+str(l+1)]=parameters['beta'+str(l+1)]-mu*(parameters['gamma'+str(l+1)]/np.sqrt(std+epsilon))
    return inf_params

def inference_forward(X,parameters,activations,epsilon=1e-8):
    A_prev=X
    L=len(parameters)//3
    for l in range(L):
        Z=np.dot(parameters['W'+str(l+1)],A_prev)
        Ztilde=parameters['gamma'+str(l+1)]*Z+parameters['beta'+str(l+1)]
        A_prev=activations[l](Ztilde)
    return A_prev

def perc_error(X,Y,subnet_parameters,supnet_parameters,hp):
    normalized_X=normalize_inputs(X)
    [(Xt,Yt)]=reshuffle_split_mini_batches(normalized_X,Y,{'mini_batch_size':Y.shape[1],'strips':hp['strips']})
    all_subnet_AL=[]
    for i in range(len(Xt)):
        subnet_AL=inference_forward(Xt[i],subnet_parameters['subnet'+str(i+1)],hp['subnet_activations'])
        all_subnet_AL.append(subnet_AL)
    Xt_concat=concat(all_subnet_AL,Yt.shape[0],m=Yt.shape[1])
    supnet_AL=inference_forward(Xt_concat,supnet_parameters,hp['supnet_activations'])
    Y_hat=np.where(supnet_AL==np.amax(supnet_AL,axis=0,keepdims=True),1,0)
    assert Y_hat.shape==Y.shape
    return 100*(np.linalg.norm(Yt-Y_hat)/np.sqrt(2*Yt.shape[1]))
