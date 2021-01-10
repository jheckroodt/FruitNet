import numpy as np
from dataset_operations import *
from initialization_operations import *
from network_operations import *

def train(X,Y,hp):
    normalized_X=normalize_inputs(X)
    split_mini_batches=reshuffle_split_mini_batches(normalized_X,Y,hp)
    subnet_parameters,supnet_parameters=build_nets(split_mini_batches[0],hp)
    for e in range(hp['epochs']):
        for (Xt,Yt) in split_mini_batches:
            all_subnet_AL=[]
            all_caches={}
            for i in range(len(Xt)):
                subnet_cache,subnet_AL=propagate_forward(Xt[i],subnet_parameters['subnet'+str(i+1)],hp['subnet_activations'])
                all_subnet_AL.append(subnet_AL)
                all_caches['subnet'+str(i+1)]=subnet_cache
                subnet_grads=propagate_backward(Yt,subnet_AL,subnet_cache,subnet_parameters['subnet'+str(i+1)],hp['subnet_activations'])
                subnet_parameters['subnet'+str(i+1)]=update_parameters(subnet_parameters['subnet'+str(i+1)],subnet_grads,hp['learning_rate'],e)
            Xt_concat=concat(all_subnet_AL,Yt.shape[0],m=Yt.shape[1])
            supnet_cache,supnet_AL=propagate_forward(Xt_concat,supnet_parameters,hp['supnet_activations'])
            all_caches['supnet']=supnet_cache
            supnet_grads=propagate_backward(Yt,supnet_AL,supnet_cache,supnet_parameters,hp['supnet_activations'])
            supnet_parameters=update_parameters(supnet_parameters,supnet_grads,hp['learning_rate'],e)
        Y_hat=np.where(supnet_AL==np.amax(supnet_AL,axis=0,keepdims=True),1,0)
        print('--------------------------------------------------')
        print('Cost after epoch '+str(e+1)+': '+str(compute_cost(supnet_AL,Yt)))
        print('Error on final mini batch in this epoch: '+str(100*(np.linalg.norm(Yt-Y_hat)/np.sqrt(2*Yt.shape[1]))))
        print('--------------------------------------------------')
        split_mini_batches=reshuffle_split_mini_batches(normalized_X,Y,hp)
    return subnet_parameters,supnet_parameters,all_caches
