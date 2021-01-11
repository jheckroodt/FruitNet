import numpy as np
from dataset_operations import *
from initialization_operations import *
from network_operations import *

def train(X,Y,hp):
    '''
    Description: trains the network on X and Y

    Inputs:
    - X: the training set
    - Y: the true labels corresponding to Y
    - hp: the hyperparameters in accordance with which our training takes place

    Returns:
    - subnet_parameters: the dictionary containing the trained parameters of each of the subnets
    - supnet_parameters: the dictionary containing the trained parameters corresponding to the supernet
    - all_cache: the cache lists for each of the subnets, as well as the supernet, returned for the purpose of adjusting the value of each of the parameters so that said parameters become inference-ready
    '''

    #normalize the dataset
    normalized_X=normalize_inputs(X)

    #split the dataset into appropriate mini batches
    split_mini_batches=reshuffle_split_mini_batches(normalized_X,Y,hp)

    #instatiate the subnet parameters, as well as the supernet parameters
    subnet_parameters,supnet_parameters=build_nets(split_mini_batches[0],hp)

    #loop across each epoch
    for e in range(hp['epochs']):

        #loop across each mini batch
        for (Xt,Yt) in split_mini_batches:

            #instatiate the variables responsible for keeping track of the subnet activations, as well as the subnet cache lists
            all_subnet_AL=[]
            all_caches={}

            #loop across each of the subsets of Xt to propagate the appropriate subset forward along the corresponding subnet, and subsequently performing the appropriate backward pass
            for i in range(len(Xt)):
                subnet_cache,subnet_AL=propagate_forward(Xt[i],subnet_parameters['subnet'+str(i+1)],hp['subnet_activations'])
                all_subnet_AL.append(subnet_AL)
                all_caches['subnet'+str(i+1)]=subnet_cache
                subnet_grads=propagate_backward(Yt,subnet_AL,subnet_cache,subnet_parameters['subnet'+str(i+1)],hp['subnet_activations'])
                subnet_parameters['subnet'+str(i+1)]=update_parameters(subnet_parameters['subnet'+str(i+1)],subnet_grads,hp['learning_rate'],e)
            
            #concatenate the subnet activations
            Xt_concat=concat(all_subnet_AL,Yt.shape[0],m=Yt.shape[1])

            #propagate the above concatenation forward along the supernet, keeping track of the resulting cache list
            supnet_cache,supnet_AL=propagate_forward(Xt_concat,supnet_parameters,hp['supnet_activations'])
            all_caches['supnet']=supnet_cache

            #propagate backward along the supernet, and update the parameters of the supernet accordingly
            supnet_grads=propagate_backward(Yt,supnet_AL,supnet_cache,supnet_parameters,hp['supnet_activations'])
            supnet_parameters=update_parameters(supnet_parameters,supnet_grads,hp['learning_rate'],e)

        #form predictions using the above produced activations
        Y_hat=np.where(supnet_AL==np.amax(supnet_AL,axis=0,keepdims=True),1,0)

        #print the results obtained from the above epoch to the terminal
        print('--------------------------------------------------')
        print('Cost after epoch '+str(e+1)+': '+str(compute_cost(supnet_AL,Yt)))
        print('Error on final mini batch in this epoch: '+str(100*(np.linalg.norm(Yt-Y_hat)/np.sqrt(2*Yt.shape[1]))))
        print('--------------------------------------------------')

        #reshuffle the mini batches
        split_mini_batches=reshuffle_split_mini_batches(normalized_X,Y,hp)

    return subnet_parameters,supnet_parameters,all_caches
