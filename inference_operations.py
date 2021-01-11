import numpy as np
from dataset_operations import *
from initialization_operations import *
from network_operations import *

def to_inf_params(parameters,cache_list,epsilon=1e-8):
    '''
    Description: converts the parameters 'parameters' to inference-ready parameters

    Inputs:
    - parameters: the parameters we wish to convert
    - cache_list: a list containing the values we are to use to perform the above conversion
    - epsilon: prevents division by zero

    Returns:
    - inf_params: the inference-ready variation of parameters
    '''

    #instantiate our return variable
    inf_params={}

    #retrieve the length of the network whose parameters we wish to convert
    L=len(parameters)//3

    #loop through each layer in the network
    for l in range(L):

        #retrieve the relevant cached values
        (Ztilde,Zhat,mu,Z,std,A_prev)=cache_list[l]

        #convert each of the parameters appropriately
        inf_params['W'+str(l+1)]=parameters['W'+str(l+1)]
        inf_params['gamma'+str(l+1)]=parameters['gamma'+str(l+1)]/np.sqrt(std+epsilon)
        inf_params['beta'+str(l+1)]=parameters['beta'+str(l+1)]-mu*(parameters['gamma'+str(l+1)]/np.sqrt(std+epsilon))

    return inf_params

def inference_forward(X,parameters,activations,epsilon=1e-8):
    '''
    Description: propagates the test set X forward along the inference-ready network in question

    Inputs:
    - X: the test set
    - parameters: inference-ready parameters corresponding to the network in question
    - activations: the activation functions of each of the layers in our network
    - epsilon: prevents division by zero

    Returns:
    - A_prev: the activations produced by the infere-ready network in question
    '''

    #assign X to A_prev so that we may propagate A_prev forward through the network in question
    A_prev=X

    #retrieve the length of the network in question
    L=len(parameters)//3

    #loop through each of the layers of the network
    for l in range(L):

        #apply the relevant weights
        Z=np.dot(parameters['W'+str(l+1)],A_prev)

        #apply the appropriate batch normalization parameters, as well as the activation function corresponding to the current layer
        Ztilde=parameters['gamma'+str(l+1)]*Z+parameters['beta'+str(l+1)]
        A_prev=activations[l](Ztilde)

    return A_prev

def perc_error(X,Y,subnet_parameters,supnet_parameters,hp):
    '''
    Description: compute the percentage error of our model on the test set

    Inputs:
    - X: the test set
    - Y: the labels corresponding to X
    - subnet_parameters: the inference-ready subnet parameters
    - supnet_parameters: the inference-ready supernet parameters
    - hp: the hyperparameters in accordance with which the error is computed

    Returns:
    - the percentage error on the test set
    '''

    #normalize the test set (this process needs to necessarily coincide with the normalization of the training set)
    normalized_X=normalize_inputs(X)

    #split the test set into appropriate mini batches
    [(Xt,Yt)]=reshuffle_split_mini_batches(normalized_X,Y,{'mini_batch_size':Y.shape[1],'strips':hp['strips']})

    #instantiate the subnet activation list
    all_subnet_AL=[]

    #loop across each of the subsets in the test set
    for i in range(len(Xt)):

        #propagate forward along each of the subnets and append the resulting activation to all_subnet_AL
        subnet_AL=inference_forward(Xt[i],subnet_parameters['subnet'+str(i+1)],hp['subnet_activations'])
        all_subnet_AL.append(subnet_AL)
    
    #concatenate the above activations
    Xt_concat=concat(all_subnet_AL,Yt.shape[0],m=Yt.shape[1])

    #feed the above concatenation into the supernet
    supnet_AL=inference_forward(Xt_concat,supnet_parameters,hp['supnet_activations'])

    #convert the activations produced by the supernet into a prediction
    Y_hat=np.where(supnet_AL==np.amax(supnet_AL,axis=0,keepdims=True),1,0)
    assert Y_hat.shape==Y.shape

    return 100*(np.linalg.norm(Yt-Y_hat)/np.sqrt(2*Yt.shape[1]))
