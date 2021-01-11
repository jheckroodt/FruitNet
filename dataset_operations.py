import numpy as np

def build_mini_batches(X,Y,size):
    '''
    Description: splits the given dataset, X, into mini batches, preserving the correspondance between the dataset and its corresponding labels

    Inputs:
    - X: the dataset to be split into mini batches
    - Y: the labels corresponding to X
    - size: the size of each mini batch

    Returns:
    - mini_batches: a list of tuples of the form (Xt,Yt), where Xt is a mini batch of X, and Yt are the labels corresponding to Xt
    '''

    #reshape the dataset X from a conventional image dataset shape to a fully connected input shape
    X=X.reshape(X.shape[0],X.shape[1],X.shape[2])
    X=X.reshape(X.shape[0],X.shape[1]*X.shape[2])
    X=X.T
    m=X.shape[1]
    assert size<=m

    #create the shuffled order of the dataset
    permutation=np.random.permutation(m)

    #shuffle the dataset, as well as its corresponding labels
    X=X[:,permutation]
    Y=Y[:,permutation]

    #instantiate the mini batch list
    mini_batches=[]

    #create the mini batches of size 'size'
    whole_batches=m//size
    for t in range(whole_batches):
        X_mini_batch=X[:,t*size:(t+1)*size]
        Y_mini_batch=Y[:,t*size:(t+1)*size]
        mini_batches.append((X_mini_batch,Y_mini_batch))

    #if there are any elements of the dataset that have not been arranged into a mini batch, arrange said elements into a mini batch
    if m%size!=0:
        X_mini_batch=X[:,whole_batches*size:m]
        Y_mini_batch=Y[:,whole_batches*size:m]
        mini_batches.append((X_mini_batch,Y_mini_batch))
    
    return mini_batches

def split_into_strips(Xt,strips):
    '''
    Description: split the mini batch Xt into subsets that comprise strips of the original image to which Xt corresponds

    Inputs:
    - Xt: mini batch to split into the aforementioned subsets
    - strips: the number of subsets into which to split our mini batches

    Returns:
    - tuple(split_mini_batch): a tuple of the form (Xt1,Xt2,...,Xtstrips), where each Xti (for i ranging from 1 to strips) is an appropriate subset of Xt
    '''

    #confirm the validity of the number of strips chosen
    if int(np.sqrt(Xt.shape[0]))%strips==0:

        #establish the width of each strip (or the height, rather)
        width=Xt.shape[0]//strips

        #instantiate the list of subsets of Xt
        split_mini_batch=[]

        #split Xt into subsets and organize into split_mini_batch appropriately
        for i in range(strips):
            split_mini_batch.append(Xt[i*width:(i+1)*width,:])
    
        return tuple(split_mini_batch)
    else:
        raise ValueError('you have elected a number of strips that results in strips of unequal width. please re-enter the number of strips you\'d like, and try again.')

def reshuffle_split_mini_batches(X,Y,hp):
    '''
    Description: splits the dataset X into minibatches, and splits each of these mini batches into appropriate subsets, as described in the description of split_into_strips

    Inputs:
    - X: dataset to be split into mini batches and subsets
    - Y: the labels corresponding to X (throughout the operations this function performs, correspondence between X and Y is preserved)
    - hp: the hyperparameters in accordance with which this function operates

    Returns:
    - split_mini_batches: a list of tuples of the form ((Xt1,Xt2,...,Xtstrips),Yt), where (Xt1,Xt2,...,Xtstrips) is as described above, and Yt is the set of labels corresponding to (Xt1,Xt2,...,Xtstrips)
    '''

    #split our dataset into mini batches
    mini_batches=build_mini_batches(X,Y,hp['mini_batch_size'])

    #instantiate our return variable
    split_mini_batches=[]

    #split the above mini batches into subsets and arrange into split_mini_batches (with Yt) accordingly
    for (Xt,Yt) in mini_batches:
        split_mini_batches.append((split_into_strips(Xt,hp['strips']),Yt))
    
    return split_mini_batches
