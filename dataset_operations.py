import numpy as np

def build_mini_batches(X,Y,size):
    X=X.reshape(X.shape[0],X.shape[1],X.shape[2])
    X=X.reshape(X.shape[0],X.shape[1]*X.shape[2])
    X=X.T
    m=X.shape[1]
    assert size<=m
    permutation=np.random.permutation(m)
    X=X[:,permutation]
    Y=Y[:,permutation]
    mini_batches=[]
    whole_batches=m//size
    for t in range(whole_batches):
        X_mini_batch=X[:,t*size:(t+1)*size]
        Y_mini_batch=Y[:,t*size:(t+1)*size]
        mini_batches.append((X_mini_batch,Y_mini_batch))
    if m%size!=0:
        X_mini_batch=X[:,whole_batches*size:m]
        Y_mini_batch=Y[:,whole_batches*size:m]
        mini_batches.append((X_mini_batch,Y_mini_batch))
    return mini_batches

def reshuffle_split_mini_batches(X,Y,hp):
    mini_batches=build_mini_batches(X,Y,hp['mini_batch_size'])
    split_mini_batches=[]
    for (Xt,Yt) in mini_batches:
        split_mini_batches.append((split_into_strips(Xt,hp['strips']),Yt))
    return split_mini_batches

def split_into_strips(Xt,strips):
    if int(np.sqrt(Xt.shape[0]))%strips==0:
        width=Xt.shape[0]//strips
        split_mini_batch=[]
        for i in range(strips):
            split_mini_batch.append(Xt[i*width:(i+1)*width,:])
        return tuple(split_mini_batch)
    else:
        raise ValueError('you have elected a number of strips that results in strips of unequal width. please re-enter the number of strips you\'d like, and try again.')
