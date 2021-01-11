# FruitNet

The following README is a breakdown of each of the key components of the above repository.

1. [  Project Description  ](#desc)
2. [  Data Handling  ](#data)
    1. [  Loading in Data  ](#loading)
3. [  Architecture  ](#arch)
    1. [  Subnets  ](#sub)
    2. [  Supernet  ](#sup)
4. [  Example Network(s)  ](#ex)
    1. [  FruitNet_v1.0  ](#fn1)
5. [  FruitNet Vision  ](#fnv)
6. [  References  ](#ref)

<a name="desc"></a>
## 1. Description

FruitNet is a deep learning framework that enables users to define and train variations of the typical fully-connected model to recognize square, grayscale image contents (otherwise stated, FruitNet is an image classification framework that applies specifically to images that are both square and grayscale (more on this [here](#data))). The project draws inspiration from _M.D. Zeiler_ and _R. Furgus_' paper, _Visualizing and Understanding Convolutional Neural Networks_ (to which the hyperlink is found [here](#ref)), which seeks to identify (to an extent) what exactly it is that convolutional layers are learning, by examining the activations produced by particular nodes in a convolutional layer (belonging to a pretrained convolutional neural network (CNN)), across large datasets. Noting that nodes in convolutional layers may be regarded as 'seeing' only limited portions of the entire image input into the network, FruitNet handles images as follows: FruitNet splits input images into horizontal strips (each of equal height, spanning along the entire width of the input image), flattens said strips, and subsequently trains a series of fully-connected neural networks (FCNNs) to make sense of each of the strips into which we've split our image. We call such networks **subnets** in this project. From here, FruitNet feeds the activations produced by these subnets into a greater FCNN, called a **supernet** in this project, to obtain finally a classification of the original input image (more on the intricacies of this process [here](#architecture)).

A reasonable question at this point is why go through all the hassle of splitting our image up into strips and training separate subnets to make sense of the contents of the aforementioned strips, instead of just feeding the entire (flattened) input image into a single FCNN? The answer lies in the fact that the purpose of thsi project is not to produce the most efficient image classification framework we can. Instead, the purpose of this project is to replicate the process presented in the aforementioned paper on CNNs, using FCNNs, instead. After all, this project is meant to be purely educational.
<a name="data"></a>
## 2. Data Handling

Note that there are two files in this repository, _dataset_operations.py_, and _fruitnet_api.py_, that collectively contain everything one might need to know regarding the way in which FruitNet handles its data. Nevertheless, we will utilize this space to provide some further visual intuition for what exactly goes on behind the scenes.

<a name="loading"></a>
### i. Loading in Data

The FruitNet API loads in data from _.h5_ files (using the _h5py_ library, of course) by reading from two datasets: an input feature dataset, and a corresponding label dataset. The input feature dataset is necessarily of the shape `(m,w,w,1)`, where `m` is the number of images in the dataset, `w` is the width of our image (and therefore also the height of our image, since our images are necessarily square), and the final dimension of the dataset shape being `1` is a consequence of the fact that we are dealing exclusively with grayscale images. Similarly, the corresponding label dataset is necessarily of shape `(c,m)`, where `m` maintains its definition, and `c` is the number of classes to which our images may belong.

<a name="arch"></a>
## 3. Architecture

<a name="sub"></a>
### i. Subnets

<a name="sup"></a>
### ii. Supernet

<a name="ex"></a>
## 4. Example Network(s)

<a name="fn1"></a>
### i. FruitNet_v1.0

<a name="fnv"></a>
## 5. FruitNet Vision

<a name="ref"></a>
## 6. References

1. <a href="https://arxiv.org/pdf/1311.2901.pdf">Zeiler, M.D. and Fergus, R., 2014, September. Visualizing and understanding convolutional networks. In European conference on computer vision (pp. 818-833). Springer, Cham</a>
