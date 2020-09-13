# Object-Detection-from-Scratch
# Short-Description
A convolutional neural network designed to detect and classify a single object in a 128 by 128 color image into any one of six classes of fruit. Built in Python without the help of any deep learning frameworks.

# Extended-Description
This convoluional neural network has 12 layers: 2 convolutional layers, followed by a single max pooling layer, followed by a further 6 convolutional layers, and concluded by 3 fully connected layers. The aim of the network is to receive, as input, a 128 by 128 color (RGB) image, and subsequently detect and classify a single object in the image as belonging to any of six classes of fruit (the possibility of none of the six possible fruits being present does not exist in the case of this network, which is a consequence of the training set, which we will discuss in detail shortly). The network does so by using a linear activation function (in the final layer).
