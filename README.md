# FruitNet

# Short Description
FruitNet is a very basic, bare-bones deep learning framework, built exclusively in Python, as an educational exercise.

# Extended Description
FruitNet is a deep learning framework built exclusively in Python, as an educational exercise. FruitNet exists primarily as a means to build convolutional neural networks that take as input a square image (either coloured, or grayscale) and generate corresponding predictions (the likes of which include classification, detection, and so on). Further details regarding the framework can be found below, under 'Features'.

# Features
- Loading Data: FruitNet is able to load data stored in `.h5` format, provided said `.h5` file contains two datasets, one of which stores the necessary input features, and the other of which stores the corresponding labels.
- Normalization: FruitNet is able to normalize a given set of data, and will return not only the normalized set of data, but also the mean and variation of the set of data that has been normalized, for the purpose of inference.
- Mini Batch Generation: FruitNet is able to shuffle a set of data, and subsequently divide the shuffled set of data into mini batches on which a FruitNet model may be trained.
- Initializing Parameters: FruitNet initializes all of the parameters in accordance with the hyperparameters the user specifies. There are some nuances, though, in that:
  - Batch normalization is applied to each layer aside from max pooling layers
  - Because each layer (with the exception of max pooling layers) is subject to batch normalization, there are no bias parameters in any FruitNet network layer
  - FruitNet necessarily uses the Adam optimizer to optimize the specified loss function, and consequently, when FruitNet initializes the parameters corresponding to a network, it also initializes the dictionaries storing the momentum and RMSprop terms corresponding to each parameter
- Activation Functions: FruitNet enables access to three activation functions, namely the sigmoid activation function, the ReLU activation function, and the sofmax activation function.
- Loss Functions: FruitNet enables access to two activations, namely the cross entropy loss function, and the mean square error loss function.
- Optimizer: As has been mentioned, FruitNet necessarily uses the Adam optimizer. The only features of this optimizer that the user can adjust are the momentum parameter (whose default value is 0.9), and the RMSprop parameter (whose default value is 0.999).
- Compiler: FruitNet, of course, has a compile method, which, when the relevant data has been loaded, and the relevant settings have been appropriately adjusted, executes the training of the given model.
- Metrics: FruitNet, throughout the training process, utilizes two metrics, namely the cost of predictions made (this value is displayed for each training iteration), and the top-2 accuracy of the predictions made (this value, as well as the cost of the predictions made, is displayed at the end of each epoch, when the model's performance on the test set is evaluated).

- Inference: FruitNet has variations of each of its operations (convolution, batch normalization, max pooling, and so on) designed especially to enable inference. This is particularly applicable to operations such as batch normalization, for which training and inference look very different.

- Saving and Loading Models: FruitNet enables the saving (either as an in-training model, or as an inference model) and loading of other FruitNet models stored again in `.h5` format. The user is required to identify the architecture of the network they wish to load before opting to load it. FruitNet also demonstrates the ability to print model summaries, should it be necessary.

# File Descriptions
The following list is a list of all of the relevant files in this repository. They contain all of the source code responsible for FruitNet.
- `initialization.py` This file is written in a functional format, and stores the normalization, mini batch generation, and parameter initialization methods.
- `operations.py` This file is, again, written in a functional format, and stores all of the forward pass and backward pass operations necessarily for each convolutional, max pooling, and fully connected layer.
- `compiler.py` This file is also written in a functional format, and puts together the functions introduced in both `initialization.py` and `operations.py` to create methods responsible for an entire forward pass, an entire backward pass, and subsequent evaluations of the generated predictions, and updating of the relevant parameters.
- `inference_operations.py` This file is also written in a functional format, and stores variations of most of the functions seen in `operations.py` specialized for inference.
- `fruitnet.py` This file is written in an object oriented format, and combines all of the relevant functions introduced in each of the files above to build the FruitNet framework.
- `fruitnet_notebook.ipynb` This is a notebook containing the contents of each of the above files. It removes the need for careful directory management, and thus also eliminates the need for any `import` lines, with the exception of H5Py and NumPy.
