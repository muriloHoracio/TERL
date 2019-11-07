# CNNTEC

Convolutional Neural Network Transposable Elements Classifier

The CNNTEC can be used to classify any genomic sequence.
This framework provides tools to train and test models.
Users can opt to deploy the trained network model.
There are vast parameters that can be used to define the network architecture and to set the model's parameters.
All the set of parameters are described here with examples of usage.

## Dataset Organization
In order to use this framework to train and test CNNs models for genomic data, users need to organize the structure of dataset's files
The files should be stored in the following way:

```
Root
└─── Train
|    └─── Class1.fa
|    └─── Class2.fa
|    └─── Class3.fa
|    └─── Class4.fa
└─── Test
     └─── Class1.fa
     └─── Class2.fa
     └─── Class3.fa
     └─── Class4.fa
```
The filenames on Train and Test folders must be identicals and reflect the class that each file represents.

## Usage Example
### Train (cnn_train.py)
This is an example how to train a model with CNNTEC. The files are stored over Train and Test folders, which are stored on Dataset folder, sotored in the CNNTEC folder.

```
Dataset
└─── Train
|    └─── LTR.fa
|    └─── LINE.fa
|    └─── SINE.fa
└─── Test
     └─── LTR.fa
     └─── LINE.fa
     └─── SINE.fa
```

This example model have the following architecture:

```
       Architecture: conv    pool    conv    pool    fc      fc      
          Functions: relu    avg     relu    avg     relu    relu    
             Widths: 30      20      30      20      1500    500     
            Strides: 1       20      1       20      
       Feature maps: 64      -       32      -       -       -       
```

Example:
```
python3 cnn_train.py -r Dataset -l 6 -a conv pool conv pool fc fc -f relu avg relu avg relu relu -w 30 20 30 20 1500 500 -s 1 20 1 20 -fm 64 32 -sg -sr -sm
```
### Test (cnn_test.py)
This is an example how to test a CNNTEC trained and saved model.

## Parameters
This section describes the parameters with its possible values and examples of usage
### -r, --root
**Required** parameter that defines the Root folder, where Train and Test folders containing sample sequences files are located.

It can be the relative or absolute path to Root

Example:
```
python3 cnn_train.py -r ~/CNNTEC/Datasets/DS1
```

### -l, --layers
Parameter that defines the number of layers that will be created. It checks if the number of defined layers 
in the model is correct. The layers can be defined without this parameter, but it is a good practice to use 
it to guarantee that the model have the correct number of layers.

Default value is 8, which is the number of layers of the default model.

Example:
```
python3 cnn_train.py -l 6
```

### -a, --architecture
Parameter that defines the architecture of the model. This defines the types of the layers and its order in the model.

**Input and classification layer should not be included**.

The supported values are:
* conv (Convolution layer)
* pool (Pooling layer)
* fc (Fully connected layer)

Default value is: conv pool conv pool conv pool fc fc

Example:
```
python3 cnn_train.py -a conv pool conv pool fc fc
```

### -f, --functions
Parameter that defines the functions of each layer. The functions should be entered according to the --architecture parameter, i.e. the first option should be the function of the first layer defined in --architecture, the second option the function of the second layer and so on...

The available activation functions for convolution and fully connected layers are:
* relu
* tanh
* sigmoid
* leaky_relu
* elu

The available funcions for pooling layers are:
* avg
* max

Default value is: -f relu avg relu avg relu avg relu relu

Example:
```
python3 cnn_train.py -f relu avg relu avg relu relu
```

### -w, --widths
Parameter that defines the widths of the filters (convolution and pooling) and the number of neurons for fully connected layers. The values should be entered according to the --architecture parameter, i.e. the first value should be the width of the first layer filter, the second value should be the width of the second layer's filter, and so on...

Default parameter is: -w 30 20 30 20 30 10 1500 500

Example:
```
python3 cnn_train.py -w 30 20 30 20 1500 500
```

### -s, --strides
Parameter that defines the strides of the layers of the model. The values should be entered according to the parameter --architecture, i.e. the first value should be the stride of the first layer, the second value the stride of the second layer, and so on...

Default value is: -s 1 20 1 20 1 10

Example:
```
python3 cnn_train.py -s 1 20 1 20
```

### -fm, --feature-maps
Parameter that defines the amount of feature maps of each convolution layer. It should be entered **n** values for a network with **n** convolution layers.

Default value is: -fm 64 32 16

Example:
```
python3 cnn_train.py -fm 64 32
```

### -o, --optimizer
Parameter that defines the optimizer that will be used to train the model and optimize the values of the learnable parameters (i.e. weights) of the model.

The available optimizers are:
* adam
* adadelta
* adagrad
* ftrl
* rmsprop
* grad_desc

Default value is: -o adam

Example:
```
python3 cnn_train.py -o adagrad
```

### -lr, --learning-rate
Parameter that defines the learning rate to be used by the optimizer.

Default value is: -lr 0.001

Example:
```
python3 cnn_train.py -lr 0.001
```

### -trb, --train-batch-size
Parameter that defines the train batch size. The train batch is the amount of samples that will be presented to the network for each step during training.

Default value is: 32

Example:
```
python3 cnn_train.py -trb 64
```

### -tsb, --test-batch-size
Parameter that defines the test batch size. The test batch is the amount of samples that will be presented to the network for each step during testing.

Default value is: 32

Example:
```
python3 cnn_train.py -tsb 64
```

### -e, --epochs
Parameter that defines the number of epochs that training will be executed. In each epoch all training samples are presented to the network during training.

Default value is: 30

Example:
```
python3 cnn_train.py -e 100
```

### -d, --dropout
Parameter that defines the dropout rate that is used to drop neurons in each convolution and fully connected layer in the model.

Default value is: 0.5

Example:
```
python3 cnn_train.py -d 0.3
```

### -sg, --save-graphs
Parameter that sets confusion matrix and learning curve graphs to be saved. The title of the graphs are defined in the --confusion-matrix-title and --learning-curve-title parameters.

By default, graphs are not saved, meaning you need to set it if you really want to save them.

Example:
```
python3 cnn_train.py -sg
```

### -cmt, --confusion-matrix-title
Parameter that defines the title of the confusion matrix graph. The title should not contain the character "-".

Default value is: Confusion Matrix

Example:
```
python3 cnn_train.py -cmt Confusion Matrix DS1
```

### -lct, --learning-curve-title
Parameter that defines the title of the learning curve graph. The title should not contain the character "-".

Default value is: Learning Curve

Example:
```
python3 cnn_train.py -lct Learning Curve DS1
```

### -p, --prefix
Parameter that defines the prefix name to be used to save files, e.g. graphs, models and reports. The name must be one string, i.e. without spaces.

Default value is: RUN_yyyymmdd_HHMMSS

Where yyyy is the 4 digit current year, mm is the 2 digit current month, dd is the 2 digits current day, hh, mm, and ss is the current hour, minute and second respectively.

Example:
```
python3 cnn_train.py -p DS1_Tests
```

### -sm, --save-model
Parameter that sets the model to be saved on the directory defined on --model-export-dir.

By default, the model is not saved. Users who want to save their models must set it with this parameter.

Example:
```
python3 cnn_train.py -sm
```

### -md, --model-export-dir
Parameter that defines the folder where the model will be exported. The value should be the relative or absolute path to the desired folder. We suggest the use of folder Models created on the folder CNNTEC.

Default value is: Models/Model_yyyymmdd_HHMMSS

Where yyyy is the 4 digit current year, mm is the 2 digit current month, dd is the 2 digits current day, hh, mm, and ss is the current hour, minute and second respectively.

Example:
```
python3 cnn_train.py -md Models/DS1_Model
```

### -sr, --save-report
Parameter that sets the reports to be saved on the folder Outputs that is located in the CNNTEC folder.

By default, reports are not saved. Users who want to save it must set it with this parameter.

Example:
```
python3 cnn_train.py -sr
```

### -sm, --save-model
Parameter that sets the model to be saved on the directory defined on --model-export-dir.

By default, the model is not saved. Users who want to save their models must set it with this parameter.

Example:
```
python3 cnn_train.py -sm
```

### -nv, --no-verbose
Parameter that disables the verbose mode, which provides useful information to the user.

The verbose mode shows the following information:
* OPTIONS (all parameters used)
* FILES (training and testing file)
* CLASSIFICATION INFO (classes, train and test size, longest sequence and vocabulary size)
* Accuracy micro, macro and simple after each epoch
* REPORT (confusion matrix and classification metrics)
* TIME (train and test times)

By default, verbose is on. Users who want to disable it must set it with this parameter.

Example:
```
python3 cnn_train.py -nv
```
