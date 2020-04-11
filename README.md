# Important

We ask you to cite the main publication related to this software whenever you use any part of this software in any scientific publication.

You may use the following .bibtex to cite the main publication of this software:
```
@article {da Cruz2020.03.25.000935,
	author = {da Cruz, Murilo Horacio Pereira and Domingues, Douglas Silva and Saito, Priscila Tiemi Maeda and Paschoal, Alexandre Rossi and Bugatti, Pedro Henrique},
	title = {TERL: Classification of Transposable Elements by Convolutional Neural Networks},
	elocation-id = {2020.03.25.000935},
	year = {2020},
	doi = {10.1101/2020.03.25.000935},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2020/03/26/2020.03.25.000935},
	eprint = {https://www.biorxiv.org/content/early/2020/03/26/2020.03.25.000935.full.pdf},
	journal = {bioRxiv}
}
```

# Instalation

To install TERL you need to clone the repository into your local machine. First you need to have git installed in your local machine. You can follow [these steps](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) to install git.
Once you have git installed, you can clone this repository with the following command:
```
git clone https://github.com/muriloHoracio/TERL
```

After the clone, you have a directory named TERL which contains all codes to train TERL and classify sequences.

Since TERL is made on python and use some libraries, we recomend the use of virtual environmnets to run it. Before installing virtualenv, make sure you have pip3 installed. To install pip3 run the followoing command:
```
sudo apt-get install python3-pip
```
After installing pip3, you must ensure that python headers are installed. If you are using a Ubuntu derived Linux distro, run the following command:
```
sudo apt-get install python3-dev
```
To create a virtual environment you need to have virtualenv installed, in order to do that you can run the following command in terminal:
```
sudo apt-get install virtualenv
```
To create the virtual environment, you need to execute the following command:
```
virtualenv -p python .venv
```
Once the virtual environment is created, you need to active the environment in order to install the dependencies of TERL. To do this, run the following command:
```
. .venv/bin/activate
```
If everything worked well, you will notice that ``(.venv)`` will appear before the user name in the command line.

Now you must install the dependencies needed to run TERL. 

If you are using GPU, you must run the following command:
```
pip3 install -r requirements-gpu.txt
```
Otherwise, you must run the following command:
```
pip3 install -r requirements.txt
```

# TERL

Transposable Elements Representation Learner

The TERL can be used to classify any genomic sequence.
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

## Train Usage (terl_train.py)
This is an example how to train a model with TERL. The files are stored over Train and Test folders, which are stored on Dataset folder, sotored in the TERL folder.

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
python3 terl_train.py -r Dataset -l 6 -a conv pool conv pool fc fc -f relu avg relu avg relu relu -w 30 20 30 20 1500 500 -s 1 20 1 20 -fm 64 32 -sg -sr -sm
```

## Train Parameters
This section describes the parameters with its possible values and examples of usage.
### -r, --root
**Required** parameter that defines the Root folder, where Train and Test folders containing sample sequences files are located.

It can be the relative or absolute path to Root

Example:
```
python3 terl_train.py -r ~/TERL/Datasets/DS1
```

### -l, --layers
Parameter that defines the number of layers that will be created. It checks if the number of defined layers 
in the model is correct. The layers can be defined without this parameter, but it is a good practice to use 
it to guarantee that the model have the correct number of layers.

Default value is 8, which is the number of layers of the default model.

Example:
```
python3 terl_train.py -l 6
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
python3 terl_train.py -a conv pool conv pool fc fc
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
python3 terl_train.py -f relu avg relu avg relu relu
```

### -w, --widths
Parameter that defines the widths of the filters (convolution and pooling) and the number of neurons for fully connected layers. The values should be entered according to the --architecture parameter, i.e. the first value should be the width of the first layer filter, the second value should be the width of the second layer's filter, and so on...

Default parameter is: -w 30 20 30 20 30 10 1500 500

Example:
```
python3 terl_train.py -w 30 20 30 20 1500 500
```

### -s, --strides
Parameter that defines the strides of the layers of the model. The values should be entered according to the parameter --architecture, i.e. the first value should be the stride of the first layer, the second value the stride of the second layer, and so on...

Default value is: -s 1 20 1 20 1 10

Example:
```
python3 terl_train.py -s 1 20 1 20
```

### -fm, --feature-maps
Parameter that defines the amount of feature maps of each convolution layer. It should be entered **n** values for a network with **n** convolution layers.

Default value is: -fm 64 32 16

Example:
```
python3 terl_train.py -fm 64 32
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
python3 terl_train.py -o adagrad
```

### -lr, --learning-rate
Parameter that defines the learning rate to be used by the optimizer.

Default value is: -lr 0.001

Example:
```
python3 terl_train.py -lr 0.001
```

### -trb, --train-batch-size
Parameter that defines the train batch size. The train batch is the amount of samples that will be presented to the network for each step during training.

Default value is: 32

Example:
```
python3 terl_train.py -trb 64
```

### -tsb, --test-batch-size
Parameter that defines the test batch size. The test batch is the amount of samples that will be presented to the network for each step during testing.

Default value is: 32

Example:
```
python3 terl_train.py -tsb 64
```

### -e, --epochs
Parameter that defines the number of epochs that training will be executed. In each epoch all training samples are presented to the network during training.

Default value is: 30

Example:
```
python3 terl_train.py -e 100
```

### -d, --dropout
Parameter that defines the dropout rate that is used to drop neurons in each convolution and fully connected layer in the model.

Default value is: 0.5

Example:
```
python3 terl_train.py -d 0.3
```

### -sg, --save-graphs
Parameter that sets confusion matrix and learning curve graphs to be saved. The title of the graphs are defined in the --confusion-matrix-title and --learning-curve-title parameters.

By default, graphs are not saved, meaning you need to set it if you really want to save them.

Example:
```
python3 terl_train.py -sg
```

### -cmt, --confusion-matrix-title
Parameter that defines the title of the confusion matrix graph. The title should not contain the character "-".

Default value is: Confusion Matrix

Example:
```
python3 terl_train.py -cmt Confusion Matrix DS1
```

### -lct, --learning-curve-title
Parameter that defines the title of the learning curve graph. The title should not contain the character "-".

Default value is: Learning Curve

Example:
```
python3 terl_train.py -lct Learning Curve DS1
```

### -p, --prefix
Parameter that defines the prefix name to be used to save files, e.g. graphs, models and reports. The name must be one string, i.e. without spaces.

Default value is: RUN_yyyymmdd_HHMMSS

Where yyyy is the 4 digit current year, mm is the 2 digit current month, dd is the 2 digits current day, hh, mm, and ss is the current hour, minute and second respectively.

Example:
```
python3 terl_train.py -p DS1_Tests
```

### -sm, --save-model
Parameter that sets the model to be saved on the directory defined on --model-export-dir.

By default, the model is not saved. Users who want to save their models must set it with this parameter.

Example:
```
python3 terl_train.py -sm
```

### -md, --model-export-dir
Parameter that defines the folder where the model will be exported. The value should be the relative or absolute path to the desired folder. We suggest the use of folder Models created on the folder TERL.

Default value is: Models/Model_yyyymmdd_HHMMSS

Where yyyy is the 4 digit current year, mm is the 2 digit current month, dd is the 2 digits current day, hh, mm, and ss is the current hour, minute and second respectively.

Example:
```
python3 terl_train.py -md Models/DS1_Model
```

### -sr, --save-report
Parameter that sets the reports to be saved on the folder Outputs that is located in the TERL folder.

By default, reports are not saved. Users who want to save it must set it with this parameter.

Example:
```
python3 terl_train.py -sr
```

### -sm, --save-model
Parameter that sets the model to be saved on the directory defined on --model-export-dir.

By default, the model is not saved. Users who want to save their models must set it with this parameter.

Example:
```
python3 terl_train.py -sm
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
python3 terl_train.py -nv
```
## Test/Classification Usage (terl_test.py)
This is an example how to test TERL or classify files. You must inform a trained and saved model to perform this operation.

Example:
```
python3 terl_test.py -m Models/TERLModel -f file1.fa file2.fa file3.fa 
```

After classification is done, three files with prefix ``TERL_YYYYmmdd_HHMMSS_`` will be created containing the results of the classification. TERL copies the sequences and changes the header according to the predicted class.

## Test/Classification Parameters
This section describes the parameters with its possible values and examples of usage.

### -m, --model
**Required** parameter that defines the model to be used for classification.

Example:
```
python3 terl_test.py -m Models/TERLModel
```

### -f, --files
Parameter that defines the FASTA files to be classified. After classifying the files, output files are created with a prefix name containing the sequences in the original file and the headers with the predicted classes.

Default value is TERL_YYYYmmdd_HHMMSS_ where YYYY, mm, dd, HH, MM and SS means the current year, month, day, hour, minutes and seconds.

Example:
```
python3 terl_test.py -m Models/TERLModel -f file1.fa file2.fa file3.fa
```

### -b, --batch
Parameter that defines the batch size that will be used to load sequences and classify them.

Default value is 32

Example:
```
python3 terl_test.py -m Models/TERLModel -f file1.fa file2.fa file3.fa -b 64
```

### -p, --prefix
Parameter that defines the prefix to be used when writing the output files.

Default value is TERL_YYYYmmdd_HHMMSS_

Example:
```
python3 terl_test.py -m Models/TERLModel -f file1.fa file2.fa file3.fa -p TERL_exp1_
```

Which will results in the following output files:
```
TERL_exp1_file1.fa
TERL_exp1_file2.fa
TERL_exp1_file3.fa
```

### -q --quiet
Parameter that deactivates verbose mode, which prints a lot of useful information. 

Default value is False, which prints useful information in the terminal screen

Example:
```
python3 terl_test.py -m Models/TERLModel -f file1.fa file2.fa file3.fa -q
```

The above command will log only Tensorflow's logs
