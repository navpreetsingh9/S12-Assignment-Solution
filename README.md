# S12-Assignment-Solution

This repository contains an implementation of a CustomResNet model for the CIFAR10 dataset using PyTorch Lightning. The model is trained using the One Cycle Policy with data augmentation and transformation techniques.

## Files Description

The repository consists of the following files:

### 1. `ERA-S12.ipynb`

This Jupyter Notebook file contains the main code and instructions to run the training process for the customResNet model on the CIFAR10 dataset using PyTorch Lightning. It imports the necessary modules from the other files and runs the training process for a specified number of epochs and displays the validation accuracy.

### 2. `dataloader.py`

The `dataloader.py` file contains the code to create the data loader for the CIFAR10 dataset. It handles loading the dataset, applying transformations, and creating batches of data for training and validation.

### 3. `model.py`

The `model.py` file defines the CustomResNet architecture for CIFAR10 using PyTorch Lightning's `LightningModule`. It includes the `PrepLayer`, `Layer1`, `Layer2`, `Layer3`, and the fully connected (`FC`) layer, with appropriate activation functions and batch normalization.

### 4. `trainer.py`

The `trainer.py` file contains the `LitCifar10` class, which serves as the PyTorch Lightning `LightningModule` responsible for defining the training and validation logic, as well as data handling for the CIFAR10 dataset.

#### - LitCifar10 class

The `LitCifar10` class extends `LightningModule` and contains the following key components:

#### - Initialization

- `__init__(self, data_dir="./data", learning_rate=0.03, weight_decay=1e-4, end_lr=10, batch_size=256)`: The constructor method initializes the parameters required for training the model. It sets the data directory, learning rate, weight decay, end learning rate for the One Cycle Policy, and batch size.

#### - Forward Pass

- `forward(self, x)`: This method performs the forward pass of the model by calling the `self.model(x)`.

#### - Training Step

- `training_step(self, batch, batch_idx)`: This method defines the training step for the model. It computes the forward pass, calculates the CrossEntropyLoss, and logs the training loss, accuracy, and learning rate for each epoch.

#### - Validation Step

- `validation_step(self, batch, batch_idx)`: This method defines the validation step for the model. It computes the forward pass, calculates the CrossEntropyLoss, and logs the validation loss and accuracy for each epoch.

#### - Test Step

- `test_step(self, batch, batch_idx)`: This method defines the test step for the model. It computes the forward pass, calculates the CrossEntropyLoss, and logs the test loss and accuracy for each test batch. It also records the incorrect predictions for further analysis.

#### - Test End

- `on_test_end(self)`: This method is called at the end of the test run. It plots and displays some of the incorrect predictions made by the model during the testing phase.

#### - Optimizer and Scheduler Configuration

- `configure_optimizers(self)`: This method sets up the optimizer and learning rate scheduler for training. It utilizes the Adam optimizer with the learning rate determined using a learning rate finder and a One Cycle Learning Rate Scheduler.

#### - Data Related Hooks

- `setup(self, stage=None)`: This method sets up the datasets for training, validation, and testing based on the provided stage. It initializes the train, validation, and test datasets using the specified transformations.

#### - Data Loaders

- `train_dataloader(self)`: This method returns the data loader for the training dataset.
- `val_dataloader(self)`: This method returns the data loader for the validation dataset.
- `test_dataloader(self)`: This method returns the data loader for the test dataset.

### 5. `utils.py`

The `utils.py` file contains utility functions used in the training process, such as functions to get the learning rate, visualize images, get the device, print model summary, and other helper functions.

### 6. `transforms.py`

The `transforms.py` file includes data augmentation and transformation functions used during training. It contains the `RandomCrop`, `FlipLR`, and `CutOut` transformations as specified in the assignment.

## Architecture

The Custom ResNet architecture for CIFAR10 is structured as follows:

==========================================================================================

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
CustomResNet                             [1, 10]                   --
├─Sequential: 1-1                        [1, 64, 32, 32]           --
│    └─Conv2d: 2-1                       [1, 64, 32, 32]           1,728
│    └─ReLU: 2-2                         [1, 64, 32, 32]           --
│    └─BatchNorm2d: 2-3                  [1, 64, 32, 32]           128
│    └─Dropout: 2-4                      [1, 64, 32, 32]           --
├─ResNetBlock: 1-2                       [1, 128, 16, 16]          --
│    └─Sequential: 2-5                   [1, 128, 16, 16]          --
│    │    └─Conv2d: 3-1                  [1, 128, 32, 32]          73,728
│    │    └─MaxPool2d: 3-2               [1, 128, 16, 16]          --
│    │    └─ReLU: 3-3                    [1, 128, 16, 16]          --
│    │    └─BatchNorm2d: 3-4             [1, 128, 16, 16]          256
│    │    └─Dropout: 3-5                 [1, 128, 16, 16]          --
│    └─Sequential: 2-6                   [1, 128, 16, 16]          --
│    │    └─Conv2d: 3-6                  [1, 128, 16, 16]          147,456
│    │    └─ReLU: 3-7                    [1, 128, 16, 16]          --
│    │    └─BatchNorm2d: 3-8             [1, 128, 16, 16]          256
│    │    └─Dropout: 3-9                 [1, 128, 16, 16]          --
│    │    └─Conv2d: 3-10                 [1, 128, 16, 16]          147,456
│    │    └─ReLU: 3-11                   [1, 128, 16, 16]          --
│    │    └─BatchNorm2d: 3-12            [1, 128, 16, 16]          256
├─Sequential: 1-3                        [1, 256, 8, 8]            --
│    └─Conv2d: 2-7                       [1, 256, 16, 16]          294,912
│    └─MaxPool2d: 2-8                    [1, 256, 8, 8]            --
│    └─ReLU: 2-9                         [1, 256, 8, 8]            --
│    └─BatchNorm2d: 2-10                 [1, 256, 8, 8]            512
│    └─Dropout: 2-11                     [1, 256, 8, 8]            --
├─ResNetBlock: 1-4                       [1, 512, 4, 4]            --
│    └─Sequential: 2-12                  [1, 512, 4, 4]            --
│    │    └─Conv2d: 3-13                 [1, 512, 8, 8]            1,179,648
│    │    └─MaxPool2d: 3-14              [1, 512, 4, 4]            --
│    │    └─ReLU: 3-15                   [1, 512, 4, 4]            --
│    │    └─BatchNorm2d: 3-16            [1, 512, 4, 4]            1,024
│    │    └─Dropout: 3-17                [1, 512, 4, 4]            --
│    └─Sequential: 2-13                  [1, 512, 4, 4]            --
│    │    └─Conv2d: 3-18                 [1, 512, 4, 4]            2,359,296
│    │    └─ReLU: 3-19                   [1, 512, 4, 4]            --
│    │    └─BatchNorm2d: 3-20            [1, 512, 4, 4]            1,024
│    │    └─Dropout: 3-21                [1, 512, 4, 4]            --
│    │    └─Conv2d: 3-22                 [1, 512, 4, 4]            2,359,296
│    │    └─ReLU: 3-23                   [1, 512, 4, 4]            --
│    │    └─BatchNorm2d: 3-24            [1, 512, 4, 4]            1,024
├─MaxPool2d: 1-5                         [1, 512, 1, 1]            --
├─Linear: 1-6                            [1, 10]                   5,130
==========================================================================================
Total params: 6,573,130
Trainable params: 6,573,130
Non-trainable params: 0
Total mult-adds (M): 379.27
==========================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 4.65
Params size (MB): 26.29
Estimated Total Size (MB): 30.96
==========================================================================================
```



## One Cycle Policy

The training process uses the One Cycle Policy with the following configurations:

- Total Epochs: 24
- Max Learning Rate (LR) at Epoch: 5
- LRMIN: To be determined during training
- LRMAX: To be determined during training
- No Annihilation during the learning rate schedule

## Data Transformations

The dataset is transformed using the following steps:

- RandomCrop of size 32x32 (after padding of 4)
- FlipLR (Flip Left-Right)
- CutOut with a mask size of 8x8

## Training Configuration

- Batch size: 512
- Optimizer: Adam
- Loss function: CrossEntropyLoss