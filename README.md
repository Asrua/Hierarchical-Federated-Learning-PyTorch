# Federated-Learning (PyTorch)

This is a hierarchical federated learning framework, which is an extension of the vanilla federated learning paper : [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629).

You can set the number of the layers and the number of servers at each layer mannually.

Experiments are produced on MNIST, Fashion MNIST and CIFAR10 (both IID and non-IID). In case of non-IID, the data amongst the users can be split equally or unequally.

The federated learning structure is forked from [AshwinRJ] [https://github.com/AshwinRJ/Federated-Learning-PyTorch]

## Requirments
Install all the packages from requirments.txt
* Python3
* Pytorch
* Torchvision

## Data
* Download train and test datasets manually or they will be automatically downloaded from torchvision datasets.
* Experiments are run on Mnist, Fashion Mnist and Cifar.
* To use your own dataset: Move your dataset to data directory and write a wrapper on pytorch dataset class.

## Running the experiments
The baseline experiment trains the model in the conventional way.

* To run the baseline experiment with MNIST on MLP using CPU:
```
python baseline_main.py --model=mlp --dataset=mnist --epochs=10
```
* Or to run it on GPU (eg: if gpu:0 is available):
```
python baseline_main.py --model=mlp --dataset=mnist --gpu=0 --epochs=10
```
-----

Federated experiment involves training a global model using many local models.

* To run the federated experiment with CIFAR on CNN (IID):
```
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --iid=1 --epochs=10
```
* To run the same experiment under non-IID condition:
```
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --iid=0 --epochs=10
```

In the hierarachial FL framework, users are divided into different groups and upload their model parameters to different edge servers. On each edge server, there is a model aggregated. On the central cloud server, there is global model same as vanilla FL. 

* To run the hierarachial FL experiment with non-IID data:
```
python hier_fed_main.py --model=cnn --dataset=cifar --gpu=0 --iid=0 --epochs=10
```

## Options
The default values for various paramters parsed to the experiment are given in ```options.py```. Details are given for the Hierarchical FL parameters:

#### Hierarchical Federated Parameters
* ```--mid_server:```      A list shows the number of servers starting from the edge layer. The cloud layer is not included because it is default as 1. 
* ```--download:```        Whether user download models from the edge server or just use local model
* ```--management:```      Top-down management to ensure the best model is on the edge server.     
