# Deep AutoEncoders for Collaborative Filtering [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=An%20Implementation%20of%20the%20research%20paper%20-%20Training%20Deep%20AutoEncoders%20for%20Collaborative%20Filtering%20by%20NVIDIA.%0Aand%20here%27s%20the%20Github%20link%20-%3E&url=https://github.com/Chinmayrane16/DeepRecommender&hashtags=Deep_Autoencoders,Recommendation_Systems,Movie_lens,NVIDIA,Pytorch,Deep_Learning)

<img src="https://upload.wikimedia.org/wikipedia/commons/9/96/Pytorch_logo.png" width="11%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository is an implementation of the research paper by NVIDIA :

**Training Deep AutoEncoders for Collaborative Filtering**<br>
\- Oleksii Kuchaiev and Boris Ginsburg<br>
[[arXiv](https://arxiv.org/pdf/1708.01715v3.pdf)][[Github](https://github.com/NVIDIA/DeepRecommender)]

## Collaborative Filtering
Collaborative Filtering is a method used by recommender systems to make predictions about an interest of an specific user by collecting taste or preferences information from many other users. The technique of Collaborative Filtering has the underlying assumption that if a user A has the same taste or opinion on an issue as the person B, A is more likely to have B’s opinion on a different issue.

![](images/movies.jpg)

## Deep AutoEncoder
**Autoencoder:**
AutoEncoders are a type of Feed forward neural networks that consits of 3 different layers - encoding layer, representation layer and decoding layer. The representation layer is also called the coding layer. An undercomplete autoencoder has less number of neurons in the coding layer in order to limit the flow of information through the network (bottleneck). An overcomplete autoencoder has greater number of neurons in the hidden layer than the input layer. So, representation layer consists of the latent representations of the input. 

They are a form of unsupervised learning, as the input itself serves as the output, for the purpose of reconstructing it's inputs.<br>
The following image is taken from the research paper.
<p align="center">
  <img width="600" height="400" src="https://github.com/Chinmayrane16/DeepRecommender/blob/master/images/AutoEncoder.png">
</p>

## Implementation
In this repository, I have implemented the state-of-the-art technique for predicting the user ratings using Deep AutoEncoders with the help of [Pytorch](https://github.com/pytorch/pytorch) framework.<br>
I have used the [Movielens 100k](https://grouplens.org/datasets/movielens/100k/) dataset.<br>
You can find the dataset on this repository - [ml-latest-small](https://github.com/Chinmayrane16/DeepRecommender/tree/master/ml-latest-small)

## Requirements
* python = 3.6
* pytorch = 1.0
* pandas = 0.22.0
* numpy = 1.14.3
* CUDA (recommended version >= 8.0)


## Dataset Preprocessing
The [ratings.csv](https://github.com/Chinmayrane16/DeepRecommender/blob/master/ml-latest-small/ratings.csv) file consists of users and their ratings.

I have sorted the dataset based on Timestamp as mentioned in the paper. For the rating prediction task, it is often most relevant to predict future ratings given the past ones instead of predicting ratings missing at random. Training interval contains ratings which came in earlier than the ones from testing interval. Users and items that do not appear in the training set are removed from the validation sets.

This is done in the [DataLoader.ipynb](https://github.com/Chinmayrane16/DeepRecommender/blob/master/DataLoader.ipynb) file.<br>
You can find the sorted train and test datasets here - ([train](https://github.com/Chinmayrane16/DeepRecommender/blob/master/train.csv)) ([test](https://github.com/Chinmayrane16/DeepRecommender/blob/master/test.csv))

## Model Structure
The model has 7 layers having the layer sizes [n, 512, 512, 1024, 512, 512, n]. Some of the tuned hyperparamaters are as follows - 

| Hyperparameters| Value         | Explanation  |
| -------------  |:-------------:| -----:|
| learning_rate  | 0.001        | learning rate |
| num of hidden neurons  | 1024    | Number of neurons in each layer |
| dropout | 0.0 | dropout probability |
| activation_function | selu | type of non-linearity |

The model uses Adam as optimizer with lr = 0.001 although the paper suggests using SGD with lr = 0.001 and momentum = 0.9.<br>
The loss function used is MMSE (Masked Mean Square Error) and to evaluate performance, I used RMSE which is experimentally found to be square root of MMSE.

At the end of the Training for 40 epochs, I acheived RMSE training loss of 0.5567 and validation loss of 0.39.
