from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

# define network
class FFNNClassifier(nn.Module):

    def __init__(self, num_features, num_hidden1, num_hidden2, num_hidden3, num_output, dropout_p):
        super(FFNNClassifier, self).__init__() 
        self.num_features = num_features
        self.num_hidden1 = num_hidden1
        self.num_hidden2 = num_hidden2
        self.num_hidden3 = num_hidden3
        self.num_output = num_output
        self.dropout_p = dropout_p
        # input layer
        self.linear1 = nn.Linear(num_features, num_hidden1)
        self.batchnorm1 = nn.BatchNorm1d(num_hidden1)
        self.dropout = nn.Dropout(p=dropout_p)
        # hidden layer
        self.linear2 = nn.Linear(num_hidden1, num_hidden2)
        self.batchnorm2 = nn.BatchNorm1d(num_hidden2)
        # hidden layer
        self.linear3 = nn.Linear(num_hidden2, num_hidden3)
        self.batchnorm3 = nn.BatchNorm1d(num_hidden3)
        # hidden layer
        self.linear4 = nn.Linear(num_hidden3, num_output)
        
        # define activation function in constructor
        self.activation = torch.nn.LeakyReLU()

    def forward(self, x):

        x = self.batchnorm1(self.linear1(x))
        x = self.dropout(self.activation(x))

        x = self.batchnorm2(self.linear2(x))
        x = self.dropout(self.activation(x))

        x = self.batchnorm3(self.linear3(x))
        x = self.dropout(self.activation(x))

        x = self.linear4(x)

        return x