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
        self.num_hidden2 = num_hidden3
        self.num_output = num_output
        self.dropout_p = dropout_p
        # input layer
        self.W_1 = Parameter(init.kaiming_normal_(torch.Tensor(num_hidden1, num_features)))
        self.b_1 = Parameter(init.constant_(torch.Tensor(num_hidden1), 0))
        self.batchnorm1 = nn.BatchNorm1d(num_hidden1)
        self.dropout = nn.Dropout(p=dropout_p)
        # hidden layer
        self.W_2 = Parameter(init.kaiming_normal_(torch.Tensor(num_hidden2, num_hidden1)))
        self.b_2 = Parameter(init.constant_(torch.Tensor(num_hidden2), 0))
        self.batchnorm2 = nn.BatchNorm1d(num_hidden2)
        # hidden layer
        self.W_3 = Parameter(init.kaiming_normal_(torch.Tensor(num_hidden3, num_hidden2)))
        self.b_3 = Parameter(init.constant_(torch.Tensor(num_hidden3), 0))
        self.batchnorm3 = nn.BatchNorm1d(num_hidden3)
        # hidden layer
        self.W_4 = Parameter(init.kaiming_normal_(torch.Tensor(num_output, num_hidden3)))
        self.b_4 = Parameter(init.constant_(torch.Tensor(num_output), 0))
        
        # define activation function in constructor
        self.activation = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.batchnorm1(F.linear(x, self.W_1, self.b_1))
        x = self.dropout(self.activation(x))

        x = self.batchnorm2(F.linear(x, self.W_2, self.b_2))
        x = self.dropout(self.activation(x))

        x = self.batchnorm3(F.linear(x, self.W_3, self.b_3))
        x = self.dropout(self.activation(x))

        x = F.linear(x, self.W_4, self.b_4)
        return x