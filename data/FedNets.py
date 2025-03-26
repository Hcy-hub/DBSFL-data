#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.cuda.manual_seed(1)

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        print("NN: MLP is created")
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)

class MLP1(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP1, self).__init__()
        print("NN: MLP is created")
        self.layer_input = nn.Linear(dim_in, dim_hidden)#784,128
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)#128,10

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.layer_hidden(x)
        return F.log_softmax(x, dim=1)

class MLP_triple(nn.Module):
    def __init__(self, dim_in, dim_hidden1, dim_hidden2,dim_out):
        super(MLP_triple, self).__init__()
        print("NN: MLP is created")
        self.layer_input = nn.Linear(dim_in, dim_hidden1)
        self.layer_hidden = nn.Linear(dim_hidden1, dim_hidden2)
        self.layer_out = nn.Linear(dim_hidden2, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.layer_hidden(x)
        x = self.layer_out(x)
        return F.log_softmax(x, dim=1)

class MLP_triple_SVD(nn.Module):
    def __init__(self, dim_in, dim_hidden1,dim12_sr,dim12_sc, dim_hidden2,dim_out):
        super(MLP_triple_SVD, self).__init__()
        print("NN: MLP is created")
        self.layer_input = nn.Linear(dim_in, dim_hidden1)
        self.layer_hidden1 = nn.Linear(dim_hidden1, dim12_sr)
        self.layer_hidden2 = nn.Linear(dim12_sr, dim12_sc)
        self.layer_hidden3 = nn.Linear(dim12_sc, dim_hidden2)
        self.layer_out = nn.Linear(dim_hidden2, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.layer_hidden1(x)
        x = self.layer_hidden2(x)
        x = self.layer_hidden3(x)
        x = self.layer_out(x)
        return F.log_softmax(x, dim=1)

class MLP_regression(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP_regression, self).__init__()
        print("NN: MLP is created")
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.layer_hidden(x)
        return F.log_softmax(x, dim=1)

class CNN_test(nn.Module):
    def __init__(self, args):
        super(CNN_test, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32* 7 * 7, args.num_classes)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return F.log_softmax(output, dim=1)                    # return x for visualization


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        print("NN: CNNMnist is created")
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNMnistWithMask(nn.Module):
    def __init__(self, args):
        super(CNNMnistWithMask, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv1_mask = nn.Parameter(torch.ones_like(self.conv1.weight))

        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_mask = nn.Parameter(torch.ones_like(self.conv2.weight))

        self.conv2_drop = nn.Dropout2d()

        # 初始化全连接层并添加掩码
        self.fc1 = nn.Linear(320, 50)
        self.fc1_mask = nn.Parameter(torch.ones_like(self.fc1.weight))

        self.fc2 = nn.Linear(50, args.num_classes)
        self.fc2_mask = nn.Parameter(torch.ones_like(self.fc2.weight))

    def forward(self, x):

        x = F.conv2d(x, self.conv1.weight * self.conv1_mask, bias=self.conv1.bias, stride=1)
        x = F.relu(F.max_pool2d(x, 2))

        x = F.conv2d(x, self.conv2.weight * self.conv2_mask, bias=self.conv2.bias, stride=1)
        x = F.relu(F.max_pool2d(self.conv2_drop(x), 2))

        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])


        x = F.linear(x, self.fc1.weight * self.fc1_mask, bias=self.fc1.bias)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = F.linear(x, self.fc2.weight * self.fc2_mask, bias=self.fc2.bias)
        return F.log_softmax(x, dim=1)


# class CNNCifar(nn.Module):
#     def __init__(self, args):
#         super(CNNCifar, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, args.num_classes)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1)

class CNNCifar(nn.Module):

    def __init__(self, args):
        super(CNNCifar, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (64 * 4 * 4 -> 500)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        # linear layer (500 -> 10)
        self.fc2 = nn.Linear(500, args.num_classes)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image input
        x = x.view(-1, 64 * 4 * 4)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNFashionMnist(nn.Module):
    def __init__(self,args):
        super(CNNFashionMnist, self).__init__()
        print("NN: CNNFashionMnist is created")
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(800, 128)
        self.fc2 = nn.Linear(128, args.num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



class VGG(nn.Module):
    cfg = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
            }
    def __init__(self, args):
        super(VGG, self).__init__()
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        self.features = self._make_layers(cfg)
        self.classifier = nn.Linear(512, 10)
    

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


# adult_net 
class MLP3(nn.Module):
    def __init__(self,args):
        super(MLP3, self).__init__()
        print("NN:  adult_net MLP  is created")
        self.l1 = nn.Linear(10,64)
        self.l2 = nn.Linear(64,32)
        
        self.l3 = nn.Linear(32,2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
