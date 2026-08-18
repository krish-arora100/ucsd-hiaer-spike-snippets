"""
Taken from spikingjelly tutorial for classifying dvs gestures: https://spikingjelly.readthedocs.io/zh-cn/latest/activation_based_en/classify_dvsg.html
"""

# spikingjelly.activation_based.model.parametric_lif_net

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from copy import deepcopy

#keli's DVSGestureNet from original paper, with ability to change encoder
class DVSGestureNet(nn.Module):
    def __init__(self, channels=128, encoder = 3, out_features = 512, spiking_neuron: callable = None, input_shape = (16, 2, 128, 128), **kwargs):
        super().__init__()

        B, C, H, W = input_shape

        conv = []
        for i in range(encoder):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            if H > 3 and W > 3: #don't want to reduce spatial dims to 1x1 which fails batchnorm
                conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, stride = 2, padding=0, bias=False))
                conv.append(layer.BatchNorm2d(channels))
                conv.append(spiking_neuron(**deepcopy(kwargs)))
                H = H // 2
                W = W // 2
            
            else:
                conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
                conv.append(layer.BatchNorm2d(channels))
                conv.append(spiking_neuron(**deepcopy(kwargs)))
            
            print(H)
            print(W)

        conv_seq = nn.Sequential(*conv)
        #print(conv_seq)
        B, C, H, W = input_shape
        dummy_input = torch.zeros((B, C, H, W))

        with torch.no_grad():
            x_out = conv_seq(dummy_input)
            #flatten but ignore batch size
            in_features = x_out.flatten(start_dim=1).shape[1]  # Flatten and get feature dim
            print(x_out.shape)

        print("Input features to first linear layer:", in_features)    

        self.conv_fc = nn.Sequential(
            *conv,
            
            layer.Flatten(),
            layer.Dropout(0.5), #default 0.5
            layer.Linear(in_features, out_features),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5), #default 0.5
            layer.Linear(out_features, 11),
            spiking_neuron(**deepcopy(kwargs)),

        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)

#default from spikingjelly

class VotingLayer(nn.Module):
    def __init__(self, voter_num: int):
        super().__init__()
        self.voting = nn.AvgPool1d(voter_num, voter_num)
    def forward(self, x: torch.Tensor):
        # x.shape = [N, voter_num * C]
        # ret.shape = [N, C]
        return self.voting(x.unsqueeze(1)).squeeze(1)


class DVSGestureNetSpikingJelly(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(**deepcopy(kwargs)))
            conv.append(layer.MaxPool2d(2, 2))


        self.conv_fc = nn.Sequential(
            *conv,

            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 4 * 4, 512),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5),
            layer.Linear(512, 110),
            spiking_neuron(**deepcopy(kwargs)),

            VotingLayer(10)
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)

class DVSGestureNetAvgPool(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            #conv.append(layer.AvgPool2d(2, 2))
            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(layer.AvgPool2d(2, 2))
            conv.append(spiking_neuron(**deepcopy(kwargs)))



        self.conv_fc = nn.Sequential(
            *conv,

            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 4 * 4, 512),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5),
            layer.Linear(512, 110),
            spiking_neuron(**deepcopy(kwargs)),

            VotingLayer(10)
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)

class DVSGestureNetAvgPoolNoVoting(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            #conv.append(layer.AvgPool2d(2, 2))
            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(layer.AvgPool2d(2, 2))
            conv.append(spiking_neuron(**deepcopy(kwargs)))



        self.conv_fc = nn.Sequential(
            *conv,

            layer.Flatten(),
            #layer.Dropout(0.5),
            layer.Linear(channels * 4 * 4, 512),
            spiking_neuron(**deepcopy(kwargs)),

            #layer.Dropout(0.5),
            layer.Linear(512, 11),
            spiking_neuron(**deepcopy(kwargs)),
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)

class DVSGestureNetLinear(nn.Module):
    def __init__(self, channels=4, spiking_neuron: callable = None, input_shape=(10, 2, 128, 128), **kwargs):
        super().__init__()
        t, c, h, w = input_shape
        self.net = nn.Sequential(
            layer.Flatten(),
            layer.Linear(c*h*w, 16384),  # 2 * 128 * 128 = 32768
            spiking_neuron(**deepcopy(kwargs)),
            layer.Linear(16384, 512),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Linear(512, 11),
            spiking_neuron(**deepcopy(kwargs))
        )

    def forward(self, x):
        return self.net(x)

class DVSGestureNetAvgPoolSimple(nn.Module):
    def __init__(self, channels=4, spiking_neuron: callable = None, input_shape=(10, 2, 128, 128), **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            layer.Conv2d(2, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            layer.AvgPool2d(2, 2),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Flatten(),
            layer.Linear(channels * 64 * 64, 11),
            spiking_neuron(**deepcopy(kwargs)),
        )

    def forward(self, x):
        return self.net(x)

class DVSGestureNetAvgPoolSimple2(nn.Module):
    def __init__(self, channels=4, spiking_neuron: callable = None, input_shape=(10, 2, 128, 128), **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            layer.Conv2d(2, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            layer.AvgPool2d(2, 2),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            layer.AvgPool2d(2, 2),
            spiking_neuron(**deepcopy(kwargs)),


            layer.Flatten(),

            layer.Linear(channels * 32 * 32, 512),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Linear(512, 11),
            spiking_neuron(**deepcopy(kwargs))
        )

    def forward(self, x):
        return self.net(x)

class DVSGestureNetAvgPoolSimple3(nn.Module):
    def __init__(self, channels=4, spiking_neuron: callable = None, input_shape=(10, 2, 128, 128), **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            layer.Conv2d(2, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            layer.AvgPool2d(2, 2),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Flatten(),
            layer.Linear(channels * 64 * 64, 4096),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Linear(4096, 1024),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Linear(1024, 512),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Linear(512, 11),
            spiking_neuron(**deepcopy(kwargs))
        )

    def forward(self, x):
        return self.net(x)

class DVSGestureNetAvgPoolSimple4(nn.Module):
    def __init__(self, channels=4, spiking_neuron: callable = None, input_shape=(10, 2, 128, 128), **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            layer.Conv2d(2, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            layer.AvgPool2d(2, 2),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            layer.AvgPool2d(2, 2),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Flatten(),

            layer.Linear(channels * 32 * 32, 4096),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Linear(4096, 1024),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Linear(1024, 512),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Linear(512, 11),
            spiking_neuron(**deepcopy(kwargs))
        )

    def forward(self, x):
        return self.net(x)

class DVSGestureNetAvgPoolSimple5(nn.Module):
    def __init__(self, channels=4, spiking_neuron: callable = None, input_shape=(10, 2, 128, 128), **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            layer.Conv2d(2, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            layer.AvgPool2d(2, 2),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            layer.AvgPool2d(2, 2),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Flatten(),

            layer.Linear(channels * 32 * 32, 2048),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Linear(2048, 512),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Linear(512, 11),
            spiking_neuron(**deepcopy(kwargs))
        )

    def forward(self, x):
        return self.net(x)

class DVSGestureNetAvgPoolSimple6(nn.Module):
    def __init__(self, channels=4, spiking_neuron: callable = None, input_shape=(10, 2, 128, 128), **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            layer.Conv2d(2, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            layer.AvgPool2d(2, 2),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            layer.AvgPool2d(2, 2),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            layer.AvgPool2d(2, 2),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Flatten(),

            layer.Linear(channels * 16 * 16, 4096),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Linear(4096, 1024),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Linear(1024, 512),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Linear(512, 11),
            spiking_neuron(**deepcopy(kwargs))
        )

    def forward(self, x):
        return self.net(x)


#default with additional linear block
class DVSGestureNet2(nn.Module):
    def __init__(self, channels=20, spiking_neuron: callable = None, *args, **kwargs):
        super().__init__()

        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(layer.MaxPool2d(2, 2))
            conv.append(spiking_neuron(*args, **kwargs))


        self.conv_fc = nn.Sequential(
           *conv,

            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 4 * 4, 512),
            spiking_neuron(*args, **kwargs),

            layer.Dropout(0.5),
            layer.Linear(512, 256),
            spiking_neuron(*args, **kwargs),

            layer.Dropout(0.5),
            layer.Linear(256, 110),
            spiking_neuron(*args, **kwargs),

            VotingLayer(10)
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)


class DVSGestureNetAvgPoolSimple7(nn.Module):
    def __init__(self, channels=4, spiking_neuron: callable = None, input_shape=(10, 2, 128, 128), **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            layer.Conv2d(2, channels, kernel_size=3, stride=2, padding=1, bias=False),
            #layer.BatchNorm2d(channels),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Flatten(),
            layer.Linear(channels * 64 * 64, 11),
            spiking_neuron(**deepcopy(kwargs)),
        )

    def forward(self, x):
        return self.net(x)
    

#default with additional conv layer in each block
class DVSGestureNet3(nn.Module):
    def __init__(self, channels=20, spiking_neuron: callable = None, *args, **kwargs):
        super().__init__()

        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(layer.MaxPool2d(2, 2))
            conv.append(spiking_neuron(*args, **kwargs))

            conv.append(layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(*args, **kwargs))



        self.conv_fc = nn.Sequential(
            *conv,

            layer.Flatten(),

            layer.Dropout(0.5),
            layer.Linear(channels * 4 * 4, 512),
            spiking_neuron(*args, **kwargs),

            layer.Dropout(0.5),
            layer.Linear(512, 110),
            spiking_neuron(*args, **kwargs),

            VotingLayer(10)
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)
    


#no max pool for following
class DVSGestureNet4(nn.Module):
    def __init__(self, channels=128, encoder = 3, out_features = 512, spiking_neuron: callable = None, input_shape = (16, 2, 128, 128), **kwargs):
        super().__init__()
        B, C, H, W = input_shape

        conv = []
        for i in range(encoder):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            if H > 3 and W > 3: #don't want to reduce spatial dims to 1x1 which fails batchnorm
                conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, stride = 2, padding=1, bias=False))
                conv.append(layer.BatchNorm2d(channels))
                conv.append(spiking_neuron(**deepcopy(kwargs)))
                H = H // 2
                W = W // 2
            
            else:
                conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
                conv.append(layer.BatchNorm2d(channels))
                conv.append(spiking_neuron(**deepcopy(kwargs)))
            
            print(H)
            print(W)

        conv_seq = nn.Sequential(*conv)
        B, C, H, W = input_shape
        dummy_input = torch.zeros((B, C, H, W))

        with torch.no_grad():
            x_out = conv_seq(dummy_input)
            #flatten but ignore batch size
            in_features = x_out.flatten(start_dim=1).shape[1]  # Flatten and get feature dim
            print(x_out.shape)

        print("Input features to first linear layer:", in_features)    

        self.conv_fc = nn.Sequential(
            *conv,
            
            layer.Flatten(),
            layer.Dropout(0.5), #default 0.5
            layer.Linear(in_features, out_features),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5), #default 0.5
            layer.Linear(out_features, 11),
            spiking_neuron(**deepcopy(kwargs)),

        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)
    

#DVSGestureNet4 with ability to change number of fc layers
class DVSGestureNet5(nn.Module):
    def __init__(self, channels=128, encoder = 3, num_fc_layers = 2, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        conv = []
        for i in range(encoder):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, stride = 2, padding=0, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(**deepcopy(kwargs)))

        fc = []
        fc.append(layer.Flatten())

        conv_seq = nn.Sequential(*conv)
        T, C, H, W = input_shape
        dummy_input = torch.zeros((1, C, H, W))

        with torch.no_grad():
            x_out = conv_seq(dummy_input)
            in_features = x_out.view(1, -1).shape[1]  # Flatten and get feature dim

        print("Input features to first linear layer:", in_features)

        out_features = 512

        for i in range(num_fc_layers - 1):
            fc.append(layer.Dropout(0.5))
            fc.append(layer.Linear(in_features, out_features))
            fc.append(spiking_neuron(**deepcopy(kwargs)))
            in_features = out_features
            out_features = out_features // 2

        fc.append(layer.Dropout(0.5))
        fc.append(layer.Linear(in_features, 11)) #11 classes for DVS Gesture
        fc.append(spiking_neuron(**deepcopy(kwargs)))

        self.conv_fc = nn.Sequential(
            *conv,
            *fc
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)
    


#DVSGestureNet4 with support for additional conv layers in each block
class DVSGestureNet6(nn.Module):
    def __init__(self, channels=128, encoder=3, num_conv_layers_per_block=2, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        conv = []
        for i in range(encoder):
            in_channels = 2 if i == 0 else channels

            for j in range(num_conv_layers_per_block):
                stride = 2 if j == 0 else 1
                padding = 0 if j == 0 else 1  #keep spatial dims for additional convs

                conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=padding, bias=False))
                conv.append(layer.BatchNorm2d(channels))
                conv.append(spiking_neuron(**deepcopy(kwargs)))

                in_channels = channels  

        conv_seq = nn.Sequential(*conv)
        T, C, H, W = input_shape
        dummy_input = torch.zeros((1, C, H, W))

        with torch.no_grad():
            x_out = conv_seq(dummy_input)
            in_features = x_out.view(1, -1).shape[1]  # Flatten and get feature dim

        print("Input features to first linear layer:", in_features)

        self.conv_fc = nn.Sequential(
            *conv,
            
            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(in_features, 512),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5),
            layer.Linear(512, 11),
            spiking_neuron(**deepcopy(kwargs)),

        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)
    
#DVSGestureNet 4 with avgpool layer instead of stride 2
class DVSGestureNet7(nn.Module):
    def __init__(self, channels=128, encoder = 3, out_features = 512, spiking_neuron: callable = None, input_shape = (16, 2, 128, 128), **kwargs):
        super().__init__()

        B, C, H, W = input_shape

        conv = []
        for i in range(encoder):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            if H > 3 and W > 3: #don't want to reduce spatial dims to 1x1 which fails batchnorm
                conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
                conv.append(layer.BatchNorm2d(channels))
                conv.append(layer.AvgPool2d(2, 2))
                conv.append(spiking_neuron(**deepcopy(kwargs)))

                H = H // 2
                W = W // 2
            
            else:
                conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
                conv.append(layer.BatchNorm2d(channels))
                conv.append(spiking_neuron(**deepcopy(kwargs)))


        conv_seq = nn.Sequential(*conv)
        B, C, H, W = input_shape
        dummy_input = torch.zeros((B, C, H, W))

        with torch.no_grad():
            x_out = conv_seq(dummy_input)
            in_features = x_out.flatten(start_dim=1).shape[1]  # Flatten and get feature dim
            print(x_out.shape)

        print("Input features to first linear layer:", in_features)    

        self.conv_fc = nn.Sequential(
            *conv,
            
            layer.Flatten(),
            layer.Dropout(0.5), #default 0.5
            layer.Linear(in_features, out_features),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5), #default 0.5
            layer.Linear(out_features, 11),
            spiking_neuron(**deepcopy(kwargs)),

        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)

    