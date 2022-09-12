import torch
import torch.nn as nn
import torch.nn.init as init
import logging

"""
This module contains the neural network architecture for the learning model.
"""

__all__ = ['SqueezeNet']

class Fire(nn.Module):
    """fire module for SqueezeNet"""
    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__() # same as super().__init__()?
        self.inplanes = inplanes
        self.squeeze = nn.Conv3d(inplanes, squeeze_planes, kernel_size=1) #1x1 convolution, inplanes = input feature maps, squeeze_planes = output feature maps
        self.squeeze_ReLU = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv3d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1) #1x1 convolution, squeeze_planes = input feature maps, expand1x1_planes = output feature maps
        self.expand1x1_ReLU = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv3d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1) #3x3 convolution, squeeze_planes = input feature maps, expand3x3_planes = output feature maps
        self.expand3x3_ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        method forward(input) returns output
        """
        x = self.squeeze_ReLU(self.squeeze(x)) #wrapping ReLU over output of 1x1 convolution squeeze layer
        return torch.cat([
            self.expand1x1_ReLU(self.expand1x1(x)),
            self.expand3x3_ReLU(self.expand3x3(x))
        ], 1) #concatenating output of expand layers (1x1 and 3x3 convolutions)


class SqueezeNet(nn.Module):

    def __init__(self, num_classes=1): #binary classification, so num_classes = 1
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
                nn.Conv3d(2, 96, kernel_size=2, stride=1), # 2 input channels (event_map, residue mask), 96 output channels, 7x7x7 kernel size, stride 2
                nn.ReLU(inplace=True), # inplace used to save memory
                nn.MaxPool3d(kernel_size=2, stride=1, ceil_mode=True), # 3x3x3 kernel size, stride 2, ceil_mode adds padding if necessary so that output size not reduced: https://stackoverflow.com/questions/59906456/in-pytorchs-maxpool2d-is-padding-added-depending-on-ceil-mode
                Fire(96, 16, 64, 64), # 96 input feature maps, squeeze layer squeezes input to 16 feature maps, 64+64=128 output feature maps (64 from 1x1 conv, 64 from 3x3 conv in expand layers)
                Fire(128, 16, 64, 64), # 128 input feature maps, 128 output feature maps
                Fire(128, 32, 128, 128), #128 input feature maps, 256 output feature maps
                nn.MaxPool3d(kernel_size=2, stride=1, ceil_mode=True),
                Fire(256, 32, 128, 128), #256 input feature maps, 256 output feature maps
                Fire(256, 48, 192, 192), #256 input feature maps, 384 output feature maps
                Fire(384, 48, 192, 192), #384 input feature maps, 384 output feature maps
                Fire(384, 64, 256, 256), #384 input feature maps, 512 output feature maps
                nn.MaxPool3d(kernel_size=2, stride=1, ceil_mode=True),
                Fire(512, 64, 256, 256), #512 input feature maps, 512 output feature maps
            )

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv3d(512, self.num_classes, kernel_size=1) #binary classification, so num_classes = 1, so only one output channel
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1,1,1)) 
            #average pooling where hyperparameters (stride, kernel size) are automatically adjusted
            #only output size is specified — output size is (1, 1, 1, 1, 1) (N, C, D, W, H) in this case — i.e. one sample. one channel, 1x1x1 tensor/scalar
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        self.act = nn.Sigmoid() #sigmoid activation function for binary classifier

    def forward(self, x):
        x = self.features(x)
        # logging.debug(f'features={x}')
        x = self.classifier(x)
        # logging.debug(x)

        # x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4]) #reshapes tensor
        x = x.view(-1)
        # logging.debug(x)
        # x = x.item() #converts tensor to scalar

        return self.act(x)