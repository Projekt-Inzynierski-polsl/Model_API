import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights


class CNNEncoder(nn.Module):
    def __init__(self, size, num_channels, dims, cnn_model, device):
        super(CNNEncoder, self).__init__()
        self.size = size
        self.num_channels = num_channels
        self.dims = dims
        self.cnn_model = cnn_model
        self.device = device

        if self.cnn_model == 0:
            self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
            if self.num_channels == 1:
                self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                                           padding=(3, 3), bias=False)
            self.encode = self.rnn_encoder
        elif self.cnn_model == 1:
            self.cnn = convnext_small(weigths=ConvNeXt_Small_Weights)
            if self.num_channels == 1:
                conv_layer = self.cnn.features[0][0]
                self.cnn.features[0][0] = nn.Conv2d(1, conv_layer.out_channels,
                                                    kernel_size=conv_layer.kernel_size, stride=conv_layer.stride,
                                                    padding=conv_layer.padding, bias=conv_layer.bias is not None)
            self.encode = self.convnext_encoder
        elif self.cnn_model == 2:
            self.cnn = efficientnet_b4(weights=EfficientNet_B4_Weights)
            if self.num_channels == 1:
                conv_layer = self.cnn.features[0][0]
                self.cnn.features[0][0] = nn.Conv2d(1, conv_layer.out_channels,
                                                    kernel_size=conv_layer.kernel_size, stride=conv_layer.stride,
                                                    padding=conv_layer.padding, bias=conv_layer.bias is not None)
            self.encode = self.efficientnet_encoder
        elif self.cnn_model == 3:
            self.cnn = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights)
            if self.num_channels == 1:
                conv_layer = self.cnn.conv1
                self.cnn.conv1 = nn.Conv2d(1, conv_layer.out_channels, kernel_size=conv_layer.kernel_size,
                                           stride=conv_layer.stride, padding=conv_layer.padding,
                                           bias=conv_layer.bias is not None)
            self.encode = self.resnext_encoder

        cnn_input_size = self.encode(torch.rand(1, self.num_channels, self.size[0], self.size[1]))
        self.linear = nn.Sequential(
            nn.Linear(cnn_input_size.shape[-1], self.dims),
            nn.GELU(),
            nn.Dropout(0.5)
        )

    def rnn_encoder(self, x):
        x = self.cnn.conv1(x)
        x = self.cnn.bn1(x)
        x = self.cnn.relu(x)
        x = self.cnn.maxpool(x)
        x = self.cnn.layer1(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        return x

    def convnext_encoder(self, x):
        x = self.cnn.features[0](x)
        x = self.cnn.features[1](x)
        x = self.cnn.features[2](x)
        x = self.cnn.features[3](x)
        x = self.cnn.features[4](x)
        x = self.cnn.features[5](x)
        x = x.view(x.size(0), x.size(1), -1)
        return x

    def efficientnet_encoder(self, x):
        x = self.cnn.features[0](x)
        x = self.cnn.features[1](x)
        x = self.cnn.features[2](x)
        x = self.cnn.features[3](x)
        x = self.cnn.features[4](x)
        x = self.cnn.features[5](x)
        x = self.cnn.features[6](x)
        x = x.view(x.size(0), x.size(1), -1)
        return x

    def resnext_encoder(self, x):
        x = self.cnn.conv1(x)
        x = self.cnn.bn1(x)
        x = self.cnn.relu(x)
        x = self.cnn.maxpool(x)
        x = self.cnn.layer1(x)
        x = self.cnn.layer2(x)
        x = self.cnn.layer3(x)
        x = x.view(x.size(0), x.size(1), -1)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.linear(x)
        return x
