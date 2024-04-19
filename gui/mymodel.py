import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ConvolutionUnit(nn.Module):
    def __init__(self, ins, outs, kernel_size=3, dilation_rate=1, dropout_rate=0.5):
        super(ConvolutionUnit, self).__init__()
        padding = ((kernel_size - 1) * dilation_rate) // 2
        self.conv1 = nn.Conv2d(ins, outs, kernel_size=kernel_size, padding=padding, dilation=dilation_rate)
        self.bn1 = nn.BatchNorm2d(outs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(dropout_rate)
        self.conv2 = nn.Conv2d(outs, outs, kernel_size=kernel_size, padding=padding, dilation=dilation_rate)
        self.bn2 = nn.BatchNorm2d(outs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        return x

class DownSamplingBlock(nn.Module):
    def __init__(self, channels=[3, 64, 128, 256, 512, 1024], dilation_rates=[1, 1, 1, 1, 1]):
        super(DownSamplingBlock, self).__init__()

        self.encoder_layers = nn.ModuleList([
            ConvolutionUnit(channels[i], channels[i+1], dilation_rate=dilation_rates[i]) for i in range(len(channels)-1)
        ])
        self.down_sampling_layer = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        encoded_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encoded_features.append(x)
            x = self.down_sampling_layer(x)
        return encoded_features

class UpSamplingBlock(nn.Module):
    def __init__(self, channels=[1024, 512, 256, 128, 64]):
        super(UpSamplingBlock, self).__init__()

        self.decoder_layers = nn.ModuleList([
            ConvolutionUnit(channels[i], channels[i+1]) for i in range(len(channels) - 1)
        ])

        self.up_sampling_layers = nn.ModuleList([
            nn.ConvTranspose2d(channels[i], channels[i+1], kernel_size=2, stride=2) for i in range(len(channels) - 1)
        ])

    def forward(self, x, encoded_features):
        for i in range(len(self.decoder_layers)):
            x = self.up_sampling_layers[i](x)
            # Resize or crop the encoded feature to match x's size
            encoded_feature = F.interpolate(encoded_features[i], size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, encoded_feature], dim=1)
            x = self.decoder_layers[i](x)
        return x

class Unet(nn.Module):
    def __init__(self, encoder_channels=[3, 64, 128, 256, 512, 1024], decoder_channels=[1024, 512, 256, 128, 64], num_classes=1, out_size=(512, 512)):
        super(Unet, self).__init__()

        self.out_size = out_size
        self.down_sampler = DownSamplingBlock(encoder_channels)
        self.up_sampler = UpSamplingBlock(decoder_channels)
        self.final_layer = nn.Conv2d(in_channels=decoder_channels[-1], out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        encoded_features = self.down_sampler(x)
        # Reverse and convert to list
        reversed_features = list(reversed(encoded_features[:-1]))
        decoded_features = self.up_sampler(encoded_features[-1], reversed_features)
        output = self.final_layer(decoded_features)
        output = F.interpolate(output, self.out_size)
        return output