#! /usr/bin/env python3

import torch
from more_itertools import pairwise


class ResidualBlock(torch.nn.Module):

    def __init__(self, size, activation='ReLU', activation_params={}):
        super().__init__()

        self.input_size = size
        self.output_size = size

        self.activation = torch.nn.__dict__[activation](**activation_params)
        self.fc1 = torch.nn.Linear(self.input_size, self.output_size)
        self.fc2 = torch.nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)

        y = self.fc2(y)
        y = self.activation(y)

        # the "residual" part
        y += x

        return y


class LinearBlock(torch.nn.Module):

    def __init__(self, input_size, output_size,
                 activation='ReLU', activation_params={}):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.activation = torch.nn.__dict__[activation](**activation_params)
        self.fc = torch.nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)

        return x


class ConvBlock(torch.nn.Module):

    def __init__(self,
                 input_size, output_size,
                 kernel_size, stride, padding,
                 maxpool_kernel_size, maxpool_stride, maxpool_padding,
                 activation='ReLU', activation_params={}):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.maxpool_padding = maxpool_padding

        self.activation = torch.nn.__dict__[activation](**activation_params)

        self.conv = torch.nn.Conv2d(in_channels=self.input_size,
                                    out_channels=self.output_size,
                                    kernel_size=self.kernel_size,
                                    stride=self.stride,
                                    padding=self.padding,
                                    )
        self.maxpool = torch.nn.MaxPool2d(kernel_size=self.maxpool_kernel_size,
                                          stride=self.maxpool_stride,
                                          padding=self.maxpool_padding,
                                          )

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.activation(x)

        return x


class CNNModel(torch.nn.Module):

    def __init__(self,
                 batch_norm=True,
                 activation='ReLU',
                 activation_params={},
                 ):
        super().__init__()

        # pure unadultered genius
        self.activation = torch.nn.__dict__[activation](**activation_params)

        self.bn_activated = batch_norm

        # first convolution section
        self.conv_1 = torch.nn.Conv2d(in_channels=1,
                                      out_channels=1,
                                      kernel_size=3,
                                      stride=1,
                                      padding=0,
                                      )
        self.maxpool_1 = torch.nn.MaxPool2d(kernel_size=2,
                                            stride=1,
                                            padding=0,
                                            )

        # second convolution section
        self.conv_2 = torch.nn.Conv2d(in_channels=1,
                                      out_channels=1,
                                      kernel_size=3,
                                      stride=1,
                                      padding=0,
                                      )
        self.maxpool_2 = torch.nn.MaxPool2d(kernel_size=2,
                                            stride=1,
                                            padding=0,
                                            )

        if self.bn_activated:
            self.norm_conv = torch.nn.BatchNorm2d(num_features=1)
        else:
            self.norm_conv = torch.nn.Identity()

        # linear section
        self.linear = torch.nn.Linear(in_features=24, out_features=2)

        if self.bn_activated:
            self.norm_linear = torch.nn.BatchNorm1d(num_features=1)
        else:
            self.norm_linear = torch.nn.Identity()

        # softmax
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.activation(x)
        x = self.maxpool_1(x)

        x = self.conv_2(x)
        x = self.activation(x)
        x = self.norm_conv(x)
        x = self.maxpool_2(x)
        x = x.reshape(-1, 1, 24)

        x = self.linear(x)
        x = self.activation(x)
        x = self.norm_linear(x)

        x = self.softmax(x)

        x = x.squeeze(1)

        return x


class FCNN(torch.nn.Module):

    def __init__(self,
                 input_size=120,
                 activation='ReLU',
                 activation_params={},
                 hidden_sizes=[200, 100, 50],
                 output_size=2,
                 ):
        super().__init__()

        # dynamic linear stuff
        self.input_size = input_size
        self.output_size = output_size

        hidden_sizes.insert(0, self.input_size)
        hidden_sizes.append(self.output_size)

        self.linear_blocks = []

        for input_size, output_size in pairwise(hidden_sizes):
            block = LinearBlock(input_size,
                                output_size,
                                activation,
                                activation_params)
            self.linear_blocks.append(block)

        # softmax
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        # flatten but keep batch size
        x = x.flatten(start_dim=1)

        for block in self.linear_blocks:
            x = block(x)

        x = self.softmax(x)

        return x


if __name__ == '__main__':

    # dummy data for testing
    # (n_samples, 12, 10)
    t = torch.rand(size=(5, 12, 10))

    mod = FCNN()

    res = mod(t)

    print(res.shape)
    print(res)
