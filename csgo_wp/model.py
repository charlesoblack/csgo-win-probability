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


class ResNet(torch.nn.Module):

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

        self.blocks = []

        for input_size, output_size in pairwise(hidden_sizes):
            block = ResidualBlock(input_size,
                                  activation,
                                  activation_params)
            self.blocks.append(block)

            block = LinearBlock(input_size,
                                output_size,
                                activation,
                                activation_params)
            self.blocks.append(block)

        # softmax
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        # flatten but keep batch size
        x = x.flatten(start_dim=1)

        for block in self.blocks:
            x = block(x)

        x = self.softmax(x)

        return x


class CNN(torch.nn.Module):

    def __init__(self,
                 input_size=(12, 10),
                 output_size=2,
                 options=((1, 1, 3, 1, 0, 2, 1, 0),),
                 activation='ReLU',
                 activation_params={},
                 ):
        super().__init__()

        # dynamic linear stuff
        self.input_size = input_size
        self.output_size = output_size

        self.conv_blocks = []

        block_output_size = input_size

        for option_set in options:
            block = ConvBlock(*option_set,
                              activation,
                              activation_params)
            self.conv_blocks.append(block)

            # conv
            block_output_size = ((block_output_size[0]
                                  + 2 * option_set[4]
                                  - option_set[2]) // option_set[3] + 1,
                                 (block_output_size[1]
                                  + 2 * option_set[4]
                                  - option_set[2]) // option_set[3] + 1,
                                 )

            # pooling
            block_output_size = ((block_output_size[0]
                                  + 2 * option_set[7]
                                  - option_set[5]) // option_set[6] + 1,
                                 (block_output_size[1]
                                  + 2 * option_set[7]
                                  - option_set[5]) // option_set[6] + 1,
                                 )

            n_channels = option_set[1]

        self.num_elements_output = int(block_output_size[0]
                                       * block_output_size[1]
                                       * n_channels)

        # linear section
        self.linear = LinearBlock(self.num_elements_output,
                                  self.output_size,
                                  activation,
                                  activation_params)

        # softmax
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):

        x = x.unsqueeze(1)

        for block in self.conv_blocks:
            x = block(x)

        x = x.reshape(-1, self.num_elements_output)

        x = self.linear(x)
        x = self.softmax(x)

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

    print('Testing FCNN')

    mod = FCNN()

    res = mod(t).detach()

    print(res.shape)
    print(res)

    print('\nTesting CNN')

    mod = CNN()

    res = mod(t).detach()

    print(res.shape)
    print(res)

    print('\nTesting ResNet')

    mod = ResNet()

    res = mod(t).detach()

    print(res.shape)
    print(res)
