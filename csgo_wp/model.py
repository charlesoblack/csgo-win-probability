#! /usr/bin/env python3

import torch
import numpy as np
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
                 input_size=(1, 12, 10),
                 activation='ReLU',
                 activation_params={},
                 hidden_sizes=[200, 100, 50],
                 output_size=1,
                 batch_norm=False,
                 dropout=False,
                 cnn_options=tuple(),
                 ):
        super().__init__()

        # dynamic linear stuff
        self.input_size = np.prod(input_size)
        self.output_size = output_size

        self.bn_activated = batch_norm
        self.dropout_activated = dropout

        hidden_sizes.insert(0, self.input_size)
        hidden_sizes.append(self.output_size)

        self.blocks = torch.nn.ModuleList()

        for input_size, output_size in pairwise(hidden_sizes):
            # last unit: don't use activation
            if output_size == self.output_size:
                activation = 'Identity'
            block = ResidualBlock(input_size,
                                  activation,
                                  activation_params)
            self.blocks.append(block)

            block = LinearBlock(input_size,
                                output_size,
                                activation,
                                activation_params)
            self.blocks.append(block)

            if self.bn_activated:
                norm = torch.nn.BatchNorm1d(num_features=output_size)
                self.blocks.append(norm)

            if self.dropout_activated:
                dropout_block = torch.nn.Dropout()
                self.blocks.append(dropout_block)

        # sigmoid
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # flatten but keep batch size
        x = x.flatten(start_dim=1)

        for block in self.blocks:
            x = block(x)

        x = self.sigmoid(x)

        return x.squeeze(1)


class CNN(torch.nn.Module):

    def __init__(self,
                 input_size=(1, 12, 10),
                 output_size=1,
                 cnn_options=((1, 1, 3, 1, 0, 2, 1, 0),),
                 activation='ReLU',
                 activation_params={},
                 batch_norm=False,
                 dropout=False,
                 hidden_sizes=[],
                 ):
        super().__init__()

        # dynamic linear stuff
        self.input_size = input_size
        self.output_size = output_size

        self.bn_activated = batch_norm
        self.dropout_activated = dropout

        self.conv_blocks = torch.nn.ModuleList()

        block_output_size = input_size[1:]

        for option_set in cnn_options:
            block = ConvBlock(*option_set,
                              activation,
                              activation_params)
            self.conv_blocks.append(block)

            if self.bn_activated:
                self.norm_conv = torch.nn.BatchNorm2d(option_set[1])
                self.conv_blocks.append(self.norm_conv)
            else:
                self.norm_conv = torch.nn.Identity()

            if self.dropout_activated:
                dropout_block = torch.nn.Dropout2d()
                self.conv_blocks.append(dropout_block)

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

        # sigmoid
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):

        for block in self.conv_blocks:
            x = block(x)

        x = x.reshape(-1, self.num_elements_output)

        x = self.linear(x)
        x = self.sigmoid(x)

        return x.squeeze(1)


class FCNN(torch.nn.Module):

    def __init__(self,
                 input_size=(1, 12, 10),
                 activation='ReLU',
                 activation_params={},
                 hidden_sizes=[200, 100, 50],
                 output_size=1,
                 batch_norm=False,
                 dropout=False,
                 cnn_options=tuple(),
                 ):
        super().__init__()

        # dynamic linear stuff
        self.input_size = np.prod(input_size)
        self.output_size = output_size

        self.bn_activated = batch_norm
        self.dropout_activated = dropout

        hidden_sizes.insert(0, self.input_size)
        hidden_sizes.append(self.output_size)

        self.linear_blocks = torch.nn.ModuleList()

        for input_size, output_size in pairwise(hidden_sizes):
            # last unit: don't use activation
            if output_size == self.output_size:
                activation = 'Identity'
            block = LinearBlock(input_size,
                                output_size,
                                activation,
                                activation_params)
            self.linear_blocks.append(block)

            if self.bn_activated and output_size != self.output_size:
                norm = torch.nn.BatchNorm1d(num_features=output_size)
                self.linear_blocks.append(norm)

            if self.dropout_activated and output_size != self.output_size:
                dropout_block = torch.nn.Dropout()
                self.linear_blocks.append(dropout_block)

        # sigmoid to get win probability
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # flatten but keep batch size
        x = x.flatten(start_dim=1)

        for block in self.linear_blocks:
            x = block(x)

        x = self.sigmoid(x)

        return x.squeeze(1)


class LR_CNN(torch.nn.Module):

    def __init__(self,
                 input_size=(6, 5, 5),
                 activation='ReLU',
                 activation_params={},
                 hidden_sizes=[200, 100, 50],
                 output_size=1,
                 batch_norm=False,
                 dropout=False,  # for compatibility
                 cnn_options=((4, 6, 1, 1, 0, 1, 1, 0),),
                 ablation=None,
                 ):
        super().__init__()

        # dynamic linear stuff
        self.linear_input_size = 10
        self.cnn_input_size = input_size
        self.output_size = output_size
        self.ablation = ablation

        hidden_sizes.insert(0, self.linear_input_size)
        hidden_sizes.append(self.output_size)

        self.linear_blocks = torch.nn.ModuleList()
        self.conv_blocks = torch.nn.ModuleList()

        for input_size, output_size in pairwise(hidden_sizes):
            # last unit: don't use activation
            if output_size == self.output_size:
                activation = 'Identity'
            block = LinearBlock(input_size,
                                output_size,
                                activation,
                                activation_params)
            self.linear_blocks.append(block)

            if output_size != self.output_size:
                norm = torch.nn.BatchNorm1d(num_features=output_size)
                self.linear_blocks.append(norm)

            if output_size != self.output_size:
                dropout_block = torch.nn.Dropout()
                self.linear_blocks.append(dropout_block)

        block_output_size = self.cnn_input_size[1:]
        n_channels = 1

        for option_set in cnn_options:
            block = ConvBlock(*option_set,
                              activation,
                              activation_params)
            self.conv_blocks.append(block)

            self.norm_conv = torch.nn.BatchNorm2d(option_set[1])
            self.conv_blocks.append(self.norm_conv)

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

        # cnn linear section
        self.cnn_linear = LinearBlock(self.num_elements_output,
                                      self.output_size,
                                      activation,
                                      activation_params)

        # after the concat
        if ablation is not None:
            concat_size = 1
        else:
            concat_size = 2
        self.final_linear = LinearBlock(concat_size,
                                        1,
                                        activation,
                                        activation_params)

        # sigmoid
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):

        # divide input into cnn/fc sections
        cnn_x = x[:, :4, :, :]
        fc_x = x[:, 4:, :, :]

        # shape: (batch_size, 2, 5)
        fc_x = fc_x.diagonal(0, dim1=2, dim2=3)

        # cnn section

        for block in self.conv_blocks:
            cnn_x = block(cnn_x)

        cnn_x = cnn_x.reshape(-1, self.num_elements_output)

        cnn_x = self.cnn_linear(cnn_x)

        # fc section

        # flatten but keep batch size
        fc_x = fc_x.flatten(start_dim=1)

        for block in self.linear_blocks:
            fc_x = block(fc_x)

        if self.ablation is None:
            x = torch.cat([cnn_x, fc_x], dim=1)
        elif self.ablation == 'distance':
            x = fc_x
        elif self.ablation == 'player_count':
            x = cnn_x

        x = self.final_linear(x)

        y = self.sigmoid(x)

        return y.squeeze(1)


class NFL_NN(torch.nn.Module):

    def __init__(self,
                 input_size=(1, 12, 10),
                 activation='ReLU',
                 activation_params={},
                 hidden_sizes=[200, 100, 50],
                 output_size=1,
                 batch_norm=False,
                 dropout=False,
                 cnn_options=tuple(),
                 ):
        super().__init__()

        # all of the function params are unused
        # we're just doing the same model as:
        # https://github.com/juancamilocampos/nfl-big-data-bowl-2020/blob/master/1st_place_zoo_solution_v2.ipynb

        self.conv1 = torch.nn.Conv2d(7, 128, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv2d(128, 160, kernel_size=1, stride=1)
        self.conv3 = torch.nn.Conv2d(160, 128, kernel_size=1, stride=1)

        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=(1, 5))
        self.avgpool1 = torch.nn.AvgPool2d(kernel_size=(1, 5))

        self.norm1 = torch.nn.BatchNorm1d(num_features=128)

        self.conv4 = torch.nn.Conv1d(128, 160, kernel_size=1, stride=1)
        self.norm2 = torch.nn.BatchNorm1d(num_features=160)
        self.conv5 = torch.nn.Conv1d(160, 96, kernel_size=1, stride=1)
        self.norm3 = torch.nn.BatchNorm1d(num_features=96)
        self.conv6 = torch.nn.Conv1d(96, 96, kernel_size=1, stride=1)
        self.norm4 = torch.nn.BatchNorm1d(num_features=96)

        self.maxpool2 = torch.nn.MaxPool1d(kernel_size=11)
        self.avgpool2 = torch.nn.AvgPool1d(kernel_size=11)

        self.fc1 = torch.nn.Linear(96, 96)
        self.norm5 = torch.nn.BatchNorm1d(num_features=96)
        self.fc2 = torch.nn.Linear(96, 256)
        self.norm6 = torch.nn.LayerNorm(normalized_shape=256)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc3 = torch.nn.Linear(256, 2)

        self.relu = torch.nn.ReLU()

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        xmax = self.maxpool1(x) * 0.3
        xavg = self.avgpool1(x) * 0.7
        x = xmax + xavg

        x = x.squeeze()
        x = self.norm1(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.norm2(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.norm3(x)

        x = self.conv6(x)
        x = self.relu(x)
        x = self.norm4(x)

        xmax = self.maxpool1(x) * 0.3
        xavg = self.avgpool1(x) * 0.7
        x = xmax + xavg

        x = x.squeeze()

        x = self.fc1(x)
        x = self.relu(x)
        x = self.norm5(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.norm6(x)
        x = self.dropout(x)

        x = self.fc3(x)

        x = self.softmax(x)

        return x[:, 1]


if __name__ == '__main__':

    # dummy data for testing
    # (n_samples, 12, 10)
    t = torch.rand(size=(5, 1, 12, 10))

    print('Testing FCNN')

    mod = FCNN(dropout=True)

    res = mod(t).detach()

    print(res.shape)
    print(res)

    print('\nTesting CNN')

    mod = CNN(dropout=True)

    res = mod(t).detach()

    print(res.shape)
    print(res)

    print('\nTesting ResNet')

    mod = ResNet(dropout=True)

    res = mod(t).detach()

    print(res.shape)
    print(res)

    print('\nTesting LR-CNN')

    t = torch.rand(size=(5, 6, 5, 5))

    mod = LR_CNN()

    res = mod(t).detach()

    print(res.shape)
    print(res)

    print('\nTesting NFL-NN')

    t = torch.rand(size=(5, 7, 5, 5))

    mod = NFL_NN()

    res = mod(t).detach()

    print(res.shape)
    print(res)
