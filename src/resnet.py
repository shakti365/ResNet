
import torch.nn as nn
import numpy as np
import torch


def conv2d(in_channel, out_channel,kernel_size, stride, padding):
    """
    returns the Conv2d layer using received inputs
    :param in_channel:
    :param out_channel:
    :param kernel_size:
    :param stride:
    :param padding:
    :return: Conv2d layer
    """
    conv = nn.Conv2d(in_channel,out_channel,kernel_size=kernel_size, stride=stride,padding=padding)
    return conv


def batch_norm(in_channels):
    """
    returns the Batch_Norm layer using received inputs
    :param in_channels:
    :return: Batch_norm layer
    """
    return nn.BatchNorm2d(in_channels)


def activation_relu():
    """
    returns the relu layer
    :return: Relu layer
    """
    return nn.ReLU(inplace=True)


def pooling(kernel_size):
    """
    returns the average pooling layer using received kernel size
    :param kernel_size:
    :return: Avg_pool layer
    """
    avg_pool = nn.AvgPool2d(kernel_size)
    return avg_pool

def fully_connected(in_features, out_features):
    """
    returns the fully connected layer using receive in and out features
    :param in_features:
    :param out_features:
    :return: FC layer
    """
    fc = nn.Linear(in_features, out_features, bias=True)
    return fc


class ResidualBlock(nn.Module):
    """
    Inherits the Module class.
    Creates a residual block using the received inputs.
    """
    def __init__(self,in_channels, out_channels,stride,padding=1):
        """
        Creates a conv layer(1*1 and stride 2) and relu layer for shortcut connection
        :param in_channels:
        :param out_channels:
        :param stride:
        :param padding:
        """
        super(ResidualBlock, self).__init__()

        self.stride = stride
        self.conv1 = conv2d(in_channels, out_channels,kernel_size=1,stride=stride, padding=padding)
        self.activation = activation_relu()

    def forward(self,x1,x2):
        """
        Forward prop of shortcut connection using  conv(1 * 1) and stride 2 if dim of x1 and x2 are not similar.
        If x1 and x2 are of same dim, they are added and then activation is applied.

        :param x1:
        :param x2:
        :return: f(x1+x2)
        """

        if self.stride!=1:
            x1 = self.conv1(x1)

        output = x1 + x2
        output = self.activation(output)

        return output


class ResnetBlock(nn.Module):

    def __init__(self, in_channels, out_channels,kernel_size=1,stride=1, padding=1,activation = True):
        """

        Inherits the Module class.
        Creates a conv block of (conv+bn+Relu) layer using the received inputs.

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param activation:
        """

        super(ResnetBlock,self).__init__()
        self.conv = conv2d(in_channels, out_channels,kernel_size=kernel_size,stride=stride, padding=padding)
        self.bn = batch_norm(out_channels)
        self.activation = activation

    def forward(self,x):
        """
        Forward prop of x using conv, bn and relu layer. If activation is false, relu is not applied. This is for those
        outputs which need to be added in residual  block.
        :param x:
        :return:
        """
        x = self.conv(x)
        x = self.bn(x)

        if self.activation:
            x = activation_relu()(x)

        return x

class Net(nn.Module):
    """
    Creates the architecture of resnet 18 model with respect to CIFAR-10 dataset taking input dim as 3,32,32
    and return the output dim of (1,10)
    """
    def __init__(self):

        super(Net, self).__init__()

        self.layer1 = ResnetBlock(3, 16,kernel_size=3,stride=1, padding=1)

        # for resnet block 1 to 3
        self.resnet_1_1 = ResnetBlock(16, 16, kernel_size=3, stride=1, padding=1)
        self.resnet_1_2 = ResnetBlock(16, 16, kernel_size=3, stride=1, padding=1, activation=False)

        self.residual_1 = ResidualBlock(16,16,1)

        # for resnet block 4 to 6
        self.resnet_2_1 = ResnetBlock(16, 32, kernel_size=3, stride=2, padding=1)
        self.resnet_2_2 = ResnetBlock(32, 32, kernel_size=3, stride=1, padding=1, activation=False)
        self.resnet_2_3 = ResnetBlock(32, 32, kernel_size=3, stride=1, padding=1)

        self.residual_2_1 = ResidualBlock(16, 32, 2, padding=0)
        self.residual_2_2 = ResidualBlock(32, 32, 1)

        # for resnet block 7 to 9
        self.resnet_3_1 = ResnetBlock(32, 64, kernel_size=3, stride=2, padding=1)
        self.resnet_3_2 = ResnetBlock(64, 64, kernel_size=3, stride=1, padding=1, activation=False)
        self.resnet_3_3 = ResnetBlock(64, 64, kernel_size=3, stride=1, padding=1)
        #
        self.residual_3_1 = ResidualBlock(32, 64, 2, padding=0)
        self.residual_3_2 = ResidualBlock(64, 64, 1)

        # Avg pooling and Fully conn layer
        self.pooling = pooling(3)
        self.fc = fully_connected(256, 10)


    def forward(self, x):
        """
        forward prop of x of dim(3,32,32) and returns the final output of dim 1,10
        :param x: image
        :return:
        """

        """.....................BLOCK 1........................."""

        # layer1
        x = self.layer1(x)
        x1 = torch.clone(x)

        """...................RESNET BLOCKS......................"""

        # first res block
        x = self.resnet_1_1(x)
        x = self.resnet_1_2(x)
        x = self.residual_1(x1,x)

        # second res block
        x1 = torch.clone(x)
        x = self.resnet_1_1(x)
        x = self.resnet_1_2(x)
        x = self.residual_1(x1, x)

        # third res block
        x1 = torch.clone(x)
        x = self.resnet_1_1(x)
        x = self.resnet_1_2(x)
        x = self.residual_1(x1, x)

        # fourth res block
        x1 = torch.clone(x)
        x = self.resnet_2_1(x)
        x = self.resnet_2_2(x)
        x = self.residual_2_1(x1, x)
        #
        # fifth res block
        x1 = torch.clone(x)
        x = self.resnet_2_3(x)
        x = self.resnet_2_2(x)
        x = self.residual_2_2(x1, x)
        # #
        # # sixth res block
        x1 = torch.clone(x)
        x = self.resnet_2_3(x)
        x = self.resnet_2_2(x)
        x = self.residual_2_2(x1, x)
        #
        # seventh res block
        x1 = torch.clone(x)
        x = self.resnet_3_1(x)
        x = self.resnet_3_2(x)
        x = self.residual_3_1(x1, x)

        # eight res block
        x1 = torch.clone(x)
        x = self.resnet_3_3(x)
        x = self.resnet_3_2(x)
        x = self.residual_3_2(x1, x)

        # ninth res block
        x1 = torch.clone(x)
        x = self.resnet_3_3(x)
        x = self.resnet_3_2(x)
        x = self.residual_3_2(x1, x)

        """................POOLING AND FULLY CONNCETED LAYER"""

        # pool
        x = self.pooling(x)
        x = x.view(-1, 256)
        
        x = self.fc(x)

        return x


if __name__=="__main__":

    # Test with random input
    x = np.random.rand(32,32,3)
    x = x.transpose((2,0,1))
    x = x.reshape((1,x.shape[0],x.shape[1],x.shape[2]))

    # create model
    model = Net()

    # test the architeture of model
    output = model(torch.FloatTensor(x))
    assert output.shape==(1,10)
    print(output.shape)
