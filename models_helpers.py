import torch
import torch.nn as nn
# Helper Methods
class RDB_Conv(nn.Module):
    '''
    Residual Dense Block Convolution Layer
    To be used in RDB Layer to create a dense block
    
    Functions:
    1) Apply convolution to input tensor
    2) Apply ReLU activation function
    3) Concatenate input tensor with output tensor

    Default kernel size is 3
    '''
    def __init__(self, in_channels, out_channels, k_size=3):
        super(RDB_Conv, self).__init__()

        self.conv = nn.Sequential(*[
            nn.Conv2d(in_channels, out_channels, k_size, padding=(k_size-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    '''
    Residual Dense Block Layer
    Instantiates RDB_Conv N times to create a dense block
    Plus a concatenation of input and output tensors for residual connection
    
    Functions:
    1) For a total of N times: Apply RDB_Conv layer to input tensor
    2) Apply 1x1 convolution to output tensor to fuse all features
    3) Concatenate input tensor with output tensor
    '''

    def __init__(self, in_channels, nth_rdb_conv_channels, n_rdb_conv, k_size):
        super(RDB, self).__init__()

        N = n_rdb_conv # Number of RDB_Conv layers
        C0 = in_channels # Number of input channels
        CN = nth_rdb_conv_channels # Number of output channels

        # Create N RDB_Conv layers
        convs = []
        for i in range(N):
            convs.append(RDB_Conv(C0 + i * CN, CN, k_size))
        self.rdb_convs = nn.Sequential(*convs)

        # Fuse Local features using 1x1 Convolution
        self.lff = nn.Conv2d(C0 + N * CN, C0, 1, padding=0, stride=1)

    def forward(self, x):
        out = self.rdb_convs(x)
        return x + self.lff(out)
