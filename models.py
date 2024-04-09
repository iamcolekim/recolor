import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class GeneratorModel(nn.Module):
    """
    Description here
    """

    def __init__(self):
        super(GeneratorModel, self).__init__()
        n_resnet_features = 128

        # Load the pretrained ResNet-101
        resnet = models.resnet18(num_classes=365)
        # Change first conv layer to accept single-channel input
        resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1))
        # Extract midlevel features from ResNet-gray
        self.resnet_encoder = nn.Sequential(*list(resnet.children())[0:6])

        self.resize = nn.AvgPool2d(8, 8)
        self.conv1 = nn.Conv2d(512, 32, 1)
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(129, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 2, 1)
        )

    def forward(self, x):
        # x -> [1 x 224 x 224] L channel

        # [128, 28, 28]
        resnet_input = self.resnet_encoder(x)

        # [1, 28, 28]
        image_input = self.resize(x)

        # [129, 28, 28]
        comb_input = torch.cat([resnet_input, image_input], axis=1)

        # [16, 224, 224]
        out1 = self.conv2_1(comb_input)

        # [2, 224, 224]
        x = F.sigmoid(self.conv3(out1))
        # x = self.conv3(out1)

        return x

from models_helpers import RDB, RDB_Conv
class baseline_refine(nn.Module):
    """
    Description here
    """

    def __init__(self):
        super(baseline_refine, self).__init__()

        self.upsample = nn.Upsample(scale_factor=4, mode="bilinear")

    def forward(self, x):
        # [3 x 56 x 56]
        return self.upsample(x)

class RefinementModel(nn.Module):
    """
    # Inspired by:
    # Residual Dense Network for Image Super-Resolution
    # https://arxiv.org/abs/1802.08797

    This refinement model is a deep residual network
    that uses residual dense blocks (RDBs) to extract features.
    
    Both global (from first conv layers) and local features from (RDBs) 
    are fused to generate the final output.

    Up-sampling is done using PixelShuffle. For more information, see:
    https://arxiv.org/abs/1609.05158
    """

    def __init__(self, rdb_channel_in, rdb_channel_out, in_channels, k_size, scale_factor, rdb_depth = [1,5]):
        super(RefinementModel, self).__init__()
        C0 = rdb_channel_in
        CN = rdb_channel_out
        self.D = rdb_depth[0]
        N = rdb_depth[1]

        # Shallow Feature Extraction
        self.sfe1 = nn.Conv2d(in_channels, C0, k_size, padding=(k_size-1)//2, stride=1)
        self.sfe2 = nn.Conv2d(C0, C0, k_size, padding=(k_size-1)//2, stride=1)

        # Residual Dense Blocks
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(RDB(C0, CN, N, k_size))

        # Global Feature Fusion using 1x1 Convolution
        self.gff = nn.Sequential(*[
            nn.Conv2d(self.D * C0, C0, 1, padding=0, stride=1),
            nn.Conv2d(C0, C0, k_size, padding=(k_size-1)//2, stride=1)
        ])

        # Up-sampling net
            # This is a simple up-sampling net that uses PixelShuffle
            # A common technique for up-sampling in super-resolution tasks
        

        if scale_factor == 2 or scale_factor == 3:
            r = scale_factor
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(C0, CN * r * r, k_size, padding=(k_size-1)//2, stride=1),
                nn.PixelShuffle(r),
                nn.Conv2d(CN, in_channels, k_size, padding=(k_size-1)//2, stride=1)
            ])
        elif scale_factor == 4:
            r = scale_factor
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(C0, CN * 4, k_size, padding=(k_size-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(CN, CN * 4, k_size, padding=(k_size-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(CN, in_channels, k_size, padding=(k_size-1)//2, stride=1)
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")
    def forward(self, x):
        f__1 = self.sfe1(x)
        x = self.sfe2(f__1)

        # Residual Dense Blocks
        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        # Global Feature Fusion
        x = self.gff(torch.cat(RDBs_out, 1)) + f__1

        # Up-sampling net
        x = self.UPNet(x)

        return x


class RecolorModel(nn.Module):
    """
    Description here
    """

    def __init__(self, generator, refinement):
        super(RecolorModel, self).__init__()

        self.generator = generator
        self.refinement = refinement

    def forward(self, x):
        return self.refinement(self.generator(x))
