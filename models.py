import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class GeneratorModel(nn.Module):
    """
    Generator model for image colorization, which predicts a low-resolution colorization.

    Inputs: [B, 1, 224, 224] L channel
    Outputs: [B, 2, 56, 56] ab channels
    """

    def __init__(self):
        super(GeneratorModel, self).__init__()

        # Load the pretrained ResNet
        resnet = models.resnet18(num_classes=365)
        resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1))
        self.resnet_encoder = nn.Sequential(*list(resnet.children())[0:6])

        # Downscaling for L channel
        self.resize = nn.AvgPool2d(8, 8)

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
        # Pass through ResNet encoder
        resnet_input = self.resnet_encoder(x)

        # Downscale L input and add to ResNet features
        image_input = self.resize(x)
        comb_input = torch.cat([resnet_input, image_input], axis=1)

        # Process encodings
        out1 = self.conv2_1(comb_input)
        x = F.sigmoid(self.conv3(out1))

        return x


class RecolorModel(nn.Module):
    """
    Combined generator-refinement model capable of performing image colorization on 224x224 images.

    Inputs: [B, 1, 224, 224] L channel
    Outputs: [B, 2, 224, 224] ab channels
    """

    def __init__(self, generator, refinement):
        super(RecolorModel, self).__init__()

        self.generator = generator
        self.refinement = refinement

    def forward(self, L):
        out = self.generator(L)
        return self.refinement([L, out])


class ResBlock(nn.Module):
    """
    Implementation of a residual block, as implemented in the EDSR model.
    """

    def __init__(self, n_feats, scale=1):
        super(ResBlock, self).__init__()

        # Body: conv -> ReLU -> conv
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, 3, padding=1)
        )

        self.res_scale = scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x  # Residual connection
        return res


class RefinementModel(nn.Module):
    """
    Refinement model which takes a low resolution colorized image and the ground truth lightness image and predicts
    a high-resolution colorized image.

    Inputs: [B, 1, 224, 224] L channel, [B, 1, 56, 56] ab channels
    Outputs: [B, 2, 224, 224] ab channels

    This model was inspired by the code provided by https://medium.com/@paren8esis/an-introduction-to-super-resolution-with-deep-learning-pt-3-ed85ec949ba8

    The info in the link is itself based on the EDSR model, which is a super-resolution model. The model is adapted
    for colorization by changing the number of input and output channels.

    Reference for EDSR:
    Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee, "Enhanced Deep Residual Networks for Single Image Super-Resolution," 2nd NTIRE: New Trends in Image Restoration and Enhancement workshop and challenge on image super-resolution in conjunction with CVPR 2017.
    """

    def __init__(self):
        super(RefinementModel, self).__init__()

        # Load the pretrained ResNet
        resnet = models.resnet18(num_classes=365)
        resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1))
        self.resnet_encoder = nn.Sequential(*list(resnet.children())[0:5])

        n_resblocks = 8
        n_features = 128

        # Head for processing of ab channels
        self.head = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # ResBlock for processing of ab channels
        self.res1 = ResBlock(64)

        # Body for processing of combined L and ab encodings
        body_layers = [ResBlock(n_features) for _ in range(n_resblocks)]
        body_layers.append(nn.Conv2d(n_features, n_features, 3, padding=1))
        self.body = nn.Sequential(*body_layers)

        # Tail for upsampling
        self.tail = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bicubic"),  # Attempt resize-convolution
            nn.Conv2d(n_features, n_features, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bicubic"),  # Attempt resize-convolution
            nn.Conv2d(n_features, n_features, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_features, 2, 3, padding=1)
        )

    def forward(self, x):
        L_in, ab_in = x

        # Pass L input through ResNet to obtain encodings
        x = self.resnet_encoder(L_in)

        # Pass ab input through head and ResBlock to obtain encodings
        y = self.head(ab_in)
        y = self.res1(y)

        # Combine L and ab encodings
        skip = torch.cat([x, y], 1)

        # Pass encodings into ResBlocks and apply additional skip connection at end
        res = self.body(skip)
        res += skip

        # Apply 4x upscaling using Upsample-Conv
        res = self.tail(res)

        return F.sigmoid(res)
