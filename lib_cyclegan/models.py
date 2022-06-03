from cv2 import sepFilter2D
import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels+2),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels+2), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class GeneratorResNet_bottleneck(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet_bottleneck, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model_encoder = [
            nn.ReflectionPad2d(channels+2),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        model_decoder = []
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model_encoder += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks//2):
            model_encoder += [ResidualBlock(out_features)]


        for _ in range(num_residual_blocks - num_residual_blocks//2):
            model_decoder += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model_decoder += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model_decoder += [nn.ReflectionPad2d(channels+2), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model_encoder = nn.Sequential(*model_encoder)
        self.model_decoder = nn.Sequential(*model_decoder)

    def forward(self, x):
        feature_bottleneck = self.model_encoder(x)
        res = self.model_decoder(feature_bottleneck)
        return res, feature_bottleneck

class GeneratorResNet_encoder(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet_encoder, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model_encoder = [
            nn.ReflectionPad2d(channels+2),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        model_decoder = []
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model_encoder += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks//2):
            model_encoder += [ResidualBlock(out_features)]


        for _ in range(num_residual_blocks - num_residual_blocks//2):
            model_decoder += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model_decoder += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model_decoder += [nn.ReflectionPad2d(channels+2), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model_encoder = nn.Sequential(*model_encoder)
        self.model_decoder = nn.Sequential(*model_decoder)

    def forward(self, x):
        feature_bottleneck = self.model_encoder(x)
        return feature_bottleneck

class GeneratorResNet_decoder(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet_decoder, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model_encoder = [
            nn.ReflectionPad2d(channels+2),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        model_decoder = []
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model_encoder += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks//2):
            model_encoder += [ResidualBlock(out_features)]


        for _ in range(num_residual_blocks - num_residual_blocks//2):
            model_decoder += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model_decoder += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model_decoder += [nn.ReflectionPad2d(channels+2), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model_encoder = nn.Sequential(*model_encoder)
        self.model_decoder = nn.Sequential(*model_decoder)

    def forward(self, x):
        res = self.model_decoder(x)
        return res

class GeneratorResNet_bottleneck_header(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet_bottleneck_header, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model_encoder = [
            nn.ReflectionPad2d(channels+2),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        model_decoder = []
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model_encoder += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks//2):
            model_encoder += [ResidualBlock(out_features)]

        model_header = nn.Sequential(
            nn.Conv2d(out_features, out_features, 1),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
        )


        for _ in range(num_residual_blocks - num_residual_blocks//2):
            model_decoder += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model_decoder += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model_decoder += [nn.ReflectionPad2d(channels+2), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model_encoder = nn.Sequential(*model_encoder)
        self.model_decoder = nn.Sequential(*model_decoder)
        self.model_header = model_header

    def forward(self, x):
        feature_bottleneck = self.model_encoder(x)
        feature_bottleneck_header = self.model_header(feature_bottleneck)
        res = self.model_decoder(feature_bottleneck)

        return res, feature_bottleneck_header

class Model_header(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(Model_header, self).__init__()

        # Initial convolution block
        in_features = input_shape[0]
        out_features = 256
        model_header = []

        for _ in range(num_residual_blocks):
            model_header += [
                nn.Conv2d(in_features, out_features, 1, stride=1, padding=0),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        self.model_header = nn.Sequential(*model_header)

    def forward(self, x):
        return self.model_header(x)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)


class Discriminator_f(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator_f, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 2, width // 2 ** 2)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 256, normalize=False),
            # *discriminator_block(64, 128),
            # *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
