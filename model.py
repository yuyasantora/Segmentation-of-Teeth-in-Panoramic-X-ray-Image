"""
PyTorch implementation of U-Net for teeth segmentation
Converted from TensorFlow/Keras implementation (backup: model_tensorflow.py)
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv2D -> Dropout -> Conv2D -> BatchNorm) block"""

    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with MaxPool then DoubleConv"""

    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_rate)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1)
        self.conv = DoubleConv(in_channels, out_channels, dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNET(nn.Module):
    """
    U-Net architecture for semantic segmentation
    Compatible with original TensorFlow implementation

    Args:
        input_shape: Tuple of (height, width, channels) - kept for compatibility
        last_activation: 'sigmoid' or 'softmax' (default: 'sigmoid')
    """

    def __init__(self, input_shape=(512, 512, 1), last_activation='sigmoid'):
        super(UNET, self).__init__()

        # Convert TensorFlow format (H, W, C) to PyTorch format (C, H, W)
        if len(input_shape) == 3:
            in_channels = input_shape[2]
        else:
            in_channels = 1

        features = 32

        # Encoder (downsampling path)
        self.encoder1 = DoubleConv(in_channels, features, dropout_rate=0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = DoubleConv(features, features * 2, dropout_rate=0.2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = DoubleConv(features * 2, features * 4, dropout_rate=0.3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = DoubleConv(features * 4, features * 8, dropout_rate=0.4)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(features * 8, features * 16, dropout_rate=0.5)

        # Decoder (upsampling path)
        self.upconv4 = Up(features * 16, features * 8, dropout_rate=0.4)
        self.upconv3 = Up(features * 8, features * 4, dropout_rate=0.3)
        self.upconv2 = Up(features * 4, features * 2, dropout_rate=0.2)
        self.upconv1 = Up(features * 2, features, dropout_rate=0.1)

        # Final output layer
        self.output = nn.Conv2d(features, 1, kernel_size=1)

        # Activation
        if last_activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif last_activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        else:
            self.activation = nn.Identity()

        # Initialize weights using He initialization (similar to 'he_normal' in Keras)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck, enc4)
        dec3 = self.upconv3(dec4, enc3)
        dec2 = self.upconv2(dec3, enc2)
        dec1 = self.upconv1(dec2, enc1)

        # Output
        out = self.output(dec1)
        return self.activation(out)


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = UNET(input_shape=(512, 512, 1), last_activation='sigmoid')
    model = model.to(device)

    # Test forward pass
    x = torch.randn(1, 1, 512, 512).to(device)
    with torch.no_grad():
        output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
