import torch.nn as nn


ARCHITECTURE_NAME = "conv_autoencoder_v1"
BOTTLENECK_ARCHITECTURE_NAME = "conv_autoencoder_bottleneck_v1"


class ConvAutoencoder(nn.Module):

    def __init__(self, input_channels=3, base_channels=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                base_channels * 2,
                base_channels * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                base_channels * 4,
                base_channels * 8,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 8,
                base_channels * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                base_channels * 4,
                base_channels * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                base_channels * 2,
                base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                base_channels,
                input_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def encode(self, images):
        return self.encoder(images)

    def forward(self, images):
        encoded = self.encode(images)
        return self.decoder(encoded)


class ConvAutoencoderBottleneck(nn.Module):

    def __init__(
        self,
        input_channels=3,
        base_channels=32,
        image_size=224,
        latent_dim=256,
    ):
        super().__init__()
        if image_size % 16 != 0:
            raise ValueError("image_size must be divisible by 16.")

        self.input_channels = input_channels
        self.base_channels = base_channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.spatial_size = image_size // 16
        encoded_channels = base_channels * 8
        encoded_features = encoded_channels * self.spatial_size * self.spatial_size

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                base_channels * 2,
                base_channels * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                base_channels * 4,
                encoded_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(encoded_channels),
            nn.ReLU(inplace=True),
        )
        self.to_latent = nn.Linear(encoded_features, latent_dim)
        self.from_latent = nn.Linear(latent_dim, encoded_features)
        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (encoded_channels, self.spatial_size, self.spatial_size)),
            nn.ConvTranspose2d(
                encoded_channels,
                base_channels * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                base_channels * 4,
                base_channels * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                base_channels * 2,
                base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                base_channels,
                input_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def encode(self, images):
        encoded = self.encoder_conv(images)
        return self.to_latent(encoded.flatten(start_dim=1))

    def forward(self, images):
        latent = self.encode(images)
        decoded_features = self.from_latent(latent)
        return self.decoder_conv(decoded_features)
