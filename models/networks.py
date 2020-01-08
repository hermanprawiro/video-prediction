import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    def __init__(self, img_dim=3, latent_dim=128, base_dim=32):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(img_dim, base_dim * 2, 7, stride=2, padding=3), # 64
            nn.BatchNorm2d(base_dim * 2),
            nn.ReLU(True),

            nn.Conv2d(base_dim * 2, base_dim * 4, 3, stride=2, padding=1), # 32
            nn.BatchNorm2d(base_dim * 4),
            nn.ReLU(True),
            nn.Conv2d(base_dim * 4, base_dim * 8, 3, stride=2, padding=1), # 16
            nn.BatchNorm2d(base_dim * 8),
            nn.ReLU(True),
            nn.Conv2d(base_dim * 8, base_dim * 16, 3, stride=2, padding=1), # 8
            nn.BatchNorm2d(base_dim * 16),
            nn.ReLU(True),
            nn.Conv2d(base_dim * 16, base_dim * 16, 3, stride=2, padding=1), # 4
            nn.BatchNorm2d(base_dim * 16),
            nn.ReLU(True),
        )
        self.linear = nn.Sequential(
            nn.Linear(base_dim * 16 * 4 * 4, latent_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.flatten(1)
        out = self.linear(out)

        return out

class LatentTransform(nn.Module):
    def __init__(self, latent_dim=128, output_frame=3):
        super().__init__()

        self.latent_dim = latent_dim
        self.output_frame = output_frame

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            # nn.Dropout(p=0.2),
            nn.ReLU(True),
            nn.Linear(latent_dim * 2, latent_dim * output_frame),
            # nn.Dropout(p=0.2),
            nn.ReLU(True),
        )

    def forward(self, x):
        out = self.linear(x)
        out = out.view(-1, self.output_frame, self.latent_dim)
        return out

class Decoder(nn.Module):
    def __init__(self, img_dim=3, latent_dim=128, base_dim=32, bottom_width=4):
        super().__init__()

        self.base_dim = base_dim
        self.bottom_width = bottom_width

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, base_dim * 16 * bottom_width * bottom_width),
            nn.ReLU(True),
        )
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2), # 8
            nn.Conv2d(base_dim * 16, base_dim * 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(base_dim * 16),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2), # 16
            nn.Conv2d(base_dim * 16, base_dim * 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(base_dim * 8),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2), # 32
            nn.Conv2d(base_dim * 8, base_dim * 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(base_dim * 4),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2), # 64
            nn.Conv2d(base_dim * 4, base_dim * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(base_dim * 2),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2), # 128
            nn.Conv2d(base_dim * 2, base_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(True),
            nn.Conv2d(base_dim, img_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        out = self.linear(z)
        out = out.view(-1, self.base_dim * 16, self.bottom_width, self.bottom_width)
        out = self.conv(out)

        return out

if __name__ == "__main__":
    x = torch.randn(2, 3, 128, 128)
    netE = ImageEncoder()
    netT = LatentTransform()
    netD = Decoder()

    z = netE(x)
    w = netT(z)
    x_recon = netD(z)
    print(z.shape, w.shape, x_recon.shape)