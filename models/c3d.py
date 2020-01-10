import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, img_dim=3, latent_dim=128, base_dim=64):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(img_dim, base_dim, 3, stride=(1, 2, 2), padding=1), # 16, 64
            nn.ReLU(True),
            nn.Conv3d(base_dim, base_dim * 2, 3, stride=2, padding=1), # 8, 32
            nn.ReLU(True),

            nn.Conv3d(base_dim * 2, base_dim * 4, 3, stride=1, padding=1), # 8, 32
            nn.ReLU(True),
            nn.Conv3d(base_dim * 4, base_dim * 4, 3, stride=2, padding=1), # 4, 16
            nn.ReLU(True),
            
            nn.Conv3d(base_dim * 4, base_dim * 8, 3, stride=1, padding=1), # 4, 16
            nn.ReLU(True),
            nn.Conv3d(base_dim * 8, base_dim * 8, 3, stride=2, padding=1), # 2, 8
            nn.ReLU(True),

            nn.Conv3d(base_dim * 8, base_dim * 8, 3, stride=1, padding=1), # 2, 8
            nn.ReLU(True),
            nn.Conv3d(base_dim * 8, base_dim * 8, 3, stride=2, padding=1), # 1, 4
            nn.ReLU(True),
        )
        self.linear = nn.Sequential(
            nn.Linear(base_dim * 8 * 4 * 4, 4096),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(4096, latent_dim)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.flatten(1)
        out = self.linear(out)

        return out

class Decoder(nn.Module):
    def __init__(self, img_dim=3, latent_dim=128, base_dim=32, bottom_width=4):
        super().__init__()

        self.base_dim = base_dim
        self.bottom_width = bottom_width

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, base_dim * 8 * bottom_width * bottom_width), # 1, 4
            nn.ReLU(True),
        )
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2), # 2, 8
            nn.Conv3d(base_dim * 8, base_dim * 8, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv3d(base_dim * 8, base_dim * 8, 3, stride=1, padding=1),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2), # 4, 16
            nn.Conv3d(base_dim * 8, base_dim * 8, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv3d(base_dim * 8, base_dim * 4, 3, stride=1, padding=1),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2), # 8, 32
            nn.Conv3d(base_dim * 4, base_dim * 4, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv3d(base_dim * 4, base_dim * 2, 3, stride=1, padding=1),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2), # 16, 64
            nn.Conv3d(base_dim * 2, base_dim, 3, stride=1, padding=1),
            nn.ReLU(True),

            nn.Upsample(scale_factor=(1, 2, 2)), # 16, 128
            nn.Conv3d(base_dim, base_dim, 3, stride=1, padding=1),
            nn.ReLU(True),

            nn.Conv3d(base_dim, img_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        out = self.linear(z)
        out = out.view(-1, self.base_dim * 8, 1, self.bottom_width, self.bottom_width)
        out = self.conv(out)

        return out

if __name__ == "__main__":
    x = torch.randn(2, 3, 16, 128, 128)
    netE = Encoder()
    netD = Decoder()

    z = netE(x)
    x_recon = netD(z)
    print(z.shape, x_recon.shape)