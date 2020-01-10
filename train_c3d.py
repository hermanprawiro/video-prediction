import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataloaders.bp4d import BP4D_Micro, BP4D_Single
from models import c3d
from utils.misc import to_one_hot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=25)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=2e-4)
parser.add_argument("--beta1", type=float, default=0.5)
parser.add_argument("--beta2", type=float, default=0.999)
parser.add_argument("--ndf", type=int, default=32, help="Base features multiplier for discriminator")
parser.add_argument("--ngf", type=int, default=32, help="Base features multiplier for generator")
parser.add_argument("--n_disc_update", type=int, default=1)
parser.add_argument("--n_workers", type=int, default=4)
parser.add_argument("--n_class", type=int, default=10)
parser.add_argument("--latent_dim", type=int, default=128)
parser.add_argument("--image_ch", type=int, default=3)
parser.add_argument("--image_res", type=int, default=128)
parser.add_argument("--image_gray", action="store_true")
parser.add_argument("--frame_length", type=int, default=16)
parser.add_argument("--checkpoint_path", type=str, default="checkpoints")
parser.add_argument("--result_path", type=str, default="results")
parser.add_argument("--save_name", type=str, default="c3d")

def main(args):
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    if args.image_gray:
        args.save_name += "_gray"
        args.image_ch = 1

    args.checkpoint_path = os.path.join(args.checkpoint_path, args.save_name)
    args.result_path = os.path.join(args.result_path, args.save_name)
    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.result_path, exist_ok=True)

    tfs = transforms.Compose([
        transforms.Resize(args.image_res),
        transforms.ToTensor(),
        # transforms.Normalize([0.5]*args.image_ch, [0.5]*args.image_ch)
    ])

    dataset = BP4D_Micro(train=True, grayscale=args.image_gray, transform=tfs, frame_length=args.frame_length, frame_dilation=0)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True, pin_memory=True)

    netE = c3d.Encoder(img_dim=args.image_ch, latent_dim=args.latent_dim, base_dim=args.ndf).to(device)
    netG = c3d.Decoder(img_dim=args.image_ch, latent_dim=args.latent_dim, base_dim=args.ngf).to(device)

    optG = torch.optim.Adam(list(netE.parameters()) + list(netG.parameters()), lr=2e-4, betas=(args.beta1, args.beta2))
    # optD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(args.beta1, args.beta2))

    netG.train()
    netE.train()
    for epoch in range(args.n_epochs):
        for i, (x_real, task_ids, _, _) in enumerate(dataloader):
            x_real = x_real.to(device)
            n_batch = x_real.shape[0]

            optG.zero_grad()
            z = netE(x_real)
            x_recon = netG(z)
            loss = F.binary_cross_entropy(x_recon, x_real)
            
            loss.backward()
            optG.step()

            if i % 50 == 0:
                print('[%d/%d][%d/%d] Loss: %.4f'
                    % (epoch + 1, args.n_epochs, i, len(dataloader), loss.item()))

                # Reconstruction from real image
                x_real = x_real[:8].detach()
                x_recon = x_recon[:8].detach()
                vis = torch.cat([x_real, x_recon], dim=2).permute(0, 2, 1, 3, 4)
                vis = torch.reshape(vis, (vis.shape[0] * vis.shape[1],) + vis.shape[2:])
                save_image(vis, '%s/epoch%03d_%04d.jpg' % (args.result_path, epoch + 1, i + 1), normalize=False, nrow=16)

        save_model((netG, netE), (optG), epoch, args.checkpoint_path)


def save_model(models, optimizers, epoch, checkpoint_path):
    netG, netE = models
    # optG, optD = optimizers

    checkpoint = {
        'state_dict': {
            'generator': netG.state_dict(),
            # 'transform': netT.state_dict(),
            'encoder': netE.state_dict(),
        },
        # 'optimizer': {
        #     'generator': optG.state_dict(),
        #     'discriminator': optD.state_dict(),
        # },
        'optimizer': optimizers.state_dict(),
        'epoch': epoch,
    }

    torch.save(checkpoint, '%s/checkpoint_%03d.pth' % (checkpoint_path, epoch + 1))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
