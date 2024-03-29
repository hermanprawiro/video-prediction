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
from models import networks
from utils.misc import to_one_hot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=32)
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
parser.add_argument("--frame_length", type=int, default=4)
parser.add_argument("--checkpoint_path", type=str, default="checkpoints")
parser.add_argument("--result_path", type=str, default="results")
parser.add_argument("--save_name", type=str, default="simple")

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

    dataset = BP4D_Micro(train=True, grayscale=args.image_gray, transform=tfs, frame_length=args.frame_length, frame_dilation=4)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True, pin_memory=True)

    netE = networks.ImageEncoder(img_dim=args.image_ch, latent_dim=args.latent_dim, base_dim=args.ndf).to(device)
    netT = networks.LatentTransform(latent_dim=args.latent_dim, output_frame=(args.frame_length - 1)).to(device)
    netG = networks.Decoder(img_dim=args.image_ch, latent_dim=args.latent_dim, base_dim=args.ngf).to(device)

    optG = torch.optim.Adam(list(netT.parameters()) + list(netE.parameters()) + list(netG.parameters()), lr=2e-4, betas=(args.beta1, args.beta2))
    # optD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(args.beta1, args.beta2))

    netG.train()
    netT.train()
    netE.train()
    for epoch in range(args.n_epochs):
        for i, (x_real, task_ids, _, _) in enumerate(dataloader):
            x_real = x_real.to(device)
            n_batch = x_real.shape[0]

            optG.zero_grad()
            x_first = x_real[:, :, 0]
            z_first = netE(x_first)
            
            t = netT(z_first)
            z_rest = z_first.unsqueeze(1) + t
            z_rest = z_rest.view(n_batch * netT.output_frame, -1)

            delta_rest = netG(z_rest).view((n_batch, netT.output_frame) + x_first.shape[1:]).permute(0, 2, 1, 3, 4)
            # recon_rest = x_first.unsqueeze(2) + delta_rest
            recon_rest = delta_rest
            recon_first = netG(z_first)

            loss_first = F.binary_cross_entropy(recon_first, x_first)
            loss_rest = F.binary_cross_entropy(recon_rest, x_real[:, :, 1:])
            loss = loss_first + loss_rest * 10
            loss.backward()
            optG.step()

            # Longer frame length, 8 maybe
            # Classify task from vector t
            # Input 2 first frames, predict the rest
            # t only modify half of z

            if i % 50 == 0:
                print('[%d/%d][%d/%d] Loss First/Rest: %.4f/%.4f | NormT: %.4f'
                    % (epoch + 1, args.n_epochs, i, len(dataloader), loss_first.item(), loss_rest.item(), t.norm().item()))

                # Reconstruction from real image
                x_real = x_real[:8].detach()
                recon_first = recon_first[:8].detach().unsqueeze(2)
                recon_rest = recon_rest[:8].detach()
                recon = torch.cat([recon_first, recon_rest], dim=2)
                errors = (x_real - recon).abs()
                errors = (errors - errors.min()) / (errors.max() - errors.min())
                delta_rest = (recon_first - recon_rest).abs()
                delta_rest = (delta_rest - delta_rest.min()) / (delta_rest.max() - delta_rest.min())
                vis = torch.cat([x_real, recon, errors, delta_rest], dim=2).permute(0, 2, 1, 3, 4)
                vis = torch.reshape(vis, (vis.shape[0] * vis.shape[1],) + vis.shape[2:])
                save_image(vis, '%s/epoch%03d_%04d.jpg' % (args.result_path, epoch + 1, i + 1), normalize=False, nrow=15)

        save_model((netG, netT, netE), (optG), epoch, args.checkpoint_path)


def save_model(models, optimizers, epoch, checkpoint_path):
    netG, netT, netE = models
    # optG, optD = optimizers

    checkpoint = {
        'state_dict': {
            'generator': netG.state_dict(),
            'transform': netT.state_dict(),
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
