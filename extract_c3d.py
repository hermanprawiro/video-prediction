import argparse
import os

import numpy as np
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
parser.add_argument("--n_epochs", type=int, default=10)
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
parser.add_argument("--result_path", type=str, default="latents")
parser.add_argument("--save_name", type=str, default="c3d")

def main(args):
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

    checkpoints = torch.load(os.path.join(args.checkpoint_path, 'checkpoint_%03d.pth' % (10)), map_location=torch.device('cpu'))
    netE.load_state_dict(checkpoints['state_dict']['encoder'])

    netE.eval()
    with torch.no_grad():
        for subject in range(139):
            for task in range(10):
                dataset = BP4D_Single(subject=subject, task=task, grayscale=True, transform=tfs, frame_length=args.frame_length, frame_dilation=0)
                if len(dataset) == 0:
                    continue
                dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=True)

                subject_name = dataset.subject
                task_name = dataset.task
                print("Processing %s %s" % (subject_name, task_name))

                latents = []
                for i, (x_real, _, _, _) in enumerate(dataloader):
                    x_real = x_real.to(device)
                    n_batch = x_real.shape[0]

                    z = netE(x_real)
                    latents.append(z.cpu().numpy())

                latents = np.concatenate(latents, axis=0)
                print(latents.shape)
                np.save(os.path.join(args.result_path, f'{subject_name}_{task_name}.npy'), latents)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
