import argparse, sys
sys.path.insert(0, ".")

import torch
from torchvision import utils
from model.stylegan2 import Generator
from tqdm import tqdm

def generate(args, g_ema, device):

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)
            if i % 2 == 0:
                g_ema.set_noise(g_ema.generate_noise())
            else:
                g_ema.set_noise(None)
            sample = g_ema(sample_z)
            
            utils.save_image(
                sample,
                f'results/{str(i).zfill(6)}.png',
                nrow=1,
                normalize=True,
                range=(-1, 1))

if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--pics', type=int, default=20)

    parser.add_argument('--ckpt', type=str, default="stylegan2-ffhq-config-f.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint['g_ema'])



    generate(args, g_ema, device)
