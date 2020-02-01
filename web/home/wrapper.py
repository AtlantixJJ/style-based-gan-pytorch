import torch
from home.stylegan import StyledGenerator
from home.optim import get_optim_func
from home import utils

class WrapedStyledGenerator(torch.nn.Module):
    def __init__(self, optim="", seg_cfg="", load_path="", gpu=-1):
        super(WrapedStyledGenerator, self).__init__()
        self.device = 'cuda' if gpu >= 0 else 'cpu'
        self.load_path = load_path
        self.seg_cfg = seg_cfg
        self.optim_func = get_optim_func(optim)

        print("=> Constructing network architecture")
        self.model = StyledGenerator(semantic=seg_cfg)
        print("=> Loading parameter from %s" % self.load_path)
        self.model.load_state_dict(torch.load(self.load_path, map_location='cpu'))
        try:
            self.model = self.model.to(self.device)
        except:
            print("=> Fall back to CPU")
            self.device = 'cpu'
        self.model.eval()
        print("=> Check running")
        self.n_class = self.model.n_class
        print("=> Optimization method %s" % str(self.optim_func))

        self.latent_param = torch.randn(1, 512, requires_grad=True, device=self.device)

    def generate_noise(self):
        sizes = [4 * 2 ** (i // 2) for i in range(18)]
        length = sum([size ** 2 for size in sizes])
        latent = torch.randn(1, 512, device=self.device)
        noise_vec = torch.randn((length,), device=self.device)
        return latent, noise_vec


    def generate_given_image_stroke(self, latent, noise, image_stroke, image_mask):
        utils.copy_tensor(self.latent_param, latent)
        noises = utils.parse_noise(noise)

        image, label, latent, noises, record = self.optim_func(
            model=self.model, latent=self.latent_param, noises=noises, image_stroke=image_stroke, image_mask=image_mask)
        noise = torch.cat([n.view(-1) for n in noise])

        image = utils.torch2numpy(image * 255).transpose(0, 2, 3, 1)
        label = utils.torch2numpy(label)
        latent = utils.torch2numpy(latent)
        noise = utils.torch2numpy(noise)
        return image.astype("uint8"), label, latent, noise, record


    def forward(self, latent, noise): # [0, 1] in torch
        noise = utils.parse_noise(noise)
        self.model.set_noise(noise)
        gen, seg = self.model(latent)

        gen = (1 + gen.clamp(-1, 1)) * 255 / 2
        gen = utils.torch2numpy(gen).transpose(0, 2, 3, 1)
        label = utils.torch2numpy(seg.argmax(1))
        return gen.astype("uint8"), label
