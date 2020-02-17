import torch
from home.stylegan import StyledGenerator
from home.optim import edit_image_stroke, edit_label_stroke
from home import utils


class WrapedStyledGenerator(torch.nn.Module):
    def __init__(self, resolution=1024, method="", seg_cfg="", load_path="", gpu=-1):
        super(WrapedStyledGenerator, self).__init__()
        self.device = 'cuda' if gpu >= 0 else 'cpu'
        self.load_path = load_path
        self.method = method
        self.seg_cfg = seg_cfg
        self.external_model = None

        print("=> Constructing network architecture")
        self.model = StyledGenerator(resolution=resolution, semantic=seg_cfg)
        print("=> Loading parameter from %s" % self.load_path)
        state_dict = torch.load(self.load_path, map_location='cpu')
        missed = self.model.load_state_dict(state_dict, strict=False)
        print(missed)
        try:
            self.model = self.model.to(self.device)
        except:
            print("=> Fall back to CPU")
            self.device = 'cpu'
        self.model.eval()
        self.mapping_network = self.model.g_mapping.simple_forward
        print("=> Check running")
        self.n_class = self.model.n_class
        self.noise_length = self.model.set_noise(None)
        print("=> Optimization method %s" % str(self.method))

        self.latent_param = torch.randn(1, 512, requires_grad=True, device=self.device)
        self.mix_latent_param = self.latent_param.expand(self.noise_length, -1).unsqueeze(0).detach()

    def generate_noise(self):
        print(self.noise_length)
        sizes = [4 * 2 ** (i // 2) for i in range(self.noise_length)]
        length = sum([size ** 2 for size in sizes])
        latent = torch.randn(1, 512, device=self.device)
        noise_vec = torch.randn((length,), device=self.device)
        return latent, noise_vec

    def generate_given_image_stroke(self, latent, noise, image_stroke, image_mask):
        utils.copy_tensor(self.latent_param, latent)
        noises = self.model.parse_noise(noise)

        image, label, latent, noises, record = edit_image_stroke(
            model=self.model, latent=self.latent_param, noises=noises, 
            image_stroke=image_stroke, image_mask=image_mask,
            method=self.method,
            external_model=self.external_model, mapping_network=self.mapping_network)

        # Currently no modification to noise
        # noise = torch.cat([n.view(-1) for n in noise])

        image = utils.torch2numpy(image * 255).transpose(0, 2, 3, 1)
        label = utils.torch2numpy(label)
        latent = utils.torch2numpy(latent)
        noise = utils.torch2numpy(noise)

        return image.astype("uint8"), label, latent, noise, record


    def generate_given_label_stroke(self, latent, noise, label_stroke, label_mask):
        utils.copy_tensor(self.latent_param, latent)
        noises = self.model.parse_noise(noise)

        image, label, latent, noises, record = edit_label_stroke(
            model=self.model, latent=self.latent_param, noises=noises, label_stroke=label_stroke, label_mask=label_mask,
            method=self.method.replace("image", "label"),
            external_model=self.external_model, mapping_network=self.mapping_network)
        
        # Currently no modification to noise
        # noise = torch.cat([n.view(-1) for n in noise])

        image = utils.torch2numpy(image * 255).transpose(0, 2, 3, 1)
        label = utils.torch2numpy(label)
        latent = utils.torch2numpy(latent)
        noise = utils.torch2numpy(noise)
        
        return image.astype("uint8"), label, latent, noise, record


    def forward(self, latent, noise): # [0, 1] in torch
        self.model.set_noise(self.model.parse_noise(noise))
        gen, seg = self.model(latent)

        gen = (1 + gen.clamp(-1, 1)) * 255 / 2
        gen = utils.torch2numpy(gen).transpose(0, 2, 3, 1)
        label = utils.torch2numpy(seg.argmax(1))
        return gen.astype("uint8"), label
