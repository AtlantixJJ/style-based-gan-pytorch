import torch
from home import stylegan, stylegan2, proggan
from home.semantic_extractor import get_semantic_extractor
from home.optim import edit_image_stroke, edit_label_stroke
from home import utils



class WrapedStyledGenerator(torch.nn.Module):
    def __init__(self, resolution=1024, method="", model_path="", n_class=16, category_groups=None, extractor="", sep_model_path="", gpu=-1):
        super(WrapedStyledGenerator, self).__init__()
        self.device = 'cuda' if gpu >= 0 else 'cpu'
        self.model_path = model_path
        self.method = method
        self.extractor = extractor
        self.sep_model_path = sep_model_path
        self.external_model = None
        self.category_groups = category_groups
        self.n_class = n_class

        print("=> Constructing network architecture")
        if "proggan" in self.model_path:
            self.model = proggan.from_pth_file(self.model_path)
        elif "stylegan2" in self.model_path:
            self.model = stylegan2.from_pth_file(self.model_path)
        elif "stylegan" in self.model_path:
            self.model = stylegan.from_pth_file(self.model_path)
        print("=> Loading parameter from %s" % self.model_path)
        state_dict = torch.load(self.model_path, map_location='cpu')
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
        self.noise_length = self.model.set_noise(None)
        print("=> Optimization method %s" % str(self.method))

        self.latent_param = torch.randn(1, 512, requires_grad=True, device=self.device)

        with torch.no_grad():
            image, stage = self.model.get_stage(self.latent_param)
            dims = [s.shape[1] for s in stage]
            print(image.shape, dims)
        func = get_semantic_extractor(self.extractor)
        if category_groups is None:
            self.sep_model = func(
                n_class=n_class,
                dims=dims)
        else:
            self.sep_model = func(
                n_class=n_class,
                category_groups=category_groups,
                dims=dims)
        self.sep_model.to(self.device).eval()
        state_dict = torch.load(sep_model_path, map_location='cpu')
        missed = self.sep_model.load_state_dict(state_dict)

    def generate_noise(self):
        print(self.noise_length)
        sizes = [4 * 2 ** (i // 2) for i in range(self.noise_length)]
        length = sum([size ** 2 for size in sizes])
        latent = torch.randn(1, 512, device=self.device)
        noise_vec = torch.randn((length,), device=self.device)
        return latent, noise_vec

    def generate_given_image_stroke(self, latent, noise, image_stroke, image_mask):
        utils.copy_tensor(self.latent_param, latent)
        self.mix_latent_param = self.latent_param.expand(self.noise_length, -1).detach()
        noises = self.model.parse_noise(noise)

        if "ML" in self.method:
            self.param = self.mix_latent_param
        else:
            self.param = self.latent_param

        image, label, latent, noises, record = edit_image_stroke(
            model=self.model, latent=self.latent_param, noises=noises, 
            image_stroke=image_stroke, image_mask=image_mask,
            method=self.method,
            sep_model=self.sep_model, mapping_network=self.mapping_network)

        # Currently no modification to noise
        # noise = torch.cat([n.view(-1) for n in noise])

        image = utils.torch2numpy(image * 255).transpose(0, 2, 3, 1)
        label = utils.torch2numpy(label)
        latent = utils.torch2numpy(latent)
        noise = utils.torch2numpy(noise)

        return image.astype("uint8"), label, latent, noise, record

    def generate_given_label_stroke(self, latent, noise, label_stroke, label_mask):
        utils.copy_tensor(self.latent_param, latent)
        self.mix_latent_param = self.latent_param.expand(self.noise_length, -1).detach()
        if "ML" in self.method:
            self.param = self.mix_latent_param
        else:
            self.param = self.latent_param
        noises = self.model.parse_noise(noise)

        image, label, latent, noises, record = edit_label_stroke(
            model=self.model, latent=self.param, noises=noises, label_stroke=label_stroke, label_mask=label_mask,
            method=self.method.replace("image", "label"),
            sep_model=self.sep_model, mapping_network=self.mapping_network)
        
        # Currently no modification to noise
        # noise = torch.cat([n.view(-1) for n in noise])

        image = utils.torch2numpy(image * 255).transpose(0, 2, 3, 1)
        label = utils.torch2numpy(label)
        latent = utils.torch2numpy(latent)
        noise = utils.torch2numpy(noise)
        
        return image.astype("uint8"), label, latent, noise, record


    def forward(self, latent, noise): # [0, 1] in torch
        self.model.set_noise(self.model.parse_noise(noise))
        image, stage = self.model.get_stage(latent)
        seg = self.sep_model(stage)[0]

        image = (1 + image.clamp(-1, 1)) * 255 / 2
        image = utils.torch2numpy(image).transpose(0, 2, 3, 1)
        label = utils.torch2numpy(seg.argmax(1))
        return image.astype("uint8"), label
