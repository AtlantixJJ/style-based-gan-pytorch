import torch
import numpy as np
from model.tf import StyledGenerator

generator = StyledGenerator()
state_dict = torch.load("checkpoint/karras2019stylegan-ffhq-1024x1024.for_g_all.pt")
