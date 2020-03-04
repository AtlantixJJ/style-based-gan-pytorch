import torch
from lib.face_parsing.unet import unet
import utils
from lib.netdissect.segmenter import UnifiedParsingSegmenter

"""
Convert single label to multilabel
"""
def convert_multi_label(seg, cg, i):
    label = utils.torch2numpy(seg[0]).argmax(0)
    label[label > 0] += cg[i][0]
    return label


def get_segmenter(task, fpath=None, device="cuda"):
    if task == "bedroom" or task == "church":
        return UnifiedParsingSegmenter(device=device)
    elif task == "celebahq" or task == "ffhq":
        return CelebAMaskHQSegmenter(fpath, device)


class CelebAMaskHQSegmenter(object):
    def __init__(self, path="checkpoint/faceparse_unet_512.pth", device="cuda"):
        self.path = path
        self.device = device
        self.resolution = 128 if "128" in path else 512
        self.model = unet(train_size=self.resolution)
        state_dict = torch.load(path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.to(device).eval()
        del state_dict
        self.labels = utils.CELEBA_CATEGORY[1:]
        self.categories = ["face"] * len(self.labels)

    def get_label_and_category_names(self):
        return list(zip(self.labels, self.categories)), self.categories

    def segment_batch(self, batch, resize=True):
        seg = self.model(batch, resize)
        return seg.argmax(1)

    def segment_batch_logits(self, batch, resize=True):
        return self.model(batch, resize)