import torch
from lib.face_parsing.unet import unet
import utils
from lib.netdissect.segmenter import UnifiedParsingSegmenter


def get_segmenter(task, fpath=None):
    if task == "bedroom":
        return UnifiedParsingSegmenter()
    elif task == "celebahq":
        return CelebAMaskHQSegmenter(fpath)


class CelebAMaskHQSegmenter(object):
    def __init__(self, path="checkpoint/faceparse_unet_512.pth"):
        self.path = path
        self.resolution = 128 if "128" in path else 512
        self.model = unet(train_size=self.resolution)
        state_dict = torch.load(path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.cuda().eval()
        del state_dict
        self.idmap = utils.CelebAIDMap()
        self.labels = utils.CELEBA_REDUCED_CATEGORY
        self.categories = ["face"] * len(self.labels)
    
    def get_label_and_category_names(self):
        return self.labels, self.categories

    def segment_batch(self, batch):
        seg = self.model(batch)
        return self.idmap.mapid(seg.argmax(1))