import model.tf
import model.inception
import model.simple
import model.semantic_extractor
import model.vgg16
import subprocess
uname = subprocess.run(["uname", "-a"], capture_output=True)
uname = uname.stdout.decode("ascii")
if "Linux" not in uname:
    import model.stylegan2
    from model.stylegan2 import from_pth_file as load_stylegan2
#from model.simple import from_pth_file as load_dcgan
from model.tf import from_pth_file as load_stylegan
from lib.netdissect.proggan import from_pth_file as load_proggan

def load_model(fpath):
    if "proggan" in fpath:
        return load_proggan(fpath)
    elif "stylegan2" in fpath:
        return load_stylegan2(fpath)
    elif "stylegan" in fpath:
        return load_stylegan(fpath)

def load_model_from_pth_file(model_name, fpath):
    if "proggan" in model_name:
        return load_proggan(fpath)
    elif "stylegan2" in model_name:
        return load_stylegan2(fpath)
    elif "stylegan" in model_name:
        return load_stylegan(fpath)
    #elif "dcgan" in model_name:
    #    return load_dcgan(fpath)



