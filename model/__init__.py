import model.tf
import model.inception
import model.simple
import model.semantic_extractor
#try:
#    import model.stylegan2
#    from model.stylegan2 import from_pth_file as load_stylegan2
#except:
#    pass
#from model.simple import from_pth_file as load_dcgan
from model.tf import from_pth_file as load_stylegan
from lib.netdissect.proggan import from_pth_file as load_proggan

def load_model_from_pth_file(model_name, fpath):
    if "proggan" in model_name:
        return load_proggan(fpath)
    elif "stylegan2" in model_name:
        return load_stylegan2(fpath)
    elif "stylegan" in model_name:
        return load_stylegan(fpath)
    #elif "dcgan" in model_name:
    #    return load_dcgan(fpath)



