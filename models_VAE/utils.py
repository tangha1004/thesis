import torch
import os
import numpy as np
import torch.nn.functional as F

cuda = True if torch.cuda.is_available() else False

def save_model_dict(folder, model_dict):
    if not os.path.exists(folder):
        os.makedirs(folder)
    if isinstance(model_dict, dict):
        # It's a dictionary of models
        for module in model_dict:
            torch.save(model_dict[module], os.path.join(folder, module + '.pt'))
    else:
        # It's a single model
        torch.save(model_dict, os.path.join(folder, 'VAE_multi.pt'))

def load_model_dict(folder, model_dict):
    for module in model_dict:
        if os.path.exists(os.path.join(folder, module+".pt")):
#            print("Module {:} loaded!".format(module))
            model_dict[module].load_state_dict(torch.load(os.path.join(folder, module+".pt"), map_location="cuda:{:}".format(torch.cuda.current_device())))
        else:
            print("WARNING: Module {:} from model_dict is not loaded!".format(module))
        if cuda:
            model_dict[module].cuda()    
    return model_dict
