import torch
import os
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
            # Add weights_only=False for PyTorch 2.6 compatibility
            loaded_obj = torch.load(
                os.path.join(folder, module+".pt"),
                map_location="cuda:{:}".format(torch.cuda.current_device()),
                weights_only=False
            )
            
            # Check if loaded object is a state_dict or a model
            if isinstance(loaded_obj, dict):
                # It's a state_dict, use load_state_dict directly
                model_dict[module].load_state_dict(loaded_obj)
            else:
                # It's a complete model, extract its state_dict
                model_dict[module].load_state_dict(loaded_obj.state_dict())
                
            print(f"Module {module} loaded successfully!")
        else:
            print("WARNING: Module {:} from model_dict is not loaded!".format(module))
    return model_dict
