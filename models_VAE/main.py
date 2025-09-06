import sys
import os
import ast
import random
import numpy as np
import torch
from models_VAE.models_deepssc import save_model_dict
from models_VAE.train_test_deepssc import train_test

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'cuda: {cuda}')

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

rseed = 42
set_seed(rseed)

if __name__ == "__main__":
    data_dir = str(sys.argv[1])
    result_dir = str(sys.argv[2])
    batch_size = int(sys.argv[3])
    batch_size_clf = int(sys.argv[4])
    view_list = list(map(int, sys.argv[5].strip("[]").split(",")))
    num_subtypes = int(sys.argv[6])
    hidden_dims = ast.literal_eval(sys.argv[7])
    n_samples =  list(map(int, sys.argv[8].strip("[]").split(",")))
    dec_vars = list(map(int, sys.argv[9].strip("[]").split(",")))
    latent_dims = list(map(int, sys.argv[10].strip("[]").split(",")))
    hidden_dim_cls =  int(sys.argv[11])
    num_class = int(sys.argv[12])
    dropout_rate_cls = float(sys.argv[13])
    epoch_pretrain_vaes = list(map(int, sys.argv[14].strip("[]").split(",")))
    lr_pretrain_vaes = list(map(int, sys.argv[15].strip("[]").split(",")))
    epoch_cls = int(sys.argv[16])
    patience = int(sys.argv[17])
    lr_clf = float(sys.argv[18])
    lr_vae = float(sys.argv[19])
    wd_clf = float(sys.argv[20])
    wd_vae = float(sys.argv[21])
    seed = int(sys.argv[22])
    folder = str(sys.argv[23])

    model_dict = train_test(data_dir, result_dir, 
               batch_size, batch_size_clf, view_list, num_subtypes, 
               hidden_dims, n_samples, dec_vars, latent_dims,
               hidden_dim_cls, num_class, dropout_rate_cls,
               epoch_pretrain_vaes, lr_pretrain_vaes,
               epoch_cls, patience, lr_clf, lr_vae, wd_clf, wd_vae, seed)

    save_model_dict(folder, model_dict)