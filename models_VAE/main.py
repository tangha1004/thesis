import sys
from train_test import train_test
from utils import save_model_dict

"""
    Paramaters for VAE_multi: 
                    self, view_list,
                    input_dim, latent_space_dim, 
                    level_2_dim, level_3_dim,
                    level_4_dim, 
                    classifier_1_dim, class_num
    Parameters for train_epoch:
        data_list, label, model, optimizer, k_view_list, k_kl, k_c, batch_size
"""

if __name__ == "__main__":
    view_list = list(map(int, sys.argv[1].strip("[]").split(",")))
    hidden_dim = list(map(int, sys.argv[2].strip("[]").split(",")))
    input_dim = list(map(int, sys.argv[3].strip("[]").split(",")))
    latent_space_dim = int(sys.argv[4])
    level_2_dim = list(map(int, sys.argv[5].strip("[]").split(",")))
    level_3_dim = list(map(int, sys.argv[6].strip("[]").split(",")))
    level_4_dim = int(sys.argv[7]) 
    classifier_1_dim = int(sys.argv[8]) 
    class_num = int(sys.argv[9]) 

    print_hyper = str(sys.argv[10])
    verbose = str(sys.argv[11])
    testonly = str(sys.argv[12])

    data_folder = str(sys.argv[13])
    model_path = str(sys.argv[14])
    saved_model_dict_folder = str(sys.argv[15])

    num_epoch = int(sys.argv[16])
    lr = float(sys.argv[17])
    batch_size = int(sys.argv[18])
    patience = int(sys.argv[19])
    k_view_list = list(map(float, sys.argv[20].strip("[]").split(",")))
    k_kl = float(sys.argv[21])
    k_c = float(sys.argv[22])

    if (print_hyper == 'True'):
        print(
            f"""
            Config:
                * Data
                - Data Folder
                    = {data_folder}
                - Saved Model Loc
                    = {model_path}
                - List Views
                    = {view_list}

                * Architecture
                - Input Dimensions
                    = {input_dim}
                - Latent Space Dim
                    = {latent_space_dim}
                - Level 2 Dim
                    = {level_2_dim}
                - Level 3 Dim
                    = {level_3_dim}
                - Level 4 Dim
                    = {level_4_dim}
                - Classifier Dim
                    = {classifier_1_dim}
                - Class Number
                    = {class_num}
                - Hidden Dim
                    = {hidden_dim}

                * Training
                - Num Epoch
                    = {num_epoch}
                - Learning Rate
                    = {lr}
                - Batch Size
                    = {batch_size}
                - View Weights
                    = {k_view_list}
                - KL Weight
                    = {k_kl}
                - Classifier Weight
                    = {k_c}

                * Other
                - Test Only
                    = {testonly}
                - Patience
                    = {patience}
                - Verbose   
                    = {verbose}
            """
        )

    if "BRCA" in data_folder:
        num_class = 5
    if "GBM" in data_folder:
        num_class = 4
    if "Lung" in data_folder:
        num_class = 2
    if "CRC" in data_folder:
        num_class = 4

    # Call train_test with proper parameters matching the function signature
    model_dict = train_test(
        testonly=testonly,
        data_folder=data_folder, 
        modelpath=model_path,
        view_list=view_list,
        input_dim=input_dim, 
        latent_space_dim=latent_space_dim,
        level_2_dim=level_2_dim, 
        level_3_dim=level_3_dim,
        level_4_dim=level_4_dim,
        classifier_1_dim=classifier_1_dim, 
        num_class=num_class,
        num_epoch=num_epoch, 
        lr=lr,
        batch_size=batch_size,
        k_view_list=k_view_list,
        k_kl=k_kl, 
        k_c=k_c,
        patience=patience,
        verbose=verbose,
        hidden_dim=hidden_dim
    )

    save_model_dict(model_path, model_dict)