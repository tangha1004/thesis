import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
import copy
from utils import EarlyStopping, prepare_data, evaluate
from models_deepssc import Subtyping_model, VariationalAutoencoder, save_model_dict
def train_clf(model, class_weight, train_loader, val_dataset, epoch, patience, lr_clf, lr_vae, wd_clf, wd_ae):
    loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    param_groups = [{'params': model.classifier.parameters(), 'lr': lr_clf, 'weight_decay': wd_clf}]

    for vae_model in model.vae_models:
        param_groups.append({'params': vae_model.parameters(), 'lr': lr_vae, 'weight_decay': wd_ae})
    
    opt = optim.Adam(param_groups)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    train_loss_his = []
    train_f1_macro_his = []
    val_loss_his = []
    val_f1_macro_his = []
    
    for ep in range(epoch):
        model.train()
        
        train_loss = 0.0
        train_f1_macro = 0.0
        nb = 0
        for data in train_loader:
            x_data = [d.cuda() for d in data[:-1]]
            yb = data[-1].cuda()
            
            # Forward pass through Subtyping_model
            preds = model(*x_data)
            loss = loss_fn(preds, yb)
            
            loss.backward()
            opt.step()
            opt.zero_grad()
            
            _, preds_label = torch.max(preds.data, dim=-1)
            train_loss += loss.item()
            train_f1_macro += f1_score(yb.data.tolist(), preds_label.tolist(), average='macro')
            nb += 1
            
        train_loss_his.append(train_loss/nb)
        train_f1_macro_his.append(train_f1_macro/nb)
        
        model.eval()
        with torch.no_grad():
            val_data = [d.cuda() for d in val_dataset[:][:-1]]
            yb = val_dataset[:][-1].cuda()
            val_preds = model(*val_data)
            val_loss = loss_fn(val_preds, yb)
            
            _, preds_label = torch.max(val_preds.data, dim=-1)
            val_f1_macro = f1_score(yb.data.tolist(), preds_label.tolist(), average='macro')
            
        val_loss_his.append(val_loss)
        val_f1_macro_his.append(val_f1_macro)
        model_dict = copy.deepcopy(model.state_dict())
        early_stopping(val_f1_macro, ep, model_dict)
        
        print(f'Epoch {ep+1}/{epoch}, Train Loss: {train_loss/nb:.4f}, Train F1: {train_f1_macro/nb:.4f}, '
              f'Val Loss: {val_loss.item():.4f}, Val F1: {val_f1_macro:.4f}')
        
        if early_stopping.early_stop:
            print(f'Early stopping at epoch {ep+1}')
            model.load_state_dict(early_stopping.best_weights)
            break
            
    return (train_loss_his, train_f1_macro_his), (val_loss_his, val_f1_macro_his)
    
def train_VAE(model, train_loader, val_dataset, epoch, lr, n_samples=5, dec_var=0.5):
    opt = optim.Adam(model.parameters(), lr=lr)
    train_loss_his = []
    val_loss_his = []

    for ep in range(epoch):
        model.train()
        train_loss = 0.0
        nb = 0
        for xb in train_loader:
            xb = xb[0].cuda()
            loss = model.get_loss(xb)
            loss.backward()
            opt.step()
            opt.zero_grad()
            
            train_loss += loss.item()
            nb += 1
        
        train_loss_his.append(train_loss/nb)
        model.eval()
        with torch.no_grad():
            xb = val_dataset[:][0].cuda()
            val_loss = model.get_loss(xb)
        val_loss_his.append(val_loss.item())
        print(f'Epoch {ep+1}/{epoch}, Train Loss: {train_loss/nb:.4f}, Val Loss: {val_loss.item():.4f}')

    return train_loss_his, val_loss_his

def train_test(cancers, main_cancer, num_subtypes, omics, data_dir, result_dir, batch_size, lr_omics, n_epoch_omics, 
               lr_AE, lr_clf, wd_AE, wd_clf, patience, seed=42, n_samples=5, dec_var=0.5):
    """Train VAE models and Subtyping_model"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        dev = "cuda:0"
        print("cuda: True")
    else:
        dev = "cpu"
        print("cuda: False")
    device = torch.device(dev)
    
    print("Random seed set as", seed)
    
    print('Loading data...')
    loader, dataset, label_weight, idx2class = prepare_data(data_dir, batch_size, omics, cancers, main_cancer, num_subtypes)
    train_clf_dl, train_omic_AE_dl = loader
    test_clf_ds, val_clf_ds, val_omics_ds = dataset
    
    print('Create result directory...')
    try:
        os.makedirs(result_dir)
    except:
        print('Result directory already exists!')
    
    # Train VAE models for each omic
    vae_models = []
    for i, omic in enumerate(omics):
        print(f'Training VAE for {omic.upper()} data...')
        input_dim = len(val_clf_ds[0][i])
        hidden_dims = [1024, 512]  # You can adjust these dimensions as needed
        
        # Create and initialize VAE model
        vae_model = VariationalAutoencoder(input_dim, hidden_dims, dropout_rate=0.5, latent_dim=128)
        # Set Monte Carlo sampling parameters
        vae_model.n_samples = n_samples
        vae_model.dec_var = dec_var
        vae_model.to(device)
        
        # Train VAE model
        train_his, val_his = train_VAE(vae_model, train_omic_AE_dl[i], val_omics_ds[i], 
                                      n_epoch_omics[i], lr_omics[i], n_samples, dec_var)
        
        # Plot training history
        plt.figure()
        plt.plot(train_his, label='train')
        plt.plot(val_his, label='validation')
        plt.legend()
        plt.savefig(os.path.join(result_dir, f'{omic}_training.png'))
        plt.clf()
        
        vae_models.append(vae_model)
    
    print('\nTraining fusion model with VAE latent representations...\n')
    # Create Subtyping_model with VAE models
    clf = Subtyping_model(vae_models, hidden_dim_cls=512, num_class=len(label_weight), dropout_rate_cls=0.3)
    clf.to(device)
    label_weight = label_weight.to(device)
    
    print("Training...")
    # Train classifier using VAE latent representations
    clf_train_his, clf_val_his = train_clf(clf, label_weight, 
                                         train_clf_dl, val_clf_ds, 
                                         200, patience, lr_clf, 
                                         lr_AE, wd_clf, wd_AE)
    
    plt.figure()
    plt.plot(clf_train_his[0], label='train loss')
    plt.plot(clf_val_his[0], label='validation loss')
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'fusion_loss.png'))
    plt.clf()
    
    plt.figure()
    plt.plot(clf_train_his[1], label='train f1 macro')
    plt.plot(clf_val_his[1], label='validation f1 macro')
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'fusion_f1.png'))
    
    evaluate(clf, test_clf_ds, idx2class, result_dir)
    save_model_dict(result_dir, clf)
    print('Results saved in the result folder!')
    
    return clf