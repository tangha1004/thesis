import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
import copy
from utils import EarlyStopping, prepare_data, evaluate
from models_deepssc import Subtyping_model, VariationalAutoencoder, save_model_dict, init_model_dict


def train_VAE(model, train_loader, val_dataset, epoch, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loss_his = []
    val_loss_his = []
    for ep in range(epoch):
        model.train()
        tmp_train_total_loss = 0.0
        nb = 0
        for data in train_loader:
            x_data_train = data[0].cuda()
            loss = model.get_loss(x_data_train)
            tmp_total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            nb += 1
        train_loss_his.append(tmp_train_total_loss/nb)

        model.eval()
        with torch.no_grad():
            x_data_val = val_dataset[:][0].cuda()
            val_loss = model.get_loss(x_data_val)
        val_loss_his.append(val_loss)

    return train_loss_his, val_loss_his

def train_clf(model, class_weight, train_loader, val_dataset, epoch, patience, lr_clf, lr_vae, wd_clf, wd_ae):
    param_groups = [{'params': model.classifier.parameters(), 'lr': lr_clf, 'weight_decay': wd_clf}]
    for vae_model in model.vae_models:
        param_groups.append({'params': vae_model.parameters(), 'lr': lr_vae, 'weight_decay': wd_ae})

    loss_fn = nn.CrossEntropyLoss(weight=class_weight)
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
            preds = model(*x_data)
            loss = loss_fn(preds, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
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

'''
    Step 1: prepare data
       def prepare_data(data_dir, batch_size, batch_size_clf, omics, num_subtypes, print_info=True)
    Step 2: init model
        class VariationalAutoencoder(nn.Module):
            def __init__(self, input_dim, hidden_dims, n_samples, dec_var, latent_dim)

        class Subtyping_model(nn.Module):
            def __init__(self, vae_models, hidden_dim_cls, num_class, dropout_rate_cls = 0.3)

    Step 3: pretraining vae
        def train_VAE(model, train_loader, val_dataset, epoch, lr)

    Step 4: def train_clf(model, class_weight, train_loader, val_dataset, epoch, patience, lr_clf, lr_vae, wd_clf, wd_ae)
'''

def train_test(data_dir, result_dir, 
               batch_size, batch_size_clf, view_list, num_subtypes, 
               hidden_dims_omics, n_samples, dec_vars, latent_dims,
               hidden_dim_cls, num_class, dropout_rate_cls,
               epoch_pretrain_vaes, lr_pretrain_vaes,
               epoch_cls, patience, lr_clf, lr_vae, wd_clf, wd_vae,
            #    postfix_tr='_tr',
            #    postfix_te='_val',
            #    verbose=False, 
                seed=42):
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
    train_loader, dataset, label_weight, idx2class = prepare_data(data_dir, batch_size, batch_size_clf, view_list, num_subtypes)
    train_clf_dl, train_omic_AE_dl = train_loader
    test_clf_ds, val_clf_ds, val_omics_ds = dataset
    
    print('Create result directory...')
    try:
        os.makedirs(result_dir)
    except:
        print('Result directory already exists!')
    
    vae_models = []
    for i, omic in enumerate(view_list):
        print(f'Training VAE for {omic.upper()} data...')
        input_dim_omic = len(val_clf_ds[0][i])
        vae_model_omic = VariationalAutoencoder(input_dim_omic, hidden_dims_omics[i], n_samples[i], dec_vars[i], latent_dims[i])
        vae_model_omic.to(device)
        train_his, val_his = train_VAE(vae_model_omic, train_omic_AE_dl[i], val_omics_ds[i], epoch_pretrain_vaes[i], lr_pretrain_vaes[i])

        plt.figure()
        plt.plot(train_his, label='train')
        plt.plot(val_his, label='validation')
        plt.legend()
        plt.savefig(os.path.join(result_dir, f'{omic}_training.png'))
        plt.clf()
        
        vae_models.append(vae_model_omic)
    
    print('\nTraining fusion model with VAE latent representations...\n')
    clf = Subtyping_model(vae_models, hidden_dim_cls, num_class, dropout_rate_cls)
    clf.to(device)
    label_weight = label_weight.to(device)
    
    print("Training...")
    clf_train_his, clf_val_his = train_clf(clf, label_weight, 
                                            train_clf_dl, val_clf_ds, 
                                         epoch_cls, patience, lr_clf, lr_vae, wd_clf, wd_vae)
    
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