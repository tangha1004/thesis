import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
import copy
from utils import EarlyStopping, create_autoencoder, prepare_data, evaluate
from models_deepssc import Subtyping_model

def train_clf(model, class_weight, train_loader, val_dataset, epoch, patience, lr_clf, lr_ae, wd_clf, wd_ae):

    loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    opt = optim.Adam([
                        {'params':model.classifier.parameters(), 'lr':lr_clf, 'weight_decay':wd_clf},
                        *[
                            {'params': ae_repr.parameters()}
                            for ae_repr in model.omics_repr
                        ]
                     ], lr=lr_ae, weight_decay=wd_ae)
    early_stopping = EarlyStopping(patience = patience, verbose = True)
    
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

            loss.backward()
            opt.step()
            opt.zero_grad()

            _, preds_label = torch.max(preds.data, dim=-1)
            train_loss += loss.item()
            train_f1_macro += f1_score(yb.data.tolist(),preds_label.tolist(), average='macro')
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
            val_f1_macro = f1_score(yb.data.tolist(),preds_label.tolist(), average='macro')

        val_loss_his.append(val_loss)
        val_f1_macro_his.append(val_f1_macro)
        model_dict = copy.deepcopy(model.state_dict())
        early_stopping(val_f1_macro, ep, model_dict)
        if early_stopping.early_stop:
            model.load_state_dict(early_stopping.best_weights)
            break

    return (train_loss_his, train_f1_macro_his), (val_loss_his, val_f1_macro_his)
    
def train_AE(model, train_loader, val_dataset, epoch, lr):

    loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    train_loss_his = []
    val_loss_his = []
    for ep in range(epoch):
        model.train()
        train_loss = 0.0
        nb = 0
        for xb in train_loader:
            xb = xb[0].cuda()

            preds = model(xb)
            loss = loss_fn(preds, xb)

            loss.backward()
            opt.step()
            opt.zero_grad()

            train_loss += loss.item()
            nb += 1

        train_loss_his.append(train_loss/nb)

        model.eval()
        with torch.no_grad():
            xb = val_dataset[:][0].cuda()

            val_preds = model(xb)
            val_loss = loss_fn(val_preds, xb)


        val_loss_his.append(val_loss)
    return train_loss_his, val_loss_his

def train_test(cancers, main_cancer, num_subtypes, omics, data_dir, result_dir, batch_size, lr_omics, n_epoch_omics, 
               lr_AE, lr_clf, wd_AE, wd_clf, patience, seed=42):

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    print('Loading data...')
    loader, dataset, label_weight, idx2class = prepare_data(data_dir, batch_size, omics, cancers, main_cancer, num_subtypes)
    train_clf_dl, train_omic_AE_dl = loader
    test_clf_ds, val_clf_ds, val_omics_ds = dataset
    print('Create result directory...')
    try:
        os.makedirs(result_dir)
    except:
        print('Result directory already exist!')
    
    ae_models = []
    for i, omic in enumerate(omics):
        print(f'Training DAE for {omic.upper()} data...')
        ae_model = create_autoencoder(omic, len(val_clf_ds[0][i]))
        ae_model.to(device)
        train_his, val_his = train_AE(ae_model, train_omic_AE_dl[i], val_omics_ds[i], n_epoch_omics[i], lr_omics[i])
        plt.plot(train_his, label='train')
        plt.plot(val_his, label='validation')
        plt.legend()
        plt.savefig(os.path.join(result_dir, f'{omic}_training.png'))
        plt.clf()
        ae_models.append(ae_model)
    
    print('\nTraining fusion model...\n')
    clf = Subtyping_model(ae_models, len(label_weight))
    clf.to(device)
    label_weight = label_weight.to(device)

    clf_train_his, clf_val_his = train_clf(clf, label_weight, 
                                        train_clf_dl, val_clf_ds, 
                                        200, patience, lr_clf, 
                                        lr_AE, wd_clf, wd_AE)
    plt.plot(clf_train_his[0], label='train loss')
    plt.plot(clf_val_his[0], label='validation loss')
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'fusion_loss.png'))
    plt.clf()
    plt.plot(clf_train_his[1], label='train f1 macro')
    plt.plot(clf_val_his[1], label='validation f1 macro')
    plt.legend()
    evaluate(clf, val_clf_ds, idx2class, result_dir)
    # save model weights
    torch.save(clf.state_dict(), os.path.join(result_dir, 'checkpoint.pt'))
    print('Please check results in your result folder!')