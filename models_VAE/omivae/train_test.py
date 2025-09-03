import os
from re import I
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F
from models import VAE_multi, init_model_dict_multi, recon_loss, kl_loss, classifier_loss 
import copy

cuda = True if torch.cuda.is_available() else False

################## Util functions #################

def one_hot_tensor(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1,1), 1)    
    return y_onehot

def train_epoch(data_list, label, model, optimizer, k_view_list, k_kl, k_c, batch_size = 32):
    """
        assume all the matrices passed into this function are tensors on gpu
    """
    model.train()
    # train_methy_recon = 0
    # train_expr_recon = 0
    # train_kl = 0
    # train_classifier = 0
    # train_correct_num = 0
    # train_total_loss = 0
    # print(data_list)


    n_samples = len(label)
    indices = torch.randperm(n_samples)
    total_loss = 0
    num_batches = 0

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        batch_data_list = []
        for data in data_list:
            batch_data_list.append(data[batch_indices])
        # print(batch_data_list)
        batch_label = label[batch_indices]

        z, GE_x_hat, CNA_x_hat, mRNA_x_hat, mean, log_var, pred_y = model(batch_data_list)

        class_loss = classifier_loss(pred_y, batch_label)
        GE_recon_loss = recon_loss(GE_x_hat, batch_data_list[0])
        CNA_recon_loss = recon_loss(CNA_x_hat, batch_data_list[1])
        mRNA_recon_loss = recon_loss(mRNA_x_hat, batch_data_list[2])

        kl = kl_loss(mean, log_var)

        optimizer.zero_grad()
        loss = k_view_list[0] * GE_recon_loss + k_view_list[1] * CNA_recon_loss + k_view_list[2] * mRNA_recon_loss + k_kl * kl + k_c * class_loss
        loss = torch.mean(loss)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        num_batches += 1

    # train_methy_recon_ave = train_methy_recon / len(train_dataset)
    # train_expr_recon_ave = train_expr_recon / len(train_dataset)
    # train_kl_ave = train_kl / len(train_dataset)
    # train_classifier_ave = train_classifier / len(train_dataset)
    # train_accuracy = train_correct_num / len(train_dataset) * 100
    # train_total_loss_ave = train_total_loss / len(train_dataset)

    return total_loss / num_batches

# def test_epoch(data_list, model):
#     model.eval()
#     with torch.no_grad():
#         logit = model.infer(data_list)
#         prob = F.softmax(logit, dim=1).data.cpu().numpy()
#     return prob

def test_epoch(data_list, model, labels=None, report_metrics=False, batch_size=32):
    model.eval()
    
    n_samples = data_list[0].shape[0]
    all_pred_y = []
    all_latent_mean = []
    
    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_data = [x[start_idx:end_idx] for x in data_list]
            
            # Forward pass (handle both 2-view and 3-view cases)
            if len(data_list) > 2:
                _, _, _, _, mean, _, pred_y = model(batch_data)
            else:
                _, _, _, mean, _, pred_y = model(batch_data)
                
            all_pred_y.append(pred_y)
            all_latent_mean.append(mean)
    
    # Concatenate batches
    pred_y = torch.cat(all_pred_y, dim=0)
    latent_mean = torch.cat(all_latent_mean, dim=0)
    
    # Convert to probabilities
    prob = F.softmax(pred_y, dim=1).cpu().numpy()
    
    return prob, latent_mean.cpu().numpy()

def save_checkpoint(model, checkpoint_path, filename="VAE_multi.pt"):
    os.makedirs(checkpoint_path, exist_ok=True)
    filename = os.path.join(checkpoint_path, filename)
    torch.save(model, filename)

def load_checkpoint(model, path):
    best_checkpoint = torch.load(path)
    model.load_state_dict(best_checkpoint)

###################### Early stopping class ##################

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.001):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_weights = None
        self.best_epoch = None
        self.early_stop = False
        self.metric_higher_better_max = 0.0
        self.delta = delta

    def __call__(self, metric_higher_better, epoch, model_dict):

        score = metric_higher_better

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metric_higher_better, epoch, model_dict)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f'Early stop at epoch {epoch}th after {self.counter} epochs not increasing score from epoch {self.best_epoch}th with best score {self.best_score}')
        else:
            self.best_score = score
            self.save_checkpoint(metric_higher_better, epoch, model_dict)
            self.counter = 0

    def save_checkpoint(self, metric_higher_better, epoch, model_dict):
        self.best_weights = copy.deepcopy(model_dict)
        self.metric_higher_better_max = metric_higher_better
        self.best_epoch = epoch

###################### 2 main functions ##################
def prepare_trte_data(data_folder, view_list, postfix_tr='_tr', postfix_te='_val'):
    num_view = len(view_list)
    labels_tr = np.loadtxt(os.path.join(data_folder, f"labels{postfix_tr}.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, f"labels{postfix_te}.csv"), delimiter=',')

    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    for i in range(1, num_view+1):
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i)+f"{postfix_tr}.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i)+f"{postfix_te}.csv"), delimiter=','))
    
    eps = 1e-10
    X_train_min = [np.min(data_tr_list[i], axis=0, keepdims=True) for i in range(len(data_tr_list))]
    data_tr_list = [data_tr_list[i] - np.tile(X_train_min[i], [data_tr_list[i].shape[0], 1]) for i in range(len(data_tr_list))]
    data_te_list = [data_te_list[i] - np.tile(X_train_min[i], [data_te_list[i].shape[0], 1]) for i in range(len(data_tr_list))]
    X_train_max = [np.max(data_tr_list[i], axis=0, keepdims=True) + eps for i in range(len(data_tr_list))]
    data_tr_list = [data_tr_list[i] / np.tile(X_train_max[i], [data_tr_list[i].shape[0], 1]) for i in range(len(data_tr_list))]
    data_te_list = [data_te_list[i] / np.tile(X_train_max[i], [data_te_list[i].shape[0], 1]) for i in range(len(data_tr_list))]

    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr+num_te)))
    data_train_list = []
    data_all_list = []
    data_test_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                       data_tensor_list[i][idx_dict["te"]].clone()),0))
        data_test_list.append(data_tensor_list[i][idx_dict["te"]].clone())
    labels = np.concatenate((labels_tr, labels_te))
    return data_train_list, data_test_list, idx_dict, labels

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
def train_test(testonly, 
               data_folder, modelpath,
               view_list, 
               input_dim, latent_space_dim, 
               level_2_dim, level_3_dim,
               level_4_dim, 
               classifier_1_dim, num_class,
               num_epoch, lr, 
               batch_size,
               k_view_list, k_kl, k_c,
               test_interval=50,
               postfix_tr='_tr',
               postfix_te='_val',
               patience=7,
               verbose=False,seed=42):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    step_size = 500

    data_tr_list, data_test_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list, postfix_tr, postfix_te)
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    labels_tr_tensor = labels_tr_tensor.cuda()

    model_dict = init_model_dict_multi(view_list,
                    input_dim, latent_space_dim, 
                    level_2_dim, level_3_dim,
                    level_4_dim, 
                    classifier_1_dim, num_class)
    model = model_dict['VAE_multi']
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.2)

    if testonly == True:
        load_checkpoint(model, os.path.join(modelpath, data_folder, 'VAE_multi.pt'))
        test_labels = labels_trte[trte_idx["te"]]
        te_prob, latent_mean = test_epoch(data_test_list, model, batch_size=batch_size)
        

        print("Test ACC: {:.5f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
        print("Test F1 weighted: {:.5f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')))
        print("Test F1 macro: {:.5f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')))

    else:    

        print("\nTraining...")
        if patience is not None:
            early_stopping = EarlyStopping(patience = patience, verbose = verbose)

        best_model_dict = None
        for epoch in range(num_epoch+1):

            # train_epoch(data_tr_list, labels_tr_tensor, model, optimizer)
            train_epoch(data_tr_list, labels_tr_tensor, model, optimizer, k_view_list, k_kl, k_c, batch_size)
            scheduler.step()
            test_labels = labels_trte[trte_idx["te"]]
            te_prob, _ = test_epoch(data_test_list, model, batch_size=batch_size)
    

            if verbose == 'True':
                if epoch % test_interval == 0:
                    print("\nTest: Epoch {:d}".format(epoch))
                    print("Test F1 weighted: {:.5f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')))
                    print("Test F1 macro: {:.5f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')))
            if patience is not None:
                if num_class == 2:
                    score = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                else:
                    score = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')

                early_stopping(score, epoch, model)
                best_model_dict = early_stopping.best_weights
                
                if early_stopping.early_stop:
                    print(f"Early stopping triggered. Using best model from epoch {early_stopping.best_epoch}")

                    model = best_model_dict 
                    break

        if early_stopping is not None:
            if early_stopping.early_stop:
                model = early_stopping.best_weights
            else:
                early_stopping.best_weights = copy.deepcopy(model)
                model = early_stopping.best_weights

    return model