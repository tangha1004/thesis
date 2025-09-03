import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn.functional as F
from sklearn import metrics
import copy
from models_deepssc import create_autoencoder, Subtyping_model

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
                if self.verbose:
                    print(f'Early stop at epoch {epoch}th after {self.counter} epochs not increasing score from epoch {self.best_epoch}th with best score {self.best_score}')
        else:
            self.best_score = score
            self.save_checkpoint(metric_higher_better, epoch, model_dict)
            self.counter = 0

    def save_checkpoint(self, metric_higher_better, epoch, model_dict):
        self.best_weights = copy.deepcopy(model_dict)
        self.metric_higher_better_max = metric_higher_better
        self.best_epoch = epoch

def prepare_data(data_dir, batch_size, omics, cancers, main_cancer, num_subtypes, print_info=True):
    unlabel_omics_data = [1] * len(omics)
    X_train_omics_clf = []
    X_val_omics = []
    X_test_omics = []
    train_omics_unlabeled = []

    for i, omic in enumerate(omics):
        train_one_omic_unlabeled = []
        for cancer in cancers:
            tmp = []
            try:
                tmp = pd.read_csv(os.path.join(data_dir, f'{cancer}_{i+1}_tr_unlabeled.csv')).to_numpy()
                train_one_omic_unlabeled.append(tmp)
            except:
                if print_info:
                    print(f'No unlabel {omic.upper()} data found!')
                unlabel_omics_data[i] = 0
            
            train_one_omic_unlabeled.append(tmp)
            
            if (cancer == main_cancer):
                X_train_omics_clf.append(pd.read_csv(os.path.join(data_dir, f'{cancer}_{i+1}_tr.csv'), header=None).to_numpy())
                X_val_omics.append(pd.read_csv(os.path.join(data_dir, f'{cancer}_{i+1}_val.csv'), header=None).to_numpy())
                X_test_omics.append(pd.read_csv(os.path.join(data_dir, f'{cancer}_{i+1}_te.csv'), header=None).to_numpy())
        train_one_omic_unlabeled_concencated = np.vstack(train_one_omic_unlabeled)
        train_omics_unlabeled.append(train_one_omic_unlabeled_concencated)

    
    train_label = pd.read_csv(os.path.join(data_dir, f'{main_cancer}_labels_tr.csv'), header=None).to_numpy()
    val_label = pd.read_csv(os.path.join(data_dir, f'{main_cancer}_labels_val.csv'), header=None).to_numpy()
    test_label = pd.read_csv(os.path.join(data_dir, f'{main_cancer}_labels_te.csv'), header=None).to_numpy()

    y_train = torch.tensor(train_label, dtype=torch.int64).squeeze()
    y_val = torch.tensor(val_label, dtype=torch.int64).squeeze()
    y_test = torch.tensor(test_label, dtype=torch.int64).squeeze()

    with open(os.path.join(data_dir, f'{main_cancer}_dct_index_subtype.json')) as file_json_id_label:
        dct_LABEL_MAPPING_NAME = json.load(file_json_id_label)
    ord_subtype_name = list(dct_LABEL_MAPPING_NAME.values())
    if print_info:
        print('Classes: ', ord_subtype_name)

    count_label = y_train.unique(return_counts=True)[1].float()
    label_weight = count_label.sum() / count_label / num_subtypes
    if print_info:
        print('Weight for these classes:', label_weight)

    X_train_omics_AE = []
    for i in range(len(omics)):
        if unlabel_omics_data[i] == 1:
            print(X_train_omics_clf[i].shape)
            print(train_omics_unlabeled[i].shape)
            X_train_omics_AE.append(torch.tensor(np.vstack((X_train_omics_clf[i], train_omics_unlabeled[i])), dtype=torch.float32))
        else:
            X_train_omics_AE.append(torch.tensor(X_train_omics_clf[i], dtype=torch.float32))
    
    X_train_omics_clf = [torch.tensor(data, dtype=torch.float32) for data in X_train_omics_clf]
    X_val_omics = [torch.tensor(data, dtype=torch.float32) for data in X_val_omics]
    X_test_omics = [torch.tensor(data, dtype=torch.float32) for data in X_test_omics]

    train_omics_AE_ds = [TensorDataset(data) for data in X_train_omics_AE]
    val_omics_ds = [TensorDataset(data) for data in X_val_omics]
    train_clf_ds = TensorDataset(*X_train_omics_clf, y_train)
    val_clf_ds = TensorDataset(*X_val_omics, y_val)
    test_clf_ds = TensorDataset(*X_test_omics, y_test)

    if 'BRCA' in data_dir:
        batch_size_clf = batch_size * 2
    elif 'CRC' in data_dir:
        batch_size_clf = batch_size = int(batch_size / 2)
    else:
        batch_size_clf = batch_size

    train_loader_clf = DataLoader(train_clf_ds, batch_size=batch_size_clf, shuffle=True)
    train_loader_AE = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in train_omics_AE_ds]

    return (train_loader_clf, train_loader_AE), (test_clf_ds, val_clf_ds, val_omics_ds), label_weight, ord_subtype_name

def load_model_dict(omics, cancers, main_cancer, num_subtypes, data_dir, result_dir, batch_size, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    loader, dataset, label_weight, idx2class = prepare_data(data_dir, batch_size, omics, cancers, main_cancer, num_subtypes, print_info=True)
    test_clf_ds, val_clf_ds, val_omics_ds = dataset

    ae_models = []
    for i, omic in enumerate(omics):
        ae_model = create_autoencoder(omic, len(val_clf_ds[0][i]))
        ae_models.append(ae_model)

    clf = Subtyping_model(ae_models, len(label_weight))
    clf.to(device)

    checkpoint_path = os.path.join(result_dir, 'checkpoint.pt')
    if os.path.exists(checkpoint_path):
        clf.load_state_dict(torch.load(checkpoint_path))        
    else:
        print(f"Checkpoint file '{checkpoint_path}' does not exist.")
    
    return clf

def evaluate(model, testdata, idx2class, result_dir):
    model.eval()
    with torch.no_grad():
        inputs = [testdata[:][i].cuda() for i in range(len(testdata[0]) - 1)]
        preds = model(*inputs)
        preds = F.softmax(preds, dim=1)
        _, preds_label = torch.max(preds.data, dim=-1)

    preds = preds.cpu()
    preds_label = preds_label.data.cpu()
    if len(idx2class) == 2:    
        print('\nTest AUC:\n', metrics.roc_auc_score(testdata[:][-1], preds.data[:,1]))
    clf_report = metrics.classification_report(testdata[:][-1], 
                                            preds_label, 
                                            target_names=idx2class, 
                                            digits=4, 
                                            zero_division=0, 
                                            output_dict=True)
    clf_df = pd.DataFrame(clf_report)
    clf_df.loc[['precision', 'recall'],'accuracy']=np.nan
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figwidth(13)
    metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(testdata[:][-1], preds_label), 
                                display_labels=idx2class).plot(cmap='Blues', ax=ax1)
    sns.heatmap(clf_df.iloc[:-1, :].T, annot=True, cmap='Blues', robust=True, ax=ax2, fmt='.2%')
    
    plt.savefig(os.path.join(result_dir, 'val_results.png'))