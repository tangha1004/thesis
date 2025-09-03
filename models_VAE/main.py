import sys
from train_test import train_test

if __name__ == "__main__":

    data_dir = str(sys.argv[1])
    result_dir = str(sys.argv[2])
    seed = int(sys.argv[3])

    omics = sys.argv[4].split(',')
    lr_omics = list(map(float, sys.argv[5].split(',')))
    n_epoch_omics = list(map(int, sys.argv[6].split(',')))

    lr_AE = float(sys.argv[7])
    lr_clf = float(sys.argv[8])
    patience = int(sys.argv[9])
    batch_size = int(sys.argv[10])
    wd_AE = float(sys.argv[11])
    wd_clf = float(sys.argv[12])

    cancers = sys.argv[13].split(',')
    main_cancer = str(sys.argv[14]) 
    num_subtypes = int(sys.argv[15])

    train_test(cancers, main_cancer, num_subtypes, omics, data_dir, result_dir, batch_size, lr_omics, n_epoch_omics,
               lr_AE, lr_clf, wd_AE, wd_clf, patience, seed)