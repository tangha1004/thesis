import torch.nn as nn
import torch
import torch.nn.functional as F

# class_num = 5
# latent_space_dim = 128
# input_dim_expr = 1000
# level_2_dim_expr = 512
# level_3_dim_expr = 256
# # level_4_dim = 128
# classifier_1_dim = 128
# classifier_out_dim = class_num
parallel = False

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)

def recon_loss(recon_x, x):
    loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    return loss

def kl_loss(mean, log_var):
    loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return loss

def classifier_loss(pred_y, y):
    loss = F.cross_entropy(pred_y, y, reduction='sum')
    return loss

def init_model_dict_single(input_dim_expr, latent_space_dim, 
                    level_2_dim_expr, level_3_dim_expr, 
                    # level_4_dim, 
                    classifier_1_dim, class_num):
    model_dict = {}
    model_dict['VAE_single'] = VAE_single(input_dim_expr, latent_space_dim, 
                    level_2_dim_expr, level_3_dim_expr, 
                    classifier_1_dim, class_num)
    model_dict['VAE_single'].apply(xavier_init)
    return model_dict

def init_model_dict_multi(view_list,
                    input_dim, latent_space_dim, 
                    level_2_dim, level_3_dim,
                    level_4_dim, 
                    classifier_1_dim, class_num):
    model_dict = {}
    model_dict['VAE_multi'] = VAE_multi(view_list,
                    input_dim, latent_space_dim, 
                    level_2_dim, level_3_dim,
                    level_4_dim, 
                    classifier_1_dim, class_num)
    model_dict['VAE_multi'].apply(xavier_init)
    return model_dict

class VAE_single(nn.Module):
    def __init__(self, input_dim_expr, latent_space_dim, 
                    level_2_dim_expr, level_3_dim_expr, 
                    # level_4_dim, 
                    classifier_1_dim, class_num):
        super(VAE_single, self).__init__()
        self.e_fc1_expr = self.fc_layer(input_dim_expr, level_2_dim_expr)
        self.e_fc2_expr = self.fc_layer(level_2_dim_expr, level_3_dim_expr)
        # self.e_fc3 = self.fc_layer(level_3_dim_expr, level_4_dim)
        self.e_fc4_mean = self.fc_layer(level_3_dim_expr, latent_space_dim, activation=0)
        self.e_fc4_log_var = self.fc_layer(level_3_dim_expr, latent_space_dim, activation=0)

        if parallel:
            self.e_fc1_expr.to('cuda:0')
            self.e_fc2_expr.to('cuda:0')
            # self.e_fc3.to('cuda:0')
            self.e_fc4_mean.to('cuda:0')
            self.e_fc4_log_var.to('cuda:0')

        self.d_fc4 = self.fc_layer(latent_space_dim, level_3_dim_expr)
        # self.d_fc3 = self.fc_layer(level_4_dim, level_3_dim_expr)
        self.d_fc2_expr = self.fc_layer(level_3_dim_expr, level_2_dim_expr)
        self.d_fc1_expr = self.fc_layer(level_2_dim_expr, input_dim_expr, activation=2)

        if parallel:
            self.d_fc4.to('cuda:1')
            # self.d_fc3.to('cuda:1')
            self.d_fc2_expr.to('cuda:1')
            self.d_fc1_expr.to('cuda:1')

        self.c_fc1 = self.fc_layer(latent_space_dim, classifier_1_dim)
        self.c_fc2 = self.fc_layer(classifier_1_dim, class_num, activation=0)
        if parallel:
            self.c_fc1.to('cuda:1')
            self.c_fc2.to('cuda:1')

    def fc_layer(self, in_dim, out_dim, activation=1, dropout=False, dropout_p=0.5):
        if activation == 0:
            layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim))
        elif activation == 2:
            layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                    nn.Sigmoid())
        else:
            if dropout:
                layer = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_p))
            else:
                layer = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU())
        
        return layer

    def encode(self, x):
        expr_level2_layer = self.e_fc1_expr(x)

        level_3_layer = self.e_fc2_expr(expr_level2_layer)

        # level_4_layer = self.e_fc3(level_3_layer)

        latent_mean = self.e_fc4_mean(level_3_layer)
        latent_log_var = self.e_fc4_log_var(level_3_layer)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)
        return mean + eps * sigma

    def decode(self, z):
        level_4_layer = self.d_fc4(z)

        # level_3_layer = self.d_fc3(level_4_layer)

        expr_level2_layer = self.d_fc2_expr(level_4_layer)

        recon_x = self.d_fc1_expr(expr_level2_layer)

        return recon_x

    def classifier(self, mean):
        level_1_layer = self.c_fc1(mean)
        level_2_layer = self.c_fc2(level_1_layer)
        # output_layer = self.c_fc3(level_2_layer)
        # return output_layer
        return level_2_layer

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        classifier_x = mean
        # if parallel:
        #     z = z.to('cuda:1')
        #     classifier_x = classifier_x.to('cuda:1')

        recon_x = self.decode(z)
        pred_y = self.classifier(classifier_x)
        return z, recon_x, mean, log_var, pred_y

class VAE_multi(nn.Module):
    def __init__(self, view_list,
                    input_dim, latent_space_dim, 
                    level_2_dim, level_3_dim,
                    level_4_dim, 
                    classifier_1_dim, class_num):
        super(VAE_multi, self).__init__()
        self.view_list = view_list
        self.input_dim = input_dim
        self.latent_space_dim = latent_space_dim
        self.level_2_dim = level_2_dim
        self.level_3_dim = level_3_dim
        self.level_4_dim = level_4_dim
        self.classifier_1_dim = classifier_1_dim
        self.class_num = class_num

        self.e_fc1_GE = self.fc_layer(input_dim[0], level_2_dim[0])
        self.e_fc1_CNA = self.fc_layer(input_dim[1], level_2_dim[1])
        if len(view_list) > 2: 
            self.e_fc1_mRNA = self.fc_layer(input_dim[2], level_2_dim[2])

        self.e_fc2_GE = self.fc_layer(level_2_dim[0], level_3_dim[0])
        self.e_fc2_CNA = self.fc_layer(level_2_dim[1], level_3_dim[1])
        if len(view_list) > 2: 
            self.e_fc2_mRNA = self.fc_layer(level_2_dim[2],level_3_dim[2])

        if len(view_list) > 2: 
            self.e_fc3 = self.fc_layer(level_3_dim[0] + level_3_dim[1] + level_3_dim[2], level_4_dim)
        else: 
            self.e_fc3 = self.fc_layer(level_3_dim[0] + level_3_dim[1], level_4_dim)

        self.e_fc4_mean = self.fc_layer(level_4_dim, latent_space_dim, activation=0)
        self.e_fc4_log_var = self.fc_layer(level_4_dim, latent_space_dim, activation=0)

        self.d_fc4 = self.fc_layer(latent_space_dim, level_4_dim)

        if len(view_list) > 2: 
            self.d_fc3 = self.fc_layer(level_4_dim, level_3_dim[0] + level_3_dim[1] + level_3_dim[2])
        else: 
            self.d_fc3 = self.fc_layer(level_4_dim, level_3_dim[0] + level_3_dim[1])

        self.d_fc2_GE = self.fc_layer(level_3_dim[0], level_2_dim[0])
        self.d_fc2_CNA = self.fc_layer(level_3_dim[1], level_2_dim[1])
        if len(view_list) > 2: 
            self.d_fc2_mRNA = self.fc_layer(level_3_dim[2], level_2_dim[2])

        self.d_fc1_GE = self.fc_layer(level_2_dim[0], input_dim[0])
        self.d_fc1_CNA = self.fc_layer(level_2_dim[1], input_dim[1])
        if len(view_list) > 2: 
            self.d_fc1_mRNA = self.fc_layer(level_2_dim[2], input_dim[2])
    
        self.c_fc1 = self.fc_layer(latent_space_dim, classifier_1_dim)
        self.c_fc2 = self.fc_layer(classifier_1_dim, class_num, activation=0)

    def fc_layer(self, in_dim, out_dim, activation=1, dropout=False, dropout_p=0.5):
        # print('DEBUG', type(in_dim))
        # print('DEBUG', type(out_dim))
        # print(out_dim)

        if activation == 0:
            layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim))
        elif activation == 2:
            layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.Sigmoid())
        else:
            if dropout:
                layer = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_p))
            else:
                layer = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU())
            return layer

    def encode(self, data_list):
        GE_level2_layer = self.e_fc1_GE(data_list[0])
        CNA_level2_layer = self.e_fc1_CNA(data_list[1])
        if (len(self.view_list) > 2):
            mRNA_level2_layer = self.e_fc1_mRNA(data_list[2])

        GE_level3_layer = self.e_fc2_GE(GE_level2_layer)
        CNA_level3_layer = self.e_fc2_GE(CNA_level2_layer)
        if (len(self.view_list) > 2):
            mRNA_level3_layer = self.e_fc2_GE(mRNA_level2_layer)

        if (len(self.view_list) > 2):
            level_3_layer = torch.cat((GE_level3_layer, CNA_level3_layer, mRNA_level3_layer), 1)
        else:
            level_3_layer = torch.cat((GE_level3_layer, CNA_level3_layer), 1)

        level_4_layer = self.e_fc3(level_3_layer)
        print(type(level_4_layer))
        print(level_4_layer)
        latent_mean = self.e_fc4_mean(level_4_layer)
        latent_log_var = self.e_fc4_log_var(level_4_layer)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)
        return mean + eps * sigma

    def decode(self, z):
        level_4_layer = self.d_fc4(z)

        level_3_layer = self.d_fc3(level_4_layer)

        GE_level3_layer = level_3_layer.narrow(1, 0, self.level_3_dim[0])
        CNA_level3_layer = level_3_layer.narrow(1, self.level_3_dim[0], self.level_3_dim[1])
        if (len(self.view_list) > 2):
            mRNA_level3_layer = level_3_layer.narrow(1, self.level_3_dim[1], self.level_3_dim[2])

        GE_level2_layer = self.d_fc2_GE(GE_level3_layer)
        CNA_level2_layer = self.d_fc2_CNA(CNA_level3_layer)
        if (len(self.view_list) > 2):
            mRNA_level2_layer = self.d_fc2_mRNA(mRNA_level3_layer)

        GE_level1_layer = self.d_fc1_GE(GE_level2_layer)
        CNA_level1_layer = self.d_fc1_CNA(CNA_level2_layer)
        if (len(self.view_list) > 2):
            mRNA_level1_layer = self.d_fc1_mRNA(mRNA_level2_layer)

        if (len(self.view_list) > 2):
            return GE_level1_layer, CNA_level1_layer, mRNA_level1_layer
    
        return GE_level1_layer, CNA_level1_layer

    def classifier(self, mean):
        level_1_layer = self.c_fc1(mean)
        level_2_layer = self.c_fc2(level_1_layer)
        # output_layer = self.c_fc3(level_2_layer)
        return level_2_layer

    def forward(self, data_list):
        print(type(data_list))
        mean, log_var = self.encode(data_list)
        z = self.reparameterize(mean, log_var)
        classifier_x = mean
        # if parallel:
        #     z = z.to('cuda:1')
        #     classifier_x = classifier_x.to('cuda:1')
        if (len(self.view_list) > 2):
            GE_x_hat, CNA_x_hat, mRNA_x_hat = self.decode(z)
            pred_y = self.classifier(classifier_x)
            return z, GE_x_hat, CNA_x_hat, mRNA_x_hat, mean, log_var, pred_y
        
        GE_x_hat, CNA_x_hat = self.decode(z)
        pred_y = self.classifier(classifier_x)
        return z, GE_x_hat, CNA_x_hat, mean, log_var, pred_y
