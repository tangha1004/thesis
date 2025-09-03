import torch
import torch.nn as nn
import torch.nn.functional as F

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)

def recon_loss(recon_x, x):
    loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    return loss

def monte_carlo_recon_loss(mean, log_var, x, decoder, n_samples=5, dec_var=0.5):
    batch_size = x.size(0)
    latent_dim = mean.size(1)
    input_dim = x.size(1)

    std = torch.exp(0.5 * log_var)
    eps = torch.randn(n_samples, batch_size, latent_dim, device=mean.device)

    z = eps * std.unsqueeze(0) + mean.unsqueeze(0)
    z_decode = z.view(-1, latent_dim)
    x_hat = decoder(z_decode)
    x_hat = x_hat.view(n_samples, batch_size, input_dim)
    x_expanded = x.unsqueeze(0)
    se = 0.5 * (1/dec_var) * (x_hat - x_expanded) ** 2
    se = se.sum(dim=2)
    return se.sum(dim=0)

def kl_loss(mean, log_var):
    loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return loss

def classifier_loss(pred_y, y):
    loss = F.cross_entropy(pred_y, y, reduction='sum')
    return loss

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim_expr, hidden_dims, latent_dim=128):
        super(VariationalAutoencoder, self).__init__()
        encoder_layers = [
            nn.Dropout(0.5),
            nn.Linear(input_dim_expr, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ELU()
        ]
        for i in range(1, len(hidden_dims)):
            encoder_layers.extend([
                nn.Linear(hidden_dims[i-1], hidden_dims[i]),
                nn.ELU()
            ])
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mean = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_log_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        decoder_layers = []
        for i in range(len(hidden_dims)-1, 0, -1):
            decoder_layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i-1]),
                nn.ELU()
            ])
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim_expr))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        mean = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        return mean, log_var
    
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var
    
    def get_loss(self, x):
        mean, log_var = self.encode(x)

        recon_loss_val = monte_carlo_recon_loss(
            mean, log_var, x, self.decode, 
            n_samples=self.n_samples, dec_var=self.dec_var
        )
        kl_loss_val = kl_loss(mean, log_var)
        total_loss = recon_loss_val + kl_loss_val
        return total_loss.mean()

def create_autoencoder(omic, input_dim_expr, n_samples=5, dec_var=0.5):
    if omic == 'meth':
        return VariationalAutoencoder(input_dim_expr, [1024, 512], n_samples=n_samples, dec_var=dec_var)
    elif omic == 'ge':
        return VariationalAutoencoder(input_dim_expr, [1024, 512], n_samples=n_samples, dec_var=dec_var)
    elif omic == 'cna':
        return VariationalAutoencoder(input_dim_expr, [1024, 256], n_samples=n_samples, dec_var=dec_var)
    else:
        return VariationalAutoencoder(input_dim_expr, [1024, 512], n_samples=n_samples, dec_var=dec_var)

class Subtyping_model(nn.Module):
    def __init__(self, vae_models, subtypes):
        super(Subtyping_model, self).__init__()
        self.vae_models = nn.ModuleList(vae_models)
        
        total_latent_dim = sum([model.latent_dim for model in vae_models])
        
        self.classifier = nn.Sequential(
            nn.Linear(total_latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(512, subtypes)
        )
        
    def forward(self, *inputs):
        latent_vectors = []
        
        for i, model in enumerate(self.vae_models):
            mean, log_var = model.encode(inputs[i])
            z = model.reparameterize(mean, log_var)
            latent_vectors.append(z)

        concatenated_latent = torch.hstack(latent_vectors)
        
        return self.classifier(concatenated_latent)
    
    def get_latent_representations(self, *inputs):
        latent_vectors = []
        means = []
        log_vars = []
        
        for i, model in enumerate(self.vae_models):
            mean, log_var = model.encode(inputs[i])
            z = model.reparameterize(mean, log_var)
            
            latent_vectors.append(z)
            means.append(mean)
            log_vars.append(log_var)
            
        concatenated_latent = torch.hstack(latent_vectors)
        
        return concatenated_latent, latent_vectors, means, log_vars

def create_autoencoder(omic, input_dim_expr):
    if omic == 'meth':
        return VariationalAutoencoder(input_dim_expr, [1024, 512])
    elif omic == 'ge':
        return VariationalAutoencoder(input_dim_expr, [1024, 512])
    elif omic == 'cna':
        return VariationalAutoencoder(input_dim_expr, [1024, 256])
    else:
        return VariationalAutoencoder(input_dim_expr, [1024, 512])