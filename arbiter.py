import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptGenerator(nn.Module):

    # https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/ebgan/ebgan.py 참조

    def __init__(self, nz=100, ngf=64, img_size=84, nc=3):
        super(PromptGenerator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            # nn.Sigmoid(),
            # nn.Tanh(),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Autoencoder(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim=10):
        super(Autoencoder, self).__init__()

        # Encoder: Reduce input to latent space of size latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

        # Decoder: Map latent space back to image dimensions (3, 84, 84)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            # nn.Sigmoid()  # Normalize to [0, 1] for image data
        )

    def forward(self, x):
        # Encoder
        x = self.encoder(x)

        # Decoder
        x = self.decoder(x)

        # Reshape output to (batch_size, 3, 84, 84)
        x = x.view(-1, 3, 84, 84)
        return x


class ConditionalAutoencoder(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim=10, condition_dim=10):
        super(ConditionalAutoencoder, self).__init__()

        self.latent_dim = latent_dim

        # Encoder: Reduce input + condition to latent space of size latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

        # Decoder: Map latent space + condition back to image dimensions (3, 84, 84)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            # nn.Sigmoid()  # Uncomment if the output should be normalized to [0, 1]
        )

    def forward(self, x, condition):
        # Ensure condition is the same shape as x in the batch dimension
        condition = condition.view(condition.size(0), -1)  # Flatten condition if needed

        # Concatenate input and condition for encoding
        x_cond = torch.cat([x, condition], dim=1)

        # Encoder
        z = self.encoder(x_cond)

        # Concatenate latent representation and condition for decoding
        z_cond = torch.cat([z, condition], dim=1)

        # Decoder
        x_reconstructed = self.decoder(z_cond)

        # Reshape output to (batch_size, 3, 84, 84)
        x_reconstructed = x_reconstructed.view(-1, 3, 84, 84)

        return x_reconstructed

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim=10):
        super(VariationalAutoencoder, self).__init__()

        # Encoder: 입력을 잠재 변수의 평균(mu)과 로그 분산(log_var)으로 변환
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # 잠재 변수의 평균(mu)과 로그 분산(log_var) 생성
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)

        # Decoder: 샘플링된 잠재 변수 z를 이미지로 복원
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Linear(512, output_dim),  # Flattened image output
            # nn.Sigmoid()  # Normalize to [0, 1] for image data
        )

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick: z = mu + std * epsilon
        """
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)  # Sample from standard normal distribution
        return mu + std * epsilon

    def forward(self, x):
        # Encode input to latent space (mu, log_var)
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        log_var = self.fc_log_var(encoded)

        # Reparameterization trick to sample z
        z = self.reparameterize(mu, log_var)

        # Decode z to reconstruct the image
        decoded = self.decoder(z)

        # Reshape to (batch_size, 3, 84, 84)
        decoded = decoded.view(-1, 3, 84, 84)
        return decoded, mu, log_var


# Conditional Variational Autoencoder class
class ConditionalVariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim=10, condition_dim=10):
        super(ConditionalVariationalAutoencoder, self).__init__()

        self.latent_dim = latent_dim

        # Encoder: Maps input + condition to mean and log-variance of latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)  # Mean of the latent space
        self.fc_logvar = nn.Linear(256, latent_dim)  # Log-variance of the latent space

        # Decoder: Maps latent space + condition back to output dimensions
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            # nn.Sigmoid()  # Output normalized to [0, 1]
        )

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var) using N(0, 1)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, condition):
        # Concatenate input and condition for encoding
        x_cond = torch.cat([x, condition], dim=1)

        # Encoder
        h = self.encoder(x_cond)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Reparameterize to get latent variable z
        z = self.reparameterize(mu, logvar)

        # Concatenate latent variable and condition for decoding
        z_cond = torch.cat([z, condition], dim=1)

        # Decoder
        x_reconstructed = self.decoder(z_cond)

        # Reshape output to (batch_size, 3, 84, 84)
        x_reconstructed = x_reconstructed.view(-1, 3, 84, 84)

        return x_reconstructed, mu, logvar