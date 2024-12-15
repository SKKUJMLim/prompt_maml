import torch
import torch.nn as nn
import torch.nn.functional as F

class PromptGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3):
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
            nn.Sigmoid(),
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
            nn.Sigmoid()  # Normalize to [0, 1] for image data
        )

    def forward(self, x):
        # Encoder
        x = self.encoder(x)

        # Decoder
        x = self.decoder(x)

        # Reshape output to (batch_size, 3, 84, 84)
        x = x.view(-1, 3, 84, 84)
        return x

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
            nn.Sigmoid()  # Normalize to [0, 1] for image data
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

# Convolutional AutoEncoder 모델 정의
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder: Reduce the input size from [64, 5, 5] to a latent representation
        self.encoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # [128, 5, 5]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # [256, 5, 5]
            nn.ReLU(),
            nn.Flatten(),  # Flatten to [256 * 5 * 5]
            nn.Linear(256 * 5 * 5, 1024),  # Latent representation
            nn.ReLU()
        )

        # Decoder: Expand the latent representation to [3, 84, 84]
        self.decoder = nn.Sequential(
            nn.Linear(1024, 256 * 21 * 21),
            nn.ReLU(),
            nn.Unflatten(1, (256, 21, 21)),  # Reshape to [256, 21, 21]
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # [128, 42, 42]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # [64, 84, 84]
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),  # [3, 84, 84]
            # nn.Tanh()  # Output values in range [0, 1]
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


class VariationalConvAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalConvAutoencoder, self).__init__()
        # Encoder: Reduce the input size from [64, 5, 5] to a latent representation
        self.encoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # [128, 5, 5]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # [256, 5, 5]
            nn.ReLU(),
            nn.Flatten(),  # Flatten to [256 * 5 * 5]
        )

        self.fc_mu = nn.Linear(256 * 5 * 5, 1024)  # Mean of latent representation
        self.fc_logvar = nn.Linear(256 * 5 * 5, 1024)  # Log variance of latent representation

        # Decoder: Expand the latent representation to [3, 84, 84]
        self.decoder = nn.Sequential(
            nn.Linear(1024, 256 * 21 * 21),
            nn.ReLU(),
            nn.Unflatten(1, (256, 21, 21)),  # Reshape to [256, 21, 21]
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # [128, 42, 42]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # [64, 84, 84]
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),  # [3, 84, 84]
            # nn.Sigmoid()  # Output values in range [0, 1]
        )

    def reparameterize(self, mu, logvar):
        # Reparameterization trick: z = mu + std * epsilon
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)

        # Decode
        decoded = self.decoder(z)
        return decoded, mu, logvar


class ConditionalUNetVAE(nn.Module):

    '''
    Image를 Input으로 하고, Gradient vector를 조건정보로 활용하기 위한 함수
    '''

    def __init__(self, input_channels=3, conditional_dim=10, output_channels=3, latent_dim=128):
        super(ConditionalUNetVAE, self).__init__()

        self.conditional_dim = conditional_dim

        # Encoder
        self.enc_conv1 = nn.Conv2d(input_channels + conditional_dim, 64, kernel_size=3, stride=1, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.fc_mu = nn.Linear(256 * 21 * 21, latent_dim)
        self.fc_logvar = nn.Linear(256 * 21 * 21, latent_dim)

        # Decoder
        self.fc_latent = nn.Linear(latent_dim + conditional_dim, 256 * 21 * 21)
        self.dec_conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.dec_conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.dec_conv3 = nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1)

    def encode(self, x, c):
        # Combine input image and conditional input
        c = c.view(c.size(0), self.conditional_dim, 1, 1).expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        x1 = F.relu(self.enc_conv1(x))
        x2 = F.relu(self.enc_conv2(x1))
        x3 = F.relu(self.enc_conv3(x2))

        x3_flat = x3.view(x3.size(0), -1)
        mu = self.fc_mu(x3_flat)
        logvar = self.fc_logvar(x3_flat)
        return mu, logvar, x3

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        # Combine latent variable and conditional input
        z = torch.cat([z, c], dim=1)
        x = self.fc_latent(z)
        x = x.view(x.size(0), 256, 21, 21)

        x = F.relu(self.dec_conv1(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.dec_conv2(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.dec_conv3(x)
        x = torch.sigmoid(x)

        return x

    def forward(self, x, c):
        mu, logvar, _ = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar