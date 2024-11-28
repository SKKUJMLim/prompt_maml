import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=10):
        super(Autoencoder, self).__init__()

        # Encoder: Reduce input to latent space of size latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )

        # Decoder: Map latent space back to image dimensions (3, 84, 84)
        self.decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3 * 84 * 84),  # Output flattened image
            nn.Sigmoid()  # Normalize output to [0, 1] for image data
        )

    def forward(self, x):
        # Encoder
        x = self.encoder(x)

        # Decoder
        x = self.decoder(x)

        # Reshape output to (batch_size, 3, 84, 84)
        x = x.view(-1, 3, 84, 84)
        return x

# Convolutional AutoEncoder 모델 정의
class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 84x84 -> 42x42
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 42x42 -> 21x21
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 21x21 -> 11x11
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 11 * 11, 256),  # 잠재 공간 (256차원)
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 128 * 11 * 11),
            nn.ReLU(),
            nn.Unflatten(1, (128, 11, 11)),  # 256 -> 11x11 크기 복원
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),  # 11x11 -> 21x21
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 21x21 -> 42x42
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 42x42 -> 84x84
            nn.Sigmoid()  # 출력 범위를 [0, 1]로 제한
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim=10):
        super(VariationalAutoencoder, self).__init__()

        # Encoder: 입력을 잠재 변수의 평균(mu)과 로그 분산(log_var)으로 변환
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )

        # 잠재 변수의 평균(mu)과 로그 분산(log_var) 생성
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_log_var = nn.Linear(512, latent_dim)

        # Decoder: 샘플링된 잠재 변수 z를 이미지로 복원
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3 * 84 * 84),  # Flattened image output
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