import torch
import torch.nn as nn
import torch.nn.functional as F


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