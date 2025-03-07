import torch
import torch.nn as nn
import torch.nn.functional as F



class Genearator_bias(nn.Module):
    def __init__(self, args, nz=100, ngf=64, img_size=84, nc=3):
        super(Genearator_bias, self).__init__()

        self.multiplier_bias = nn.Parameter(torch.ones(1, nc, img_size, img_size))  # 1로 초기화
        self.offset_bias = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        out = self.multiplier_bias * x + self.offset_bias

        return out

class PromptGenerator(nn.Module):

    # https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/ebgan/ebgan.py 참조

    def __init__(self, args, nz=100, ngf=64, img_size=84, nc=3):
        super(PromptGenerator, self).__init__()

        self.args = args

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

        num_steps = self.args.number_of_training_steps_per_iter

        self.step_bais = nn.ModuleList()

        for i in range(num_steps):
          self.step_bais.append(Genearator_bias(args=self.args, nz=100, ngf=64, img_size=84, nc=3))


    def forward(self, z, num_step=0):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)

        # img = self.step_bais[num_step](img)

        # img = self.multiplier_bias * img + self.offset_bias
        # Tanh의 출력값을 miniImageNet 정규화 범위로 변환
        # min_val, max_val = -2.12, 2.64
        # img = img * (max_val - min_val) / 2 + (max_val + min_val) / 2

        return img


class TaskAwareAttention(nn.Module):
    def __init__(self, image_channels=3, task_dim=100, embed_dim=100):
        super(TaskAwareAttention, self).__init__()

        # 1x1 conv setting
        # kernel_size = 1
        # stride = 1
        # padding = 0

        # 3x3 conv setting
        kernel_size = 3
        stride = 1
        padding = 1

        # 7x7 conv setting
        # kernel_size = 7
        # stride = 1
        # padding = 3

        # Query: Task Embedding 변환
        # self.query_proj = nn.Linear(task_dim, embed_dim)

        # Key, Value 변환 (이미지 특징 추출)
        self.key_proj = nn.Conv2d(image_channels, embed_dim, kernel_size=kernel_size, padding=padding)
        self.value_proj = nn.Conv2d(image_channels, image_channels, kernel_size=kernel_size, padding=padding)

        self.softmax = nn.Softmax(dim=-1)

        # Bias 추가
        self.multiplier_bias = nn.Parameter(torch.ones(1, image_channels, 84, 84))  # 채널별 조정
        self.offset_bias = nn.Parameter(torch.zeros(1))  # 전체적인 조정

    def forward(self, image, task_embedding):
        """
        image: (B, 3, 84, 84) - Key, Value 역할
        task_embedding: (B, 100) - Query 역할
        """
        batch_size, _, height, width = image.shape

        # Key, Value 변환
        key = self.key_proj(image).view(batch_size, -1, height * width)  # (B, embed_dim, H*W)
        value = self.value_proj(image).view(batch_size, -1, height * width)  # (B, 3, H*W)

        # Query 변환
        # query = self.query_proj(task_embedding).unsqueeze(1)  # (B, 1, embed_dim)

        # print("key == ", key.shape)
        # print("query == ", task_embedding.shape)
        # print("query == ", query.shape)
        # print("value == ", value.shape)

        # Attention Score 계산 (Query @ Key^T) / sqrt(embed_dim)
        scores = torch.matmul(task_embedding, key) / (key.shape[1] ** 0.5)  # (B, 1, H*W) query_layer를 사용하지 않을때
        # scores = torch.matmul(query, key) / (key.shape[1] ** 0.5)  # (B, 1, H*W)
        attention_weights = self.softmax(scores)  # (B, 1, H*W)

        prompt = (value * attention_weights).view(batch_size, -1, height, width)  # (B, 3, 84, 84)

        # Bias 적용
        prompt = self.multiplier_bias * prompt + self.offset_bias


        return prompt # , attention_weights  # (B, 3, 84, 84), (B, 1, H*W)

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