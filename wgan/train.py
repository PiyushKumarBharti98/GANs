import torch
from torch import optim
import torchvision
from torchvision import transforms
from torchvision import datasets

from torch.utils.data import DataLoader

# from torch.utils.tensorboard import SummaryWriter
# import matplotlib.pyplot as plt
# import numpy as np
from model import Generator, Critic
from model import gradient_penalty

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training parameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NUM_EPOCHS = 100
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10
FEATURES_CRITIC = 64
NUM_EPOCHS = 100
LAMBDA_GP = 10
KERNEL_SIZE = 4
STRIDE = 2
PADDING = 1
FEATURES_GEN = 64
Z_DIM = 512


generator = Generator(
    Z_DIM, CHANNELS_IMG, FEATURES_GEN, KERNEL_SIZE, STRIDE, PADDING
).to(device)
critic = Critic(CHANNELS_IMG, FEATURES_CRITIC, KERNEL_SIZE, STRIDE, PADDING).to(device)

opt_gen = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

dataset = datasets.ImageNet(root="/train", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

fixed_noise = torch.randn((32, Z_DIM, 1, 1), device=device)

for epochs in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)

        noise = torch.rand((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
        with torch.no_grad():
            fake = generator(noise)
        critic_real = critic(real).to(device)
        critic_fake = critic(fake).to(device)
        gp = gradient_penalty(critic, real, fake).to(device)
        loss_critic = (
            -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
        )

        critic.zero_grad()
        loss_critic.backward(retain_graph=True)
        opt_critic.step()

        noise = torch.rand((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
        fake = generator(noise)
        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)

        generator.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epochs}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} "
                f"Loss D: {loss_critic:.4f}, Loss G: {loss_gen:.4f}"
            )

    # Save generated images
    with torch.no_grad():
        fake = generator(fixed_noise)
        torchvision.utils.save_image(fake, f"generated_{epochs}.png", normalize=True)
