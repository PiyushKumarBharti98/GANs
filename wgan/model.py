import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KERNEL_SIZE = 4
STRIDE = 2
PADDING = 1


class Generator(nn.Module):
    """docstring""" ""

    def __init__(
        self,
        in_features,
        out_features,
        img_channels,
        kernel_size=KERNEL_SIZE,
        stride=STRIDE,
        padding=PADDING,
    ) -> None:
        super().__init__()
        self.layer1 = self._generator_class(
            in_features, out_features * 16, kernel_size, stride, padding
        )  # (input=noise,output=features)
        self.layer2 = self._generator_class(
            out_features * 16, out_features * 8, kernel_size, stride, padding
        )
        self.layer3 = self._generator_class(
            out_features * 8, out_features * 4, kernel_size, stride, padding
        )
        self.layer4 = self._generator_class(
            out_features * 4, out_features * 2, kernel_size, stride, padding
        )
        self.layer5 = nn.ConvTranspose2d(
            out_features * 2, img_channels, kernel_size, stride, padding
        )  # outputs the final image channels
        self.generator = nn.Sequential(
            self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, nn.Tanh()
        )

    def _generator_class(
        self,
        in_features,
        out_features,
        kernel_size=KERNEL_SIZE,
        stride=STRIDE,
        padding=PADDING,
    ):
        """docstring"""
        return nn.Sequential(
            nn.Conv2d(
                in_features, out_features, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
        )

    def forward(self, x):
        """docstring"""
        return self.generator(x)


class Critic(nn.Module):
    """docstring"""

    def __init__(
        self,
        in_features,
        out_features,
        kernel_size=KERNEL_SIZE,
        stride=STRIDE,
        padding=PADDING,
    ) -> None:
        super().__init__()
        self.layer1 = nn.Conv2d(in_features, out_features, kernel_size, stride, padding)
        self.layer2 = nn.LeakyReLU()
        self.layer3 = self._critic_block(
            in_features, out_features * 2, kernel_size, stride, padding
        )  # input = image, out =a
        self.layer4 = self._critic_block(
            out_features * 2, out_features * 4, kernel_size, stride, padding
        )
        self.layer5 = self._critic_block(
            out_features * 4, out_features * 8, kernel_size, stride, padding
        )
        self.layer6 = nn.Conv2d(out_features * 8, 1, kernel_size, stride, padding)
        self.critic = nn.Sequential(
            self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6
        )

    def _critic_block(
        self,
        in_features,
        out_features,
        kernel_size=KERNEL_SIZE,
        stride=STRIDE,
        padding=PADDING,
    ):
        return nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size, stride, padding),
            nn.InstanceNorm2d(out_features),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        """docstring"""
        return self.critic(x)


def gradient_penalty(critic, real_samples, fake_samples, lambda_gp=10):
    """docstring"""
    batch_size = real_samples.size(0)
    eplison = torch.rand(batch_size, 1, 1, 1, device=device)
    eplison = eplison.expand_as(real_samples)

    interpolated = eplison * real_samples + fake_samples * (1 - eplison)
    interpolated.required_grad_(True)

    interpolated_scores = critic(interpolated)

    gradients = torch.autograd.grad(
        outputs=interpolated_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(interpolated_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradients_norm = gradients.norm(2, dim=1)

    final_penalty = lambda_gp * ((gradients_norm - 1) ** 2).mean()

    return final_penalty
