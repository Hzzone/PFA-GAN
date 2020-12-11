from torch import nn
import torch
import torch.nn.functional as F
from ops import get_norm_layer, group2feature, pfa_encoding


class Generator(nn.Module):

    def __init__(self, age_group, norm_layer='bn'):
        super(Generator, self).__init__()
        self.gs = nn.ModuleList()
        for _ in range(age_group - 1):
            self.gs.append(SubGenerator(norm_layer=norm_layer))
        self.age_group = age_group

    def forward(self, x, source_label: torch.Tensor, target_label: torch.Tensor):
        condition = pfa_encoding(source_label, target_label, self.age_group).to(x).float()
        for i in range(self.age_group - 1):
            aging_effects = self.gs[i](x)
            x = x + aging_effects * condition[:, i]
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels, norm_layer):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            get_norm_layer(norm_layer, nn.Conv2d(channels, channels, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            get_norm_layer(norm_layer, nn.Conv2d(channels, channels, 3, 1, 1)),
        )

    def forward(self, x):
        residual = x
        x = self.main(x)
        return F.leaky_relu(residual + x, 0.2, inplace=True)


class SubGenerator(nn.Module):

    def __init__(self, in_channels=3, repeat_num=4, norm_layer='bn'):
        super(SubGenerator, self).__init__()
        layers = [
            get_norm_layer(norm_layer, nn.Conv2d(in_channels, 32, 9, 1, 4)),
            nn.LeakyReLU(0.2, inplace=True),
            get_norm_layer(norm_layer, nn.Conv2d(32, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            get_norm_layer(norm_layer, nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for _ in range(repeat_num):
            layers.append(ResidualBlock(128, norm_layer))
        layers.extend([
            get_norm_layer(norm_layer, nn.ConvTranspose2d(128, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            get_norm_layer(norm_layer, nn.ConvTranspose2d(64, 32, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, 9, 1, 4),
        ])
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class PatchDiscriminator(nn.Module):

    def __init__(self, age_group, conv_dim=64, repeat_num=3, norm_layer='bn'):
        super(PatchDiscriminator, self).__init__()

        use_bias = True
        self.age_group = age_group

        self.conv1 = nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)
        sequence = []
        nf_mult = 1

        for n in range(1, repeat_num):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                get_norm_layer(norm_layer, nn.Conv2d(conv_dim * nf_mult_prev + (self.age_group if n == 1 else 0),
                                                     conv_dim * nf_mult, kernel_size=4, stride=2, padding=1,
                                                     bias=use_bias)),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** repeat_num, 8)

        sequence += [
            get_norm_layer(norm_layer,
                           nn.Conv2d(conv_dim * nf_mult_prev, conv_dim * nf_mult, kernel_size=4,
                                     stride=1, padding=1,
                                     bias=use_bias)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(conv_dim * nf_mult, 1, kernel_size=4, stride=1,
                      padding=1)  # output 1 channel prediction map
        ]
        self.main = nn.Sequential(*sequence)

    def forward(self, inputs, condition):
        x = F.leaky_relu(self.conv1(inputs), 0.2, inplace=True)
        condition = group2feature(condition, feature_size=x.size(2), age_group=self.age_group).to(x)
        return self.main(torch.cat([x, condition], dim=1))


class AuxiliaryAgeClassifier(nn.Module):

    def __init__(self, age_group, conv_dim=64, repeat_num=3):
        super(AuxiliaryAgeClassifier, self).__init__()
        age_classifier = [
            nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(True),
        ]
        nf_mult = 1
        for n in range(1, repeat_num + 2):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            age_classifier += [
                nn.Conv2d(conv_dim * nf_mult_prev,
                          conv_dim * nf_mult, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(conv_dim * nf_mult),
                nn.ReLU(True),
            ]
        age_classifier += [
            nn.Flatten(),
            nn.Linear(conv_dim * nf_mult * 16, 101),
        ]
        self.age_classifier = nn.Sequential(*age_classifier)
        self.group_classifier = nn.Linear(101, age_group)

    def forward(self, inputs):
        age_logit = self.age_classifier(F.hardtanh(inputs))
        group_logit = self.group_classifier(age_logit)
        return age_logit, group_logit
