import torch.nn as nn


##
# MB-Melgan follows the general structure of basic MelGAN
# Generator :   x200 through 3 upsampling layer with different three channels
#               kernel-size is twice of the stride
#               ResStack has 4 layers with 1,3,9 and 27 with kernel size 3

class MultiDownSampler(nn.Module):
    def __init__(self, factor=2, num_scaling=3):
        super().__init__()
        self.num_scaling=num_scaling

        kernel_size = 4 # as written in paper
        self.scaler = nn.AvgPool1d(kernel_size, 2, 1, count_include_pad=False)

    def forward(self, input):
        scales = [input]
        x = input
        for _ in range(self.num_scaling - 1):
            x = self.scaler(x)
            scales.append(x)
        return scales
            


class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, factor):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor 
        # kernel_size being twice of the stride (from paper)
        self.net = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=self.factor*2, stride=self.factor, padding=self.factor//2)

    def forward(self, x):
        # target_length
        t = x.size(-1)*self.factor
        out = self.net(x)[..., :t]
        return out
    
class ResidualStack(nn.Module):
    def __init__(self, num_channels, leaky=0.2):
        super().__init__()
        # repeat 3 times
        # i=0,1,2
        # input
        # |
        # lReLU 3 x 1, dilation=3^i conv
        # lReLU 3 x 1, dilation=1 conv
        # | + input
        blocks = [self._create_residual_block(num_channels, 3**i) for i in range(4)]
        self.blocks = nn.ModuleList(blocks)
    
    def _create_residual_block(self, num_channels, dilation):
        return nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.Conv1d(num_channels, num_channels, kernel_size=3, dilation=dilation, padding=dilation, padding_mode='reflect')),
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=1)),
        )

    def forward(self, input):
        x = input
        for block in self.blocks:
            x = block(x) + x 
        return x



##
# Generator
class BaseGenerator(nn.Module):
    def __init__(self, n_mels, channels, upsample_factors, leak=0.2):
        """
        Args:
        in_channels (int) : n_mels
        """
        super().__init__()
        self.n_mels = n_mels
        self.channels = channels
        self.upsample_factors = upsample_factors

        
        self.prev_conv = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(n_mels, channels[0], kernel_size=7, padding=3, padding_mode='reflect')),
            nn.LeakyReLU(leak),
        )
            
        self.up_resnet1= nn.Sequential(
            UpSampling(channels[0], channels[1], upsample_factors[0]),
            ResidualStack(channels[1]),
            nn.LeakyReLU(leak),
        )

        self.up_resnet2= nn.Sequential(
            UpSampling(channels[1], channels[2], upsample_factors[1]),
            ResidualStack(channels[2]),
            nn.LeakyReLU(leak),
        )

        self.up_resnet3= nn.Sequential(
            UpSampling(channels[2], channels[3], upsample_factors[2]),
            ResidualStack(channels[3]),
            nn.LeakyReLU(leak),
        )

        self.post_conv = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(channels[3], channels[4], kernel_size=7, padding=3, padding_mode='reflect')),
            nn.Tanh(),
        )

    def _f(self, input):
        x = self.prev_conv(input)
        x = self.up_resnet1(x)
        x = self.up_resnet2(x)
        x = self.up_resnet3(x)
        x = self.post_conv(x)
        return x

    def forward(self, input):
        # input : [B, n_mels, L]
        # out : [B, 4, 20*L]
        out = self._f(input)
        return out


class MultiBandGenerator(BaseGenerator):
    def __init__(self, n_mels, leak=0.2):
        super().__init__(n_mels, [384, 192, 96, 48, 4], [2, 5, 5], leak)


class FullBandGenerator(BaseGenerator):
    def __init__(self, n_mels, leak=0.2):
        super().__init__(n_mels, [512, 256, 128, 64, 1], [5, 5, 8], leak)
        


class DiscriminatorBlock(nn.Module):
    def __init__(self, leak=0.2):
        super().__init__()
        #
        self.net = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(1, 16, kernel_size=15, padding=7)),
            nn.LeakyReLU(leak),
            nn.utils.weight_norm(nn.Conv1d(16, 64, kernel_size=41, stride=4, padding=20, groups=4)),
            nn.LeakyReLU(leak),
            nn.utils.weight_norm(nn.Conv1d(64, 256, kernel_size=41, stride=4, padding=20, groups=16)),
            nn.LeakyReLU(leak),
            nn.utils.weight_norm(nn.Conv1d(256, 512, kernel_size=41, stride=4, padding=20, groups=64)),
            nn.LeakyReLU(leak),
            nn.utils.weight_norm(nn.Conv1d(512, 512, kernel_size=5, padding=2)),
            nn.LeakyReLU(leak),
            nn.utils.weight_norm(nn.Conv1d(512, 1, kernel_size=3, padding=1)),
        )
        
    def forward(self, input):
        return self.net(input)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        num = 3
        self.discriminators = nn.ModuleList(
            [DiscriminatorBlock() for _ in range(num)]
        )
        self.downsampler = MultiDownSampler(num_scaling=num)

    def forward(self, input):
        x = self.downsampler(input)
        out = []
        for x, discirminator in zip(x, self.discriminators):
            out.append(discirminator(x))
        return out

