import torch.nn as nn




# for multi-band MelGAN
#--------------------------------------------------
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




# for styleMelGAN
#----------------------------------------------

#Temporal Adaptive DE-normalization (TADE)

class TADEResBlock(nn.Module):
    def __init__(self):
        pass
