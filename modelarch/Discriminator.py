from torch import nn
from fairscale.nn import checkpoint_wrapper

class DiscriminatorBlock(nn.Module):
    def __init__(self, inp_ch, out_ch, first_block=False):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(inp_ch, out_ch, kernel_size=3, stride=1, padding=1))
        if not first_block:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)
    
    def forward(self, inp):
        return self.model(inp)

class Discriminator(nn.Module):
    def __init__(self, chn):
        super().__init__() 
        layers = []
        layers.append(checkpoint_wrapper(DiscriminatorBlock(3, chn, True)))
        layers.append(checkpoint_wrapper(DiscriminatorBlock(chn, chn*2)))
        layers.append(checkpoint_wrapper(DiscriminatorBlock(chn*2, chn*4)))
        layers.append(checkpoint_wrapper(DiscriminatorBlock(chn*4, chn*8)))
        layers.append(nn.Conv2d(chn*8, 1, kernel_size=3, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, inp):
        return self.model(inp)