from torch import nn, sigmoid, cat
from torch.nn.functional import interpolate

class SCPA(nn.Module):
    def __init__(self, chn=64):
        super().__init__()
        self.xd1 = nn.Conv2d(chn,chn//2,1,1,0)
        self.xdd1 = nn.Conv2d(chn,chn//2,1,1,0)
        self.lrelu = nn.LeakyReLU(0.2)
        self.xd31 = nn.Conv2d(chn//2,chn//2,3,1,1)
        self.xd1p = nn.Conv2d(chn//2,chn//2,1,1,0)
        self.xd32 = nn.Conv2d(chn//2,chn//2,3,1,1)
        self.xdd3 = nn.Conv2d(chn//2,chn//2,3,1,1)
        self.x1 = nn.Conv2d(chn,chn,1,1,0)
    
    def forward(self,inp):
        x1 = self.xd1(inp)
        x1 = self.lrelu(x1)
        x1p = x1
        x1 = self.xd31(x1)
        x1p = self.xd1p(x1p)
        x1p = sigmoid(x1p)
        x1 = x1*x1p
        x1 = self.xd32(x1)
        x1 = self.lrelu(x1)
        x2 = self.xdd1(inp)
        x2 = self.lrelu(x2)
        x2 = self.xdd3(x2)
        x2 = self.lrelu(x2)
        x = cat([x1,x2],dim=1)
        x = self.x1(x)
        x = x+inp
        return x

class UPA(nn.Module):
    def __init__(self,chn,hchn=False):
        super().__init__()
        if not hchn:
            hchn = chn
        self.xc1 = nn.Conv2d(chn,hchn,3,1,1)
        self.xpa = nn.Conv2d(hchn,hchn,1,1,0)
        self.xc2 = nn.Conv2d(hchn,hchn,3,1,1)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, inp):
        x = interpolate(inp,scale_factor=2,mode='nearest')
        x = self.xc1(x)
        xp = self.xpa(x)
        xp = sigmoid(xp)
        x = xp*x
        x = self.lrelu(x)
        x = self.xc2(x)
        x = self.lrelu(x)
        return x

class PAN(nn.Module):
    def __init__(self, scpa=16, chn=40, hchn=24):
        super().__init__()
        self.firstconv = nn.Conv2d(3,chn,3,1,1)
        self.scpablocks = nn.ModuleList([SCPA(chn) for _ in range(scpa)])
        self.upa1 = UPA(chn,hchn)
        self.upa2 = UPA(hchn)
        self.lastconv = nn.Conv2d(hchn,3,3,1,1)
    
    def forward(self, inp):
        x = self.firstconv(inp)
        for block in self.scpablocks:
            x = block(x)
        x = self.upa1(x)
        x = self.upa2(x)
        x = self.lastconv(x)
        return x