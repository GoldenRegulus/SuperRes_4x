from torch import nn, no_grad, ones
from torchvision.models import vgg19

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.example_input_array = ones((1, 3, 360, 320))
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_22 = nn.Sequential(*list(vgg19_model.features.children())[:8])
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[8:35])
        for param in self.vgg19_54.parameters():
            param.requires_grad=False
        for param in self.vgg19_22.parameters():
            param.requires_grad=False

    def forward(self, img):
        self.vgg19_54.eval()
        self.vgg19_22.eval()
        with no_grad():
            features22 = self.vgg19_22(img)
            features54 = self.vgg19_54(features22)
        return features54