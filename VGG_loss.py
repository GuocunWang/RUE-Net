import torch.nn as nn

class VGG_loss(nn.Module):
    def __init__(self, model):
        super(VGG_loss, self).__init__()
        self.features = nn.Sequential(*list(model.children())[0][:-3]).to("cuda")
    def forward(self, x):
        return self.features(x)