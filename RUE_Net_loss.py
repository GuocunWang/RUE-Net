from VGG_loss import *
from ssim_loss import *
from torchvision import models

class RUENet_loss(nn.Module):
    def __init__(self):
        super(RUENet_loss, self).__init__()
        vgg = models.vgg19_bn(pretrained=True)
        print("VGG model is loaded")
        self.vggloss = VGG_loss(vgg)
        self.ssimloss = ssimLoss()
        for param in self.vggloss.parameters():
            param.requires_grad = False
        self.mseloss = nn.MSELoss().to("cuda")
        self.l1loss = nn.L1Loss().to("cuda")

    def forward(self, out, label):
        inp_vgg = self.vggloss(out)
        label_vgg = self.vggloss(label)
        mse_loss = self.mseloss(out, label)
        vgg_loss = self.l1loss(inp_vgg, label_vgg)
        ssim_loss = self.ssimloss(out, label)
        total_loss = mse_loss*5 + vgg_loss + ssim_loss
        return total_loss, mse_loss, vgg_loss, ssim_loss
