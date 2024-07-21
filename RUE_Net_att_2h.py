import torch
import torch.nn as nn
from torch.nn import functional as F
from RFE import UNet

class FGD_Module(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FGD_Module, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel//2, kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(in_channels=in_channel//2, out_channels=in_channel//2, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=.20)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.relu(self.drop(self.conv2(self.relu(self.drop(self.conv1(self.relu(self.drop(self.conv(x)))))))))
        return out
    
class RFE_Module(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFE_Module, self).__init__()
        self.unet = UNet(in_channel, out_channel)
    
    def forward(self, x):   
        x1, x2, x3, x4 = self.unet(x)
        return x1, x2, x3, x4
    
class Conv1Mul1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv1Mul1, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=.20)

    def forward(self, x):
        out = self.relu(self.drop(self.conv(x)))
        return out
    
class Conv_head(nn.Module):
    def __init__(self, in_channel):
        super(Conv_head, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel//2, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels=in_channel//2, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=.20)

    def forward(self, x):
        out = self.conv1(self.relu(self.drop(self.conv(x))))
        return out
    
class Conv_head2(nn.Module):
    def __init__(self, in_channel):
        super(Conv_head2, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel//2, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(in_channels=in_channel//2, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=.20)

    def forward(self, x):
        out = self.conv1(self.relu(self.drop(self.conv(x))))
        return out

    
       
class RUE_Net_att(nn.Module):
    def __init__(self):
        super(RUE_Net_att, self).__init__()
        self.input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.FGD1t2 = FGD_Module(64+3, 64)
        self.FGD2t3 = FGD_Module(64+3+3, 64)
        self.FGD3t4 = FGD_Module(64+3+8+3, 64)
        self.FGD4t5 = FGD_Module(64+3+8+8+3, 64)
        self.head = Conv_head(64+3+3*8)
        self.head2 = Conv_head2(64+3+3*8)
        self.downchannelx2 = Conv1Mul1(64, 8)
        self.downchannelx3 = Conv1Mul1(64, 8)
        self.downchannelx4 = Conv1Mul1(64, 8)
        self.cbam2 = CBAM(64)
        self.cbam3 = CBAM(64)
        self.cbam4 = CBAM(64)
        self.cbam5 = CBAM(64)
        self.RFE = RFE_Module(64,64)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, gray, gx, gy):
        x1 = self.relu(self.input(x))
        x2_RFE, x3_RFE, x4_RFE, x5_RFE = self.RFE(x1)
        x2_FGD = self.FGD1t2(torch.cat((x1, gray, gx, gy),dim = 1))
        x2 = x2_FGD + x2_RFE
        x2 = self.cbam2(x2)
        
        x2_downchannel = self.downchannelx2(x2)
        x2 = torch.cat((x2, x, gray, gx, gy), 1)
        x3_FGD = self.FGD2t3(x2)
        x3 = x3_FGD + x3_RFE
        x3 = self.cbam3(x3)
        
        x3_downchannel = self.downchannelx3(x3)
        x3 = torch.cat((x3, x2_downchannel, x, gray, gx, gy), 1)       
        x4_FGD = self.FGD3t4(x3)
        x4 = x4_FGD + x4_RFE
        x4 = self.cbam4(x4)
        
        x4_downchannel = self.downchannelx4(x4)
        x4 = torch.cat((x4, x3_downchannel, x2_downchannel, x, gray, gx, gy), 1)
        x5_FGD = self.FGD4t5(x4)
        x5 = x5_FGD + x5_RFE
        x5 = self.cbam5(x5)
        
        x5 = torch.cat((x5, x4_downchannel, x3_downchannel, x2_downchannel, x), 1)
        out = self.head(x5)
        out2 = self.head2(x5)

        return out, out2
    
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = self.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3)
        return x * out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return x * out


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        return out
