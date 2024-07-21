import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1, num_heads=8):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_heads = num_heads

        self.query_conv = nn.Conv2d(in_channels, in_channels * num_heads, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.key_conv = nn.Conv2d(in_channels, in_channels * num_heads, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.value_conv = nn.Conv2d(in_channels, in_channels * num_heads, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        self.unify_heads = nn.Conv2d(in_channels * num_heads, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        queries = self.query_conv(x).view(batch_size, self.num_heads, channels, height, width).permute(0, 2, 1, 3, 4)
        keys = self.key_conv(x).view(batch_size, self.num_heads, channels, height, width).permute(0, 2, 1, 3, 4)
        values = self.value_conv(x).view(batch_size, self.num_heads, channels, height, width).permute(0, 2, 1, 3, 4)

        attention_scores = torch.einsum("bqchw,bkchw->bqhwk", queries, keys) / ((channels * self.num_heads) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_values = torch.einsum("bqhwk,bkchw->bqchw", attention_weights, values)
        output = F.relu(self.unify_heads(attended_values.view(batch_size, -1, height, width)))

        return output


class Conv1Mul1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv1Mul1, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=.20)

    def forward(self, x):
        out = self.relu(self.drop(self.conv(x)))
        return out

class decoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(decoder, self).__init__()

        self.up1 = nn.Sequential(
            Conv1Mul1(in_channels, mid_channels),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.Dropout2d(p=.20),
            nn.ReLU(inplace=True)
            )
        self.up2 = nn.Sequential(
            Conv1Mul1(mid_channels, out_channels),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout2d(p=.20),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, x_128, x_256):
        x = self.up1(x)
        x = x + x_128
        x = self.decoder1(x)
        
        x = self.up2(x)
        x = x + x_256
        x = self.decoder2(x)
        return x

class model(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=.20),
            nn.ReLU(inplace=True),
        )
       
        self.conv_128 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=.20),
            nn.ReLU(inplace=True),
        )

       
        self.decoder_256 = nn.Sequential(
            Conv1Mul1(mid_channels, out_channels),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=.20),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        output_128 = self.conv_128(x)
        output_256 = self.decoder_256(x)

        return output_128, output_256

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.drop = nn.Dropout2d(p=.20)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.drop(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.drop(out)
        out = self.relu(out)

        out = out + residual

        return out
    
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=3, padding=1),
            nn.Dropout2d(p=.20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.encoder2 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.Dropout2d(p=.20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.model = model(96, 96, out_channels)
        self.res1 = ResBlock(128, 128)
        self.res2 = ResBlock(128, 128)
        self.bottleneck = SelfAttention(128)  

        self.decoder1 = decoder(128, 96, out_channels)
        self.decoder2 = decoder(128, 96, out_channels)
        self.decoder3 = decoder(128, 96, out_channels)
        self.decoder4 = decoder(128, 96, out_channels)
        
    def forward(self, x):
        # Encoder
        x = self.encoder1(x)
        x_128, x_256 = self.model(x)
        x = self.encoder2(x)

        # Bottleneck
        x = self.res1(x)
        x = self.res2(x)
        x = self.bottleneck(x)

        # Decoder
        out1 = self.decoder1(x, x_128, x_256)
        out2 = self.decoder2(x, x_128, x_256)
        out3 = self.decoder3(x, x_128, x_256)
        out4 = self.decoder4(x, x_128, x_256)
        
        return out1, out2, out3, out4
    
    
if __name__=="__main__":
    in_channels = 64  
    out_channels = 64  
    Model = UNet(in_channels, out_channels)
    input_data = torch.randn(1, in_channels, 256, 256) 
    out1, out2, out3, out4 = Model(input_data)
    print(out1.shape, out2.shape, out3.shape, out4.shape)