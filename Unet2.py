import torch
from torch import nn
import torch.nn.functional as F

#Reference: https://github.com/milesial/Pytorch-UNet

class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = self.double_conv(in_channels, 16)
        self.conv2 = self.down_sample(16, 32)
        self.conv3 = self.down_sample(32, 64)

        self.upconv2 = self.up_sample(64, 32)
        self.upconv1 = self.up_sample(32, 16)
        self.output = self.out(16, out_channels)

    def __call__(self, x, layer_name=None):
        # if you want to obtain output of certain layer, just pass the desired output layer in the layer_name 
        # downsampling part
        x = x.float()
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        #upsampling part
        concat2 = self.concat(self.upconv2,conv3,conv2)
        up2 = self.upconv2[1:](concat2)
        concat1 = self.concat(self.upconv1,up2,conv1)
        up1 = self.upconv1[1:](concat1)
        output = self.output(up1)
        
        layers = {
            'conv1':conv1,
            'conv2':conv2,
            'conv3':conv3,
            'concat2':concat2,
            'up2':up2,
            'concat1':concat1,
            'up1':up1,
            'output':output,
        }
        
        if layer_name:
            return layers[layer_name]
        else:
            return output
    

    def double_conv(self, in_channels, out_channels, mid_channels=None, kernel_size=3, padding=1):

        if not mid_channels:
            mid_channels = out_channels
            
        double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        return double_conv
    
    
    def down_sample(self, in_channels, out_channels, mid_channels=None, kernel_size=3, padding=1):        
        down = nn.Sequential(
            nn.MaxPool2d(2),
            self.double_conv(in_channels, out_channels, mid_channels=mid_channels, kernel_size=kernel_size, padding=padding)
        )
        
        return down
    

    def up_sample(self, in_channels, out_channels, mid_channels=None, kernel_size=3, padding=1):
        up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        conv = self.double_conv(in_channels, out_channels, mid_channels=mid_channels, kernel_size=kernel_size, padding=padding)
        return nn.Sequential(up,conv)
    
    def concat(self, up_sample,x1,x2):
     
        up, conv = up_sample[0], up_sample[1:]
        x1 = up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        
        return x
    
    def out(self, in_channels, out_channels):
        out = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        return out