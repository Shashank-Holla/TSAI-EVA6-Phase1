import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Convolution layer-1  
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),  
            #Output size- 32. RF=3
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16)  
            #Output size- 32. RF=5                    
        ) 

        # Dilation convolution - 1
        self.dilationblock1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), dilation=2, padding=2, stride=2, bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(16)
        ) #Output size- 16. RF=9

        # Convolution layer-2  
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),  
            #Output size- 16. RF=13    
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), dilation=2, padding=2, bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(32),
            #Output size- 16. RF=21
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, groups=32, bias=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64), 
            #Output size- 16. RF=25             
        ) 

        # Dilation convolution - 2
        self.dilationblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), dilation=2, padding=2, stride=2, bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(32)
        ) #Output size- 8. RF=33

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),  
            #Output size- 8. RF=39
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, groups=64, bias=False),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128), 
            #Output size- 8. RF=45                 
        ) 

        # Dilation convolution - 3
        self.dilationblock3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), dilation=2, padding=2, stride=2, bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(32)
        ) #Output size- 4. RF=57

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),  
            #Output size- 4. RF=65
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10) 
            #Output size- 4. RF=73                  
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        # nn.AvgPool2d(kernel_size=4)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.dilationblock1(x)
        x = self.convblock2(x)
        x = self.dilationblock2(x)
        x = self.convblock3(x)
        x = self.dilationblock3(x)
        x = self.convblock4(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return x