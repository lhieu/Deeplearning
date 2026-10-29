import torch
import torch.nn as nn

class ConvBlock (nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, bias  = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )
    def forward(self, x):
        return self.block(x)

class CustomFruitNet(nn.Module):
    def __init__(self, num_classes = 36, dropout_rate = 0.3):
        super().__init__()
        self.stage1 = nn.Sequential(
            ConvBlock(3, 32),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.stage2 = nn.Sequential(
                    ConvBlock(32, 64),
                    nn.MaxPool2d(kernel_size = 2, stride = 2)
                )
        self.stage3 = nn.Sequential(
                    ConvBlock(64, 128),
                    nn.MaxPool2d(kernel_size = 2, stride = 2)
                )
        self.stage4 = nn.Sequential(
                    ConvBlock(128, 256),
                    nn.MaxPool2d(kernel_size = 2, stride = 2)
                )
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p = dropout_rate)
        self.classifier = nn.Linear(256, num_classes)
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout (x)
        x = self.classifier(x)
        return x
    def build_custom_net(num_classes = 36):
        return CustomFruitNet(num_classes = num_classes)