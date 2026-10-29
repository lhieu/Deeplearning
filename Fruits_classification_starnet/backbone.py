import torch
import torch.nn as nn

class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size = 1, stride = 1, padding = 0, groups = 1, with_bn = True):
        super().__init__()
        self.add_module('conv', nn.Conv2d(
            in_planes, out_planes, kernel_size = kernel_size, stride = stride, padding = padding, groups = groups, bias = False
        ))
        if with_bn:
            self.add_module('bn', nn.BatchNorm2d(out_planes))

class StarBlock(nn.Module):
    def __init__(self, dim, mlp_ratio = 2):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, kernel_size = 7, padding = 3, groups = dim, with_bn = True)

        mid_dim = int(mlp_ratio * dim)
        self.f1 = nn.Conv2d(dim, mid_dim, kernel_size = 1)
        self.f2 = nn.Conv2d(dim, mid_dim, kernel_size = 1)
        self.act = nn.ReLU6()

        self.g = ConvBN(mid_dim, dim, kernel_size = 1, with_bn = True)

    def forward(self, x):
        identity = x
        out = self.dwconv(x)
        out = self.g(self.f1(out) * self.act(self.f2(out)))
        return identity + out

class StarNetLite(nn.Module):
    def __init__(self, base_dim = 16, depths = [1, 1, 3,1], mlp_ratio = 2, num_classes = 36):
        super().__init__()
        self.num_classes = num_classes
        self.stem = nn.Sequential(
            ConvBN(3, base_dim, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU6(),
            ConvBN(base_dim, base_dim, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU6()
        )

        dims  = [base_dim * (2**i) for i in range (4)]

        stages = []
        cur_dim = base_dim

        for i in range (4):
            blocks = []

            if i > 0:
                blocks.append(ConvBN(cur_dim, dims[i], kernel_size = 3, stride = 2, padding = 1))
                cur_dim = dims[i]

            for _ in range (depths[i]):
                blocks.append(StarBlock(dim=cur_dim, mlp_ratio=mlp_ratio))
            
            stages.append(nn.Sequential(*blocks))

        self.stages = nn.ModuleList(stages)
        
        # Classifier Head tối giản
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.BatchNorm1d(dims[-1])
        self.fc = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.norm(x)
        x = self.fc(x)
        return x

def build_starnet_tiny(num_classes = 36):
    return StarNetLite(base_dim = 16, depths = [1,1,3,1], mlp_ratio = 2, num_classes = num_classes)    
