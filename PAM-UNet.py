#author 吴思睿
from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np
from thop import profile
from thop import clever_format
import math


class DropBlock(nn.Module):
    def __init__(self, block_size: int = 5, p: float = 0.1):
        super().__init__()
        self.block_size = block_size
        self.p = p

    def calculate_gamma(self, x: Tensor) -> float:
        """计算gamma
        Args:
            x (Tensor): 输入张量
        Returns:
            Tensor: gamma
        """

        invalid = (1 - self.p) / (self.block_size ** 2)
        # valid = (x.shape[-1] ** 2) / ((x.shape[-1] - self.block_size + 1) ** 2)
        #这样写更准确
        valid = (x.shape[-1] * x.shape[-2]) / ((x.shape[-1] - self.block_size + 1) * (x.shape[-2] - self.block_size + 1))
        return invalid * valid

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.size()
        if self.training:
            gamma = self.calculate_gamma(x)
            mask_shape = (N, C, H - self.block_size + 1, W - self.block_size + 1)
            mask = torch.bernoulli(torch.full(mask_shape, gamma, device=x.device))
            mask = F.pad(mask, [self.block_size // 2] * 4, value=0)
            mask_block = 1 - F.max_pool2d(
                mask,
                kernel_size=(self.block_size, self.block_size),
                stride=(1, 1),
                padding=(self.block_size // 2, self.block_size // 2),
            )
            x = mask_block * x * (mask_block.numel() / mask_block.sum())
        return x

class DropBlock2D_Rectangle(nn.Module):
    def __init__(self, block_size: Tuple[int, int] = (5, 5), p: float = 0.1):
        super().__init__()
        self.block_size = block_size
        self.p = p

    def calculate_gamma(self, x: Tensor) -> float:
        """计算gamma
        Args:
            x (Tensor): 输入张量
        Returns:
            Tensor: gamma
        """
        # invalid = (1 - self.p) / (math.pi*(self.block_radius**2))   # Area of the circle
        invalid = (1 - self.p) / (self.block_size[0]*self.block_size[1])
        # valid = (x.shape[-1] ** 2) / ((x.shape[-1] - 2 * self.block_radius+1) ** 2)
        valid = (x.shape[2] * x.shape[3]) / (
                (x.shape[2] - self.block_size[0] + 1) * (x.shape[3] - self.block_size[1]+ 1))
        return invalid * valid

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.size()
        if self.training:
            gamma = self.calculate_gamma(x)
            mask_shape = (N, C, H - self.block_size[0] + 1, W - self.block_size[1] + 1)
            mask = torch.bernoulli(torch.full(mask_shape, gamma, device=x.device))
            mask = F.pad(mask, [self.block_size[1] // 2, self.block_size[1] // 2, self.block_size[0] // 2,
                                self.block_size[0] // 2], value=0)
            mask_block = 1 - F.max_pool2d(
                mask,
                kernel_size=(self.block_size[0], self.block_size[1]),
                stride=(1, 1),
                padding=(self.block_size[0] // 2, self.block_size[1] // 2),
            )
            x = mask_block * x * (mask_block.numel() / mask_block.sum())
        return x

class DropBlock2D_Diagonal(nn.Module):
    def __init__(self, a: int = 1,num_diag: int = 2, p: float = 0.1,diag_type: str = 'main'):
        super().__init__()
        self.a = a
        self.p = p
        self.num_diag = num_diag
        self.diag_type=diag_type

    def calculate_gamma(self, x: torch.Tensor) -> float:
        diag_l=1+2*self.a
        area=0
        times=self.num_diag
        while times!=0:
            area+=diag_l-times
            times=times-1
        invalid = (1 - self.p) / (math.sqrt(2)*diag_l+math.sqrt(2)*(2*area))
        valid = (x.shape[2] * x.shape[3]) / (
                (x.shape[2] - (2 * self.a + 1)+1) * (x.shape[3] - (2 * self.a + 1)+1))
        return invalid * valid

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.size()
        if self.training:
            gamma = self.calculate_gamma(x)
            mask_shape = (N, C, H - (2 * self.a + 1) + 1, W - (2 * self.a + 1) + 1)
            mask = torch.bernoulli(torch.full(mask_shape, gamma, device=x.device))
            mask = F.pad(mask, [(2 * self.a + 1) // 2] * 4, value=0).to(x.device)

            # Create a diag mask
            y_d, x_d = torch.meshgrid(torch.arange(-self.a, self.a + 1),
                                      torch.arange(-self.a, self.a + 1), indexing='ij')
            # diag_mask = (x_d + y_d <= self.a ** 2).to(mask.dtype).to(x.device)
            diag_mask = torch.zeros_like(x_d).to(mask.dtype).to(x.device)

            if self.diag_type=='main':
                for i in range(self.num_diag+1):
                    diag_mask[x_d == y_d + i] = 1
                    diag_mask[x_d == y_d - i] = 1

            if self.diag_type == 'secondary':
                for i in range(self.num_diag + 1):
                    diag_mask[x_d == -y_d + i] = 1
                    diag_mask[x_d == -y_d - i] = 1

            diag_mask = diag_mask.view(1, 1, 2 * self.a + 1, 2 * self.a + 1)
            # 使用 expand 操作复制数据
            diag_mask = diag_mask.expand(C, 1, 2 * self.a + 1, 2 * self.a + 1)

            mask_block = 1 - F.conv2d(mask, diag_mask, stride=(1, 1), padding=self.a, groups=C)
            mask_block[(mask_block < 0)] = 0
            x = mask_block * x * (mask_block.numel() / mask_block.sum())
        return x

class DropBlock2D_Circle(nn.Module):
    def __init__(self, block_radius: int = 3, p: float = 0.1):
        super().__init__()
        self.block_radius = block_radius
        self.p = p

    def calculate_gamma(self, x: Tensor) -> float:
        # invalid = (1 - self.p) / (math.pi*(self.block_radius**2)) # Area of the circle
        invalid = (1 - self.p) / ((self.block_radius ** 2) + (self.block_radius + 1) ** 2)
        valid = (x.shape[2] * x.shape[3]) / (
                    (x.shape[2] - (2 * self.block_radius + 1)+1) * (x.shape[3] - (2 * self.block_radius + 1)+1))
        return invalid * valid

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.size()
        if self.training:
            gamma = self.calculate_gamma(x)
            mask_shape = (N, C, H - (2 * self.block_radius+1)+1, W - (2 * self.block_radius+1)+1)
            mask = torch.bernoulli(torch.full(mask_shape, gamma, device=x.device))
            mask = F.pad(mask, [(2 * self.block_radius+1)//2] * 4, value=0).to(x.device)

            # Create a circular mask
            y_d, x_d = torch.meshgrid(torch.arange(-self.block_radius, self.block_radius + 1),
                                  torch.arange(-self.block_radius, self.block_radius + 1),indexing='ij')
            circular_mask = (x_d ** 2 + y_d ** 2 <= self.block_radius ** 2).to(mask.dtype).to(x.device)

            circular_mask=circular_mask.view(1, 1, 2 * self.block_radius + 1, 2 * self.block_radius + 1)
            # 使用 expand 操作复制数据
            circular_mask = circular_mask.expand(C, 1, 2 * self.block_radius + 1, 2 * self.block_radius + 1)

            mask_block = 1-F.conv2d(mask, circular_mask, stride=(1, 1), padding=self.block_radius,groups=C)
            mask_block[(mask_block < 0)] = 0
            x = mask_block * x * (mask_block.numel() / mask_block.sum())
        return x

class DropBlock2D_Triangular(nn.Module):
    def __init__(self, l: int = 5, p: float = 0.1,type:str = 'Upper'):
        super().__init__()
        self.l = l
        self.p = p
        self.type=type

    def calculate_gamma(self, x: Tensor) -> float:
        # invalid = (1 - self.p) / (self.l ** 2 // 2)  # 面积（近似）
        invalid = (1 - self.p) / sum(range(1, self.l + 1))  # 统计关系
        valid = (x.shape[2] * x.shape[3]) / ((x.shape[2] - self.l+1) *(x.shape[3] - self.l+1))
        return invalid * valid

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.size()

        if self.training:
            gamma = self.calculate_gamma(x)
            mask_shape = (N, C, H - self.l+1, W - self.l+1)
            mask = torch.bernoulli(torch.full(mask_shape, gamma, device=x.device))
            mask = F.pad(mask, [self.l//2] * 4, value=0).to(x.device)

            # Create an triangular mask
            y_d, x_d = torch.meshgrid(torch.arange(0, self.l),
                                      torch.arange(0, self.l), indexing='ij')

            if self.type=='Upper':
                # Create a boolean mask for the upper triangular region
                triangular_mask = (x_d >= y_d).to(mask.dtype).to(x.device)
            else:
                # Create a boolean mask for the lower triangular region
                triangular_mask = (x_d <= y_d).to(mask.dtype).to(x.device)

            # Reshape the mask to have the shape (1, 1, self.block_size, self.block_size)
            triangular_mask = triangular_mask.view(1, 1, self.l, self.l)
            # 使用 expand 操作复制数据
            triangular_mask = triangular_mask.expand(C, 1, self.l,self.l)

            mask_block = 1 - F.conv2d(mask, triangular_mask, stride=(1, 1), padding=self.l//2, groups=C)
            mask_block[(mask_block < 0)] = 0
            x = mask_block * x * (mask_block.numel() / mask_block.sum())
        return x

class SE_CA(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(SE_CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.fc1 = nn.Linear(in_features=inp, out_features=inp//reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=inp//reduction, out_features=inp, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(inp, inp // reduction, 1, bias=False)
        self.conv2 = nn.Conv2d(inp // reduction, inp, 1, bias=False)

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_a = self.avg_pool(x)

        x_a = x_a.view([n,c])
        x_a = self.fc1(x_a)
        x_a = self.relu(x_a)
        x_a = self.fc2(x_a)
        x_a_w = self.sigmoid(x_a)
        x_a_w = x_a_w.view([n,c,1,1])
        se_x = x * x_a_w

        se_x_h = self.pool_h(se_x)
        se_x_w = self.pool_w(se_x)

        se_x_h = self.conv1(se_x_h)
        se_x_h = self.relu(se_x_h)
        se_x_h = self.conv2(se_x_h)
        se_x_h_w = self.sigmoid(se_x_h)

        se_x_w = self.conv1(se_x_w)
        se_x_w = self.relu(se_x_w)
        se_x_w = self.conv2(se_x_w)
        se_x_w_w = self.sigmoid(se_x_w)

        out = identity * se_x_h_w * se_x_w_w

        return out

class InConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None,dp=1):
        if mid_channels is None:
            mid_channels = out_channels
        super(InConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            DropBlock2D_Diagonal(3,0,dp),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            DropBlock2D_Diagonal(3,0,dp),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None,dp=1):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            DropBlock2D_Diagonal(3,0,dp),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            DropBlock2D_Diagonal(3,0,dp),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class OneConv(nn.Sequential):
    def __init__(self, in_channels, out_channels,dp=1):
        super(OneConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels,dp=1):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels,dp=dp),
        )

class LastDown(nn.Sequential):
    def __init__(self, in_channels, out_channels,dp=1):
        super(LastDown, self).__init__(
            nn.MaxPool2d(2, stride=2),
            OneConv(in_channels, out_channels, dp=dp),
            SE_CA(out_channels,out_channels),
            OneConv(out_channels, out_channels,dp=dp)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True,dp=1):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2,dp=dp)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels,dp=dp)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        return x

class Merger(nn.Module):
    def __init__(self,channels,factor):
        super(Merger, self).__init__()
        self.up2=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv1=nn.Conv2d(channels*2//factor,channels,kernel_size=1)
        self.conv2=nn.Conv2d(channels*4//factor,channels,kernel_size=1)
        self.conv=nn.Conv2d(channels*3,channels,kernel_size=1)

    def forward(self,x1,x2,x3):
        x1 =self.conv2(x1)
        x1=self.up4(x1)
        # [N, C, H, W]
        diff_y = x3.size()[2] - x1.size()[2]
        diff_x = x3.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x2 = self.conv1(x2)
        x2=self.up2(x2)
        # [N, C, H, W]
        diff_y = x3.size()[2] - x2.size()[2]
        diff_x = x3.size()[3] - x2.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x2= F.pad(x2, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x=x1+x2+x3
        return x

class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 1,
                 bilinear: bool = True,
                 base_c: int = 64,
                 dp=1,
                 Drop=None
                 ):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = InConv(in_channels, base_c,dp=dp)
        self.down1 = Down(base_c, base_c * 2,dp=dp)
        self.down2 = Down(base_c * 2, base_c * 4,dp=dp)

        factor = 2 if bilinear else 1
        self.down3 = LastDown(base_c * 4, base_c * 8 // factor, dp=dp)

        self.up1 = Up(base_c * 8, base_c * 4 // factor, bilinear,dp=dp)
        self.up2 = Up(base_c * 4, base_c * 2 // factor, bilinear,dp=dp)
        self.up3 = Up(base_c * 2, base_c, bilinear,dp=dp)
        self.merger = Merger(base_c, factor)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        xo = self.up1(x4, x3)
        xs = self.up2(xo, x2)
        xt = self.up3(xs, x1)

        x = self.merger(xo, xs, xt)
        logits = self.out_conv(x)

        return logits

if __name__ == '__main__':
    unet=UNet(in_channels=3)
    x=torch.ones((1,3,565,584))
    # 使用thop库的profile函数估算FLOPs
    flops, params = profile(unet, inputs=(x,))

    # 将FLOPs和参数数目进行格式化，以便更容易阅读
    flops_formatted = clever_format([flops], "%.2f")
    params_formatted = clever_format([params], "%.2f")

    print(f"FLOPs: {flops_formatted}")
    print(f"Parameters: {params_formatted}")
    output=unet(x)
    print(output.shape)