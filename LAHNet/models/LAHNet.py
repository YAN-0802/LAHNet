import torch
from torch import nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
import torch.utils.checkpoint as cp
import torchvision

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_momentum=0.1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=bn_momentum)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_momentum=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class StageModule(nn.Module):
    def __init__(self, stage, output_branches, c, bn_momentum):
        super(StageModule, self).__init__()
        self.stage = stage
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.stage):
            w = c * (2 ** i)
            branch = nn.Sequential(
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()
        # for each output_branches (i.e. each branch in all cases but the very last one)
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.stage):  # for each branch
                if i == j:
                    self.fuse_layers[-1].append(nn.Sequential())  # Used in place of "None" because it is callable
                elif i < j:
                    self.fuse_layers[-1].append(nn.Sequential(
                        nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=(1, 1), stride=(1, 1), bias=False),
                        nn.BatchNorm2d(c * (2 ** i), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Upsample(scale_factor=(2.0 ** (j - i)), mode='bilinear', align_corners=True),
                    ))
                elif i > j:
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                      bias=False),  # downsampling
                            nn.BatchNorm2d(c * (2 ** j), eps=1e-05, momentum=0.1, affine=True,
                                           track_running_stats=True),
                            nn.ReLU(inplace=True),
                        ))
                    ops.append(nn.Sequential(
                        nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                  bias=False),
                        nn.BatchNorm2d(c * (2 ** i), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    ))
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        assert len(self.branches) == len(x)

        x = [branch(b) for branch, b in zip(self.branches, x)]

        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(0, len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    x_fused[i] = x_fused[i] + self.fuse_layers[i][j](x[j])

        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])

        return x_fused

def channel_shuffle(x, groups):
    """Channel Shuffle operation.
    This function enables cross-group information flow for multiple groups
    convolution layers.
    Args:
        x (Tensor): The input tensor.
        groups (int): The number of groups to divide the input tensor
            in the channel dimension.
    Returns:
        Tensor: The output tensor after channel shuffle operation.
    """

    batch_size, num_channels, height, width = x.size()
    assert (num_channels % groups == 0), ('num_channels should be '
                                          'divisible by groups')
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x

class ShuffleUnit(nn.Module):
    """InvertedResidual block for ShuffleNetV2 backbone.
    Args:
        in_channels (int): The input channels of the block.
        out_channels (int): The output channels of the block.
        stride (int): Stride of the 3x3 convolution layer. Default: 1
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 with_cp=False):
        super().__init__()
        self.stride = stride
        self.with_cp = with_cp

        branch_features = out_channels // 2
        if self.stride == 1:
            assert in_channels == branch_features * 2, (
                f'in_channels ({in_channels}) should equal to '
                f'branch_features * 2 ({branch_features * 2}) '
                'when stride is 1')

        if in_channels != branch_features * 2:
            assert self.stride != 1, (
                f'stride ({self.stride}) should not equal 1 when '
                f'in_channels != branch_features * 2')

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                ConvModule(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=self.stride,
                    padding=1,
                    groups=in_channels,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=None),
                ConvModule(
                    in_channels,
                    branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
            )

        self.branch2 = nn.Sequential(
            ConvModule(
                in_channels if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                groups=branch_features,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None),
            ConvModule(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))

    def forward(self, x):

        def _inner_forward(x):
            if self.stride > 1:
                out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)  # (b)
            else:
                x1, x2 = x.chunk(2, dim=1)
                out = torch.cat((x1, self.branch2(x2)), dim=1)  # (a)

            out = channel_shuffle(out, 2)

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out  # channels & width, height no change

class Stem(nn.Module):
    def __init__(self, in_channels,mid_channels,out_channels, bn_momentum=0.1):
        super(Stem, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.shuffle_unit = ShuffleUnit(mid_channels, out_channels, stride=1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        out = self.shuffle_unit(x)

        return out

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)

        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class AttentionFusionModule(nn.Module):
    def __init__(self, x_chan, y_chan, out_chan, *args, **kwargs):
        super(AttentionFusionModule, self).__init__()
        in_chan = x_chan+y_chan
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv = nn.Conv2d(out_chan,
                               out_chan,
                               kernel_size = 1,
                               stride = 1,
                               padding = 0,
                               bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid = nn.Sigmoid()
        self.xARM = AttentionRefinementModule(x_chan, out_chan)
        self.yARM = AttentionRefinementModule(y_chan, out_chan)
        self.init_weight()

    def forward(self, x, y):
        y_up = F.interpolate(y, x.size()[2:], mode='bilinear')
        fcat = torch.cat([x, y_up], dim=1)
        feat = self.convblk(fcat)

        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat

        x_new = self.xARM(x)
        y_new = self.yARM(y)
        y_new = F.interpolate(y_new, x_new.size()[2:], mode='bilinear')
        feat_out = feat_out + x_new + y_new
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

class DPblock2(nn.Module):
    def __init__(self, width, up_kwargs=None, norm_layer=nn.BatchNorm2d):
        super(DPblock2, self).__init__()
        self.up_kwargs = up_kwargs
        self.conv_out = nn.Sequential(
            nn.Conv2d(2 * width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation1 = nn.Sequential(
            SeparableConv2d(width, width, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(
            SeparableConv2d(width, width, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, inputs):
        feat = torch.cat([self.dilation1(inputs), self.dilation2(inputs)], dim=1)
        feat = self.conv_out(feat)
        return feat

class DPblock3(nn.Module):
    def __init__(self, width, up_kwargs=None, norm_layer=nn.BatchNorm2d):
        super(DPblock3, self).__init__()
        self.up_kwargs = up_kwargs
        self.conv_out = nn.Sequential(
            nn.Conv2d(3 * width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation1 = nn.Sequential(
            SeparableConv2d(width, width, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(
            SeparableConv2d(width, width, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(
            SeparableConv2d(width, width, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, inputs):
        feat = torch.cat([self.dilation1(inputs), self.dilation2(inputs), self.dilation3(inputs)], dim=1)
        feat = self.conv_out(feat)
        return feat

class DPblock4(nn.Module):
    def __init__(self, width, up_kwargs=None, norm_layer=nn.BatchNorm2d):
        super(DPblock4, self).__init__()
        self.up_kwargs = up_kwargs
        self.conv_out = nn.Sequential(
            nn.Conv2d(4 * width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation1 = nn.Sequential(
            SeparableConv2d(width, width, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(
            SeparableConv2d(width, width, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(
            SeparableConv2d(width, width, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation4 = nn.Sequential(
            SeparableConv2d(width, width, kernel_size=3, padding=8, dilation=8, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, inputs):
        feat = torch.cat([self.dilation1(inputs), self.dilation2(inputs), self.dilation3(inputs), self.dilation4(inputs)], dim=1)
        feat = self.conv_out(feat)
        return feat

class CoarseHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CoarseHead, self).__init__()
        self.raw_seg = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        return self.raw_seg(x)

class LAHNet(nn.Module):
    def __init__(self, c=4, bn_momentum=0.1, last_inp_channels=32):
        super(LAHNet, self).__init__()

        # Input (stem net)s
        self.stem = Stem(in_channels=3, mid_channels=8, out_channels=8, bn_momentum=bn_momentum)

        # Stage 1 (layer1)      - First group of bottleneck (resnet) modules
        self.layer1 = nn.Sequential(
            Bottleneck(8, 4),
            ShuffleUnit(8, 8),
            ShuffleUnit(8, 8),
        )

        # Fusion layer 1 (transition1)      - Creation of the first two branches (one full and one half resolution)
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(8, c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv2d(8, c * (2 ** 1), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c * (2 ** 1), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),
        ])

        # Stage 2 (stage2)      - Second module with 2 group of bottleneck (resnet) modules. This has 2 branches
        self.stage2 = nn.Sequential(
            StageModule(stage=2, output_branches=2, c=c, bn_momentum=bn_momentum),
            StageModule(stage=2, output_branches=2, c=c, bn_momentum=bn_momentum)
        )

        # Fusion layer 2 (transition2)      - Creation of the third branch (1/4 resolution)
        self.transition2 = nn.ModuleList([
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv2d(c * (2 ** 1), c * (2 ** 2), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c * (2 ** 2), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),
        ])

        # Stage 3 (stage3)      - Third module with 3 groups of bottleneck (resnet) modules. This has 3 branches
        self.stage3 = nn.Sequential(
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
        )
        # Fusion layer 3 (transition3)      - Creation of the fourth branch (1/8 resolution)
        self.transition3 = nn.ModuleList([
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv2d(c * (2 ** 2), c * (2 ** 3), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c * (2 ** 3), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),
        ])
        # Stage 4 (stage4)      - Fourth module with 2 groups of bottleneck (resnet) modules. This has 4 branches
        self.stage4 = nn.Sequential(
            StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum),
            StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum),
        )

        # Final layer (final_layer)
        self.final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(last_inp_channels, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False)
        )

        # Segmentation head (seg_head)
        self.seg_head = nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1)

        self.DP22 = DPblock2(width=8)
        self.DP33 = DPblock3(width=16)
        self.DP44 = DPblock4(width=32)
        self.cov3_3 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1)

        self.AFM1 = AttentionFusionModule(4, 8, 8)
        self.conv_head1 = ConvBNReLU(8, 8, ks=3, stride=1, padding=1)
        self.AFM2 = AttentionFusionModule(8, 16, 16)
        self.conv_head2 = ConvBNReLU(16, 16, ks=3, stride=1, padding=1)
        self.AFM3 = AttentionFusionModule(16, 32, 32)
        self.conv_head3 = ConvBNReLU(32, 32, ks=3, stride=1, padding=1)

        self.aux1 = CoarseHead(8, 1)
        self.aux2 = CoarseHead(16, 1)
        self.aux3 = CoarseHead(32, 1)

        self.conv1 = ConvBNReLU(18, 8, 3, stride=1)
        self.conv2 = ConvBNReLU(34, 16, 3, stride=1)
        self.conv3 = ConvBNReLU(66, 32, 3, stride=1)


    def forward(self, _input):
        x = _input[:,0:3,:,:]
        loc = _input[:,3,:,:].unsqueeze(1)
        h,w = x.size(2), x.size(3)

        x = self.stem(x)
        x = self.layer1(x)

        x = [trans(x) for trans in self.transition1]  # Since now, x is a list (# == nof branches)
        x = self.stage2(x)
        x[1] = self.DP22(x[1])

        # x = [trans(x[-1]) for trans in self.transition2]    # New branch derives from the "upper" branch only
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage3(x)
        # x = [trans(x) for trans in self.transition3]    # New branch derives from the "upper" branch only
        x[2] = self.DP33(x[2])
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage4(x)
        x[3] = self.DP44(x[3])
        x[3] = self.cov3_3(x[3])

        feat12 = self.AFM1(x[0], x[1])
        feat12 = self.conv_head1(feat12)
        loc_0 = torchvision.transforms.Resize((feat12.size()[2:]))(loc)
        feat12_1 = feat12 + loc_0 * feat12
        auxmask1 = self.aux1(feat12)
        feat12_2 = feat12 + torch.sigmoid(auxmask1) * feat12
        feat12 = torch.cat([feat12_1, feat12_2, auxmask1, auxmask1], dim=1)
        feat12 = self.conv1(feat12)

        feat23 = self.AFM2(feat12, x[2])
        feat23 = self.conv_head2(feat23)
        feat23_1 = feat23 + loc_0 * feat23
        auxmask2 = self.aux2(feat23)
        feat23_2 = feat23 + torch.sigmoid(auxmask2) * feat23
        feat23 = torch.cat([feat23_1, feat23_2, auxmask2, auxmask2], dim=1)
        feat23 = self.conv2(feat23)

        feat34 = self.AFM3(feat23, x[3])
        feat34 = self.conv_head3(feat34)
        feat34_1 = feat34 + loc_0 * feat34
        auxmask3 = self.aux3(feat34)
        feat34_2 = feat34 + torch.sigmoid(auxmask3) * feat34
        feat34 = torch.cat([feat34_1, feat34_2, auxmask3, auxmask3], dim=1)
        feat34 = self.conv3(feat34)

        feat = nn.functional.interpolate(feat34, size=(h, w), mode='bilinear', align_corners=True)
        x = self.final_layer(feat)
        seg = self.seg_head(x)

        return {
            'coarse_masks': [auxmask1, auxmask2, auxmask3],
            'pred_masks': [seg],
        }


# Calculate model parameters and computation
if __name__ == '__main__':
    from thop import profile
    net = LAHNet(4, 0.1, 32).cuda()
    example_input = torch.randn(1,4,640,480).cuda()

    flops, params = profile(net, (example_input,))
    print('net profile \n    Flops: {:.2f}G \n    Params: {:.2f}M'.format(flops/1e9, params/1e6))