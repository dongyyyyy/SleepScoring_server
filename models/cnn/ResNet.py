from include.header import *


def conv_1d(in_planes, out_planes, kernel_size=3, stride=1, groups=1, padding=1, dilation=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=dilation)


def conv3x3_1d(in_planes, out_planes, kernel_size=3, stride=1, groups=1, padding=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1_1d(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, block_kernel_size=3, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        # if groups != 1 or base_width != 64:
        #     raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # if dilation > 1:
        #     raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.dilation = dilation
        self.conv1 = conv_1d(in_planes=inplanes, out_planes=planes, kernel_size=block_kernel_size, stride=stride,
                             groups=groups, padding=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_1d(in_planes=planes, out_planes=planes, kernel_size=block_kernel_size, groups=groups,
                             padding=dilation)
        self.bn2 = norm_layer(planes)
        self.dropout = nn.Dropout(p=0.2)
        self.block_kernel_size = block_kernel_size
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # print('kernel size : ',self.block_kernel_size)
        # print('dilation : ', self.dilation)
        identity = x

        out = self.conv1(x)
        # print('out1.shape : ',out.shape )
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        # print('out2.shape : ',out.shape)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        # print('out : ',out.shape)
        # print('id : ',identity.shape)
        out += identity
        out = self.relu(out)
        out = self.dropout(out)

        return out


class ResNet_200hz(nn.Module):

    def __init__(self, block, layers, first_conv=[200, 40, 100], layer_filters=[64, 128, 256, 512], in_channel=1,
                 block_kernel_size=3, padding=1, num_classes=5, use_batchnorm=False, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_200hz, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer
        self.use_batchnorm = use_batchnorm
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1_1d = nn.Conv1d(in_channel, self.inplanes, kernel_size=first_conv[0], stride=first_conv[1],
                                  padding=first_conv[2],
                                  bias=False)
        # self.conv1_1d = nn.Conv1d(in_channel, self.inplanes, kernel_size=200, stride=40, padding=100,
        #                           bias=False)
        self.dropout = nn.Dropout(p=0.2)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.block_kernel_size = block_kernel_size
        self.padding = self.block_kernel_size // 2
        self.layer1 = self._make_layer(block, layer_filters[0], layers[0])
        self.layer2 = self._make_layer(block, layer_filters[1], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, layer_filters[2], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, layer_filters[3], layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool1d(1)  #
        self.fc = nn.Linear(layer_filters[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                '''
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                '''

                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1_1d(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []

        layers.append(
            block(self.inplanes, planes, block_kernel_size=self.block_kernel_size, stride=stride, downsample=downsample,
                  groups=self.groups,
                  base_width=self.base_width, dilation=self.padding,
                  norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, block_kernel_size=self.block_kernel_size, groups=self.groups,
                                base_width=self.base_width, dilation=self.padding,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1_1d(x)
        x = self.bn1(x)

        x = self.relu(x)
        x = self.dropout(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # -3
        x = torch.flatten(x, 1)  # -2
        x = self.fc(x)  # -1

        return x

    def forward(self, x):
        return self._forward_impl(x)


class ResNet_custom_FE(nn.Module):

    def __init__(self, block, layers, first_conv=[200, 40, 100], layer_filters=[64, 128, 256, 512], in_channel=1,
                 block_kernel_size=3, padding=1, num_classes=5, use_batchnorm=False, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_custom_FE, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer
        self.use_batchnorm = use_batchnorm
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1_1d = nn.Conv1d(in_channel, self.inplanes, kernel_size=first_conv[0], stride=first_conv[1],
                                  padding=first_conv[2],
                                  bias=False)
        # self.conv1_1d = nn.Conv1d(in_channel, self.inplanes, kernel_size=200, stride=40, padding=100,
        #                           bias=False)
        self.dropout = nn.Dropout(p=0.2)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.block_kernel_size = block_kernel_size
        self.padding = self.block_kernel_size // 2
        self.layer1 = self._make_layer(block, layer_filters[0], layers[0])
        self.layer2 = self._make_layer(block, layer_filters[1], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, layer_filters[2], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, layer_filters[3], layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool1d(1)  #

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                '''
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                '''

                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1_1d(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []

        layers.append(
            block(self.inplanes, planes, block_kernel_size=self.block_kernel_size, stride=stride, downsample=downsample,
                  groups=self.groups,
                  base_width=self.base_width, dilation=self.padding,
                  norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, block_kernel_size=self.block_kernel_size, groups=self.groups,
                                base_width=self.base_width, dilation=self.padding,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        # print(x.shape)
        x = self.conv1_1d(x)
        # print(x.shape)
        x = self.bn1(x)

        x = self.relu(x)
        x = self.dropout(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # -3
        x = torch.flatten(x, 1)  # -2

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet_custom(block, layers, layer_filters=[64, 128, 256, 512], in_channel=3, first_conv=[200, 40, 100],
                   block_kernel_size=3, padding=1, num_classes=1000, use_batchnorm=False, **kwargs):
    model = ResNet_200hz(block, layers, block_kernel_size=block_kernel_size, padding=padding, first_conv=first_conv,
                         layer_filters=layer_filters, in_channel=in_channel, num_classes=num_classes,
                         use_batchnorm=use_batchnorm, **kwargs)
    return model


# Batch_norm과 dropout을 같이한 모델
def resnet18_custom(in_channel=3, layer_filters=[64, 128, 256, 512], first_conv=[200, 40, 100], block_kernel_size=3,
                    padding=1, use_batchnorm=True, num_classes=5, progress=True, **kwargs):
    return _resnet_custom(BasicBlock, [2, 2, 2, 2], block_kernel_size=block_kernel_size, padding=padding,
                          layer_filters=layer_filters, in_channel=in_channel, first_conv=first_conv,
                          num_classes=num_classes, use_batchnorm=True,
                          **kwargs)


def _resnet_custom_FE(block, layers, layer_filters=[64, 128, 256, 512], in_channel=3, first_conv=[200, 40, 100],
                      block_kernel_size=3, padding=1, num_classes=1000, use_batchnorm=False, **kwargs):
    model = ResNet_custom_FE(block, layers, block_kernel_size=block_kernel_size, padding=padding, first_conv=first_conv,
                             layer_filters=layer_filters, in_channel=in_channel, num_classes=num_classes,
                             use_batchnorm=use_batchnorm, **kwargs)
    return model


def resnet18_200hz_FE(in_channel=3, layer_filters=[64, 128, 256, 512], first_conv=[200, 40, 100], block_kernel_size=3,
                      padding=1, use_batchnorm=True, num_classes=5, progress=True, **kwargs):
    return _resnet_custom_FE(BasicBlock, [2, 2, 2, 2], block_kernel_size=block_kernel_size, padding=padding,
                             layer_filters=layer_filters, in_channel=in_channel, first_conv=first_conv,
                             num_classes=num_classes, use_batchnorm=True,
                             **kwargs)


class BasicBlock_withoutDropout(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, block_kernel_size=3, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock_withoutDropout, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        # if groups != 1 or base_width != 64:
        #     raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # if dilation > 1:
        #     raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.dilation = dilation
        self.conv1 = conv_1d(in_planes=inplanes, out_planes=planes, kernel_size=block_kernel_size, stride=stride,
                             groups=groups, padding=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_1d(in_planes=planes, out_planes=planes, kernel_size=block_kernel_size, groups=groups,
                             padding=dilation)
        self.bn2 = norm_layer(planes)
        self.block_kernel_size = block_kernel_size
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # print('kernel size : ',self.block_kernel_size)
        # print('dilation : ', self.dilation)
        identity = x

        out = self.conv1(x)
        # print('out1.shape : ',out.shape )
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # print('out2.shape : ',out.shape)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        # print('out : ',out.shape)
        # print('id : ',identity.shape)
        out += identity
        out = self.relu(out)

        return out


class ResNet_200hz_withoutDropout(nn.Module):

    def __init__(self, block, layers, first_conv=[200, 40, 100], layer_filters=[64, 128, 256, 512], in_channel=1,
                 block_kernel_size=3, padding=1, num_classes=5, use_batchnorm=False, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_200hz_withoutDropout, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer
        self.use_batchnorm = use_batchnorm
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1_1d = nn.Conv1d(in_channel, self.inplanes, kernel_size=first_conv[0], stride=first_conv[1],
                                  padding=first_conv[2],
                                  bias=False)
        # self.conv1_1d = nn.Conv1d(in_channel, self.inplanes, kernel_size=200, stride=40, padding=100,
        #                           bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.block_kernel_size = block_kernel_size
        self.padding = self.block_kernel_size // 2
        self.layer1 = self._make_layer(block, layer_filters[0], layers[0])
        self.layer2 = self._make_layer(block, layer_filters[1], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, layer_filters[2], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, layer_filters[3], layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool1d(1)  #
        self.fc = nn.Linear(layer_filters[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                '''
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                '''

                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1_1d(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []

        layers.append(
            block(self.inplanes, planes, block_kernel_size=self.block_kernel_size, stride=stride, downsample=downsample,
                  groups=self.groups,
                  base_width=self.base_width, dilation=self.padding,
                  norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, block_kernel_size=self.block_kernel_size, groups=self.groups,
                                base_width=self.base_width, dilation=self.padding,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1_1d(x)
        x = self.bn1(x)

        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # -3
        x = torch.flatten(x, 1)  # -2
        x = self.fc(x)  # -1

        return x

    def forward(self, x):
        return self._forward_impl(x)


class ResNet_custom_withoutDropout_FE(nn.Module):

    def __init__(self, block, layers, first_conv=[200, 40, 100], layer_filters=[64, 128, 256, 512], in_channel=1,
                 block_kernel_size=3, padding=1, num_classes=5, use_batchnorm=False, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_custom_withoutDropout_FE, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer
        self.use_batchnorm = use_batchnorm
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1_1d = nn.Conv1d(in_channel, self.inplanes, kernel_size=first_conv[0], stride=first_conv[1],
                                  padding=first_conv[2],
                                  bias=False)
        # self.conv1_1d = nn.Conv1d(in_channel, self.inplanes, kernel_size=200, stride=40, padding=100,
        #                           bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.block_kernel_size = block_kernel_size
        self.padding = self.block_kernel_size // 2
        self.layer1 = self._make_layer(block, layer_filters[0], layers[0])
        self.layer2 = self._make_layer(block, layer_filters[1], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, layer_filters[2], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, layer_filters[3], layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool1d(1)  #

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                '''
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                '''

                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1_1d(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []

        layers.append(
            block(self.inplanes, planes, block_kernel_size=self.block_kernel_size, stride=stride, downsample=downsample,
                  groups=self.groups,
                  base_width=self.base_width, dilation=self.padding,
                  norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, block_kernel_size=self.block_kernel_size, groups=self.groups,
                                base_width=self.base_width, dilation=self.padding,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        # print(x.shape)
        x = self.conv1_1d(x)
        # print(x.shape)
        x = self.bn1(x)

        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # -3
        x = torch.flatten(x, 1)  # -2

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet_custom_withoutDropout(block, layers, layer_filters=[64, 128, 256, 512], in_channel=3,
                                  first_conv=[200, 40, 100], block_kernel_size=3, padding=1, num_classes=1000,
                                  use_batchnorm=False, **kwargs):
    model = ResNet_200hz_withoutDropout(block, layers, block_kernel_size=block_kernel_size, padding=padding,
                                        first_conv=first_conv, layer_filters=layer_filters, in_channel=in_channel,
                                        num_classes=num_classes, use_batchnorm=use_batchnorm, **kwargs)
    return model


# Batch_norm과 dropout을 같이한 모델
def resnet18_custom_withoutDropout(in_channel=3, layer_filters=[64, 128, 256, 512], first_conv=[200, 40, 100],
                                   block_kernel_size=3, padding=1, use_batchnorm=True, num_classes=5, progress=True,
                                   **kwargs):
    return _resnet_custom_withoutDropout(BasicBlock_withoutDropout, [2, 2, 2, 2], block_kernel_size=block_kernel_size,
                                         padding=padding, layer_filters=layer_filters, in_channel=in_channel,
                                         first_conv=first_conv, num_classes=num_classes, use_batchnorm=use_batchnorm,
                                         **kwargs)


def _resnet_custom_withoutDropout_FE(block, layers, layer_filters=[64, 128, 256, 512], in_channel=3,
                                     first_conv=[200, 40, 100], block_kernel_size=3, padding=1, num_classes=1000,
                                     use_batchnorm=False, **kwargs):
    model = ResNet_custom_withoutDropout_FE(block, layers, block_kernel_size=block_kernel_size, padding=padding,
                                            first_conv=first_conv, layer_filters=layer_filters, in_channel=in_channel,
                                            num_classes=num_classes, use_batchnorm=use_batchnorm, **kwargs)
    return model


def resnet18_200hz_withoutDropout_FE(in_channel=3, layer_filters=[64, 128, 256, 512], first_conv=[200, 40, 100],
                                     block_kernel_size=3, padding=1, use_batchnorm=True, num_classes=5, progress=True,
                                     **kwargs):
    return _resnet_custom_withoutDropout_FE(BasicBlock_withoutDropout, [2, 2, 2, 2],
                                            block_kernel_size=block_kernel_size, padding=padding,
                                            layer_filters=layer_filters, in_channel=in_channel, first_conv=first_conv,
                                            num_classes=num_classes, use_batchnorm=True,
                                            **kwargs)


class resnet18_200hz_withoutDropout_ensemble(nn.Module):
    def __init__(self, block=BasicBlock_withoutDropout, layers=[2, 2, 2, 2], eeg_channel=[0], eog_channel=[1, 2],
                 layer_filters=[64, 128, 256, 512], first_conv=[199, 40, 99], block_kernel_size=3, padding=1,
                 use_batchnorm=True,
                 num_classes=5, progress=True):
        super(resnet18_200hz_withoutDropout_ensemble, self).__init__()
        self.eeg_channel = eeg_channel
        self.eog_channel = eog_channel
        self.eeg_featureExtract = ResNet_custom_withoutDropout_FE(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv, layer_filters=layer_filters,
                                                                  in_channel=1,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)
        self.eog_featureExtract = ResNet_custom_withoutDropout_FE(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv, layer_filters=layer_filters,
                                                                  in_channel=2,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(layer_filters[3] * 2, num_classes)

    def forward(self, x):
        out_eeg = self.eeg_featureExtract(x[:, self.eeg_channel, :])
        out_eog = self.eog_featureExtract(x[:, self.eog_channel, :])
        out = torch.cat((out_eeg, out_eog), dim=1)

        out = self.dropout(out)
        out = self.fc(out)
        return out

class resnet18_200hz_withoutDropout_ensemble_branch(nn.Module):
    def __init__(self, block=BasicBlock_withoutDropout, layers=[2, 2, 2, 2], eeg_channel=[0], eog_channel=[1, 2],
                 layer_filters=[64, 128, 256, 512], first_conv_small=[99,40,49],first_conv_big=[199, 40, 99], block_kernel_size=3, padding=1,
                 use_batchnorm=True,
                 num_classes=5, progress=True):
        super(resnet18_200hz_withoutDropout_ensemble_branch, self).__init__()
        self.eeg_channel = eeg_channel
        self.eog_channel = eog_channel
        self.eeg_featureExtract_small = ResNet_custom_withoutDropout_FE(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv_small, layer_filters=layer_filters,
                                                                  in_channel=1,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)

        self.eeg_featureExtract_big = ResNet_custom_withoutDropout_FE(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv_big, layer_filters=layer_filters,
                                                                  in_channel=1,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)
        self.eog_featureExtract_small = ResNet_custom_withoutDropout_FE(block, layers,
                                                                        block_kernel_size=block_kernel_size,
                                                                        padding=padding,
                                                                        first_conv=first_conv_small,
                                                                        layer_filters=layer_filters,
                                                                        in_channel=2,
                                                                        num_classes=num_classes,
                                                                        use_batchnorm=use_batchnorm)

        self.eog_featureExtract_big = ResNet_custom_withoutDropout_FE(block, layers,
                                                                      block_kernel_size=block_kernel_size,
                                                                      padding=padding,
                                                                      first_conv=first_conv_big,
                                                                      layer_filters=layer_filters,
                                                                      in_channel=2,
                                                                      num_classes=num_classes,
                                                                      use_batchnorm=use_batchnorm)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(layer_filters[3] * 4, num_classes)

    def forward(self, x):
        out_eeg_big = self.eeg_featureExtract_big(x[:, self.eeg_channel, :])
        out_eeg_small = self.eeg_featureExtract_small(x[:, self.eeg_channel, :])
        out_eog_big = self.eog_featureExtract_big(x[:, self.eog_channel, :])
        out_eog_small = self.eog_featureExtract_small(x[:, self.eog_channel, :])
        out = torch.cat((out_eeg_big,out_eeg_small,out_eog_big, out_eog_small), dim=1)

        out = self.dropout(out)
        out = self.fc(out)
        return out

class resnet18_200hz_withoutDropout_ensemble_branch_new(nn.Module):
    def __init__(self, block=BasicBlock_withoutDropout, layers=[2, 2, 2, 2], eeg_channel=[0], eog_channel=[1, 2],
                 layer_filters=[64, 128, 256, 512], first_conv_small=[99,40,49],first_conv_big=[199, 40, 99], block_kernel_size=3, padding=1,
                 use_batchnorm=True,
                 num_classes=5, progress=True):
        super(resnet18_200hz_withoutDropout_ensemble_branch_new, self).__init__()
        self.eeg_channel = eeg_channel
        self.eog_channel = eog_channel
        self.eeg_featureExtract_small = ResNet_custom_withoutDropout_FE(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv_small, layer_filters=layer_filters,
                                                                  in_channel=1,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)

        self.eeg_featureExtract_big = ResNet_custom_withoutDropout_FE(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv_big, layer_filters=layer_filters,
                                                                  in_channel=1,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)


        self.eog_featureExtract_big = ResNet_custom_withoutDropout_FE(block, layers,
                                                                      block_kernel_size=block_kernel_size,
                                                                      padding=padding,
                                                                      first_conv=first_conv_big,
                                                                      layer_filters=layer_filters,
                                                                      in_channel=2,
                                                                      num_classes=num_classes,
                                                                      use_batchnorm=use_batchnorm)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(layer_filters[3] * 3, num_classes)

    def forward(self, x):
        out_eeg_big = self.eeg_featureExtract_big(x[:, self.eeg_channel, :])
        out_eeg_small = self.eeg_featureExtract_small(x[:, self.eeg_channel, :])
        out_eog_big = self.eog_featureExtract_big(x[:, self.eog_channel, :])
        out = torch.cat((out_eeg_big,out_eeg_small,out_eog_big), dim=1)

        out = self.dropout(out)
        out = self.fc(out)
        return out

class resnet18_200hz_withoutDropout_ensemble_oneChannel(nn.Module):
    def __init__(self, block=BasicBlock_withoutDropout, layers=[2, 2, 2, 2], eeg_channel=[0], eog_channel=[1, 2],
                 layer_filters=[64, 128, 256, 512], first_conv_small=[99,40,49],first_conv_big=[399, 40, 199], block_kernel_size=3, padding=1,
                 use_batchnorm=True,
                 num_classes=5, progress=True):
        super(resnet18_200hz_withoutDropout_ensemble_oneChannel, self).__init__()
        self.eeg_channel = eeg_channel
        self.eog_channel = eog_channel
        self.eeg_featureExtract_small = ResNet_custom_withoutDropout_FE(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv_small, layer_filters=layer_filters,
                                                                  in_channel=2,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)

        self.eeg_featureExtract_big = ResNet_custom_withoutDropout_FE(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv_big, layer_filters=layer_filters,
                                                                  in_channel=2,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)


        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(layer_filters[3] * 2, num_classes)

    def forward(self, x):
        out_eog_big = self.eeg_featureExtract_big(x)
        out_eog_small = self.eeg_featureExtract_small(x)
        out = torch.cat((out_eog_big,out_eog_small), dim=1)

        out = self.dropout(out)
        out = self.fc(out)
        return out


class ResNet_custom_withoutDropout_FE_pretrained(nn.Module):
    def __init__(self, in_channel=1, layer_filters=[64, 128, 256, 512], first_conv=[200, 40, 100],
                                   block_kernel_size=3, padding=1, use_batchnorm=True, num_classes=5, progress=True,
                                   **kwargs):
        super(ResNet_custom_withoutDropout_FE_pretrained, self).__init__()
        self.featureExtract = ResNet_custom_withoutDropout_FE(block=BasicBlock_withoutDropout,layers=[2,2,2,2],first_conv=first_conv,
                                                              layer_filters=layer_filters,in_channel=in_channel,block_kernel_size=block_kernel_size,padding=padding,
                                                              num_classes=num_classes,use_batchnorm=use_batchnorm)
        self.dropout = nn.Dropout(p=0.5)
        self.flat = layer_filters[3]

    def forward(self, input):
        out = self.featureExtract(input)

        out = out.unsqueeze(0)

        return out

class resnet18_200hz_withoutDropout_ensemble_branch_FE(nn.Module):
    def __init__(self, block=BasicBlock_withoutDropout, layers=[2, 2, 2, 2], eeg_channel=[0], eog_channel=[1, 2],
                 layer_filters=[64, 128, 256, 512], first_conv_small=[99,40,49],first_conv_big=[199, 40, 99], block_kernel_size=3, padding=1,
                 use_batchnorm=True,
                 num_classes=5, progress=True):
        super(resnet18_200hz_withoutDropout_ensemble_branch_FE, self).__init__()
        self.eeg_channel = eeg_channel
        self.eog_channel = eog_channel
        self.eeg_featureExtract_small = ResNet_custom_withoutDropout_FE(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv_small, layer_filters=layer_filters,
                                                                  in_channel=1,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)

        self.eeg_featureExtract_big = ResNet_custom_withoutDropout_FE(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv_big, layer_filters=layer_filters,
                                                                  in_channel=1,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)
        self.eog_featureExtract_small = ResNet_custom_withoutDropout_FE(block, layers,
                                                                        block_kernel_size=block_kernel_size,
                                                                        padding=padding,
                                                                        first_conv=first_conv_small,
                                                                        layer_filters=layer_filters,
                                                                        in_channel=2,
                                                                        num_classes=num_classes,
                                                                        use_batchnorm=use_batchnorm)

        self.eog_featureExtract_big = ResNet_custom_withoutDropout_FE(block, layers,
                                                                      block_kernel_size=block_kernel_size,
                                                                      padding=padding,
                                                                      first_conv=first_conv_big,
                                                                      layer_filters=layer_filters,
                                                                      in_channel=2,
                                                                      num_classes=num_classes,
                                                                      use_batchnorm=use_batchnorm)



    def forward(self, x):
        out_eeg_big = self.eeg_featureExtract_big(x[:, self.eeg_channel, :])
        out_eeg_small = self.eeg_featureExtract_small(x[:, self.eeg_channel, :])
        out_eog_big = self.eog_featureExtract_big(x[:, self.eog_channel, :])
        out_eog_small = self.eog_featureExtract_small(x[:, self.eog_channel, :])
        out = torch.cat((out_eeg_big,out_eeg_small,out_eog_big, out_eog_small), dim=1)

        out = out.unsqueeze(0)

        return out


class resnet18_200hz_withoutDropout_ensemble_FE(nn.Module):
    def __init__(self, block=BasicBlock_withoutDropout, layers=[2, 2, 2, 2], eeg_channel=[0], eog_channel=[1, 2],
                 layer_filters=[64, 128, 256, 512], first_conv=[199, 40, 99], block_kernel_size=3, padding=1,
                 use_batchnorm=True,
                 num_classes=5, progress=True):
        super(resnet18_200hz_withoutDropout_ensemble_FE, self).__init__()
        self.eeg_channel = eeg_channel
        self.eog_channel = eog_channel
        self.eeg_featureExtract = ResNet_custom_withoutDropout_FE(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv, layer_filters=layer_filters,
                                                                  in_channel=1,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)
        self.eog_featureExtract = ResNet_custom_withoutDropout_FE(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv, layer_filters=layer_filters,
                                                                  in_channel=2,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)


    def forward(self, x):
        out_eeg = self.eeg_featureExtract(x[:, self.eeg_channel, :])
        out_eog = self.eog_featureExtract(x[:, self.eog_channel, :])
        out = torch.cat((out_eeg, out_eog), dim=1)

        out = out.unsqueeze(0)
        return out


class resnet18_200hz_withDropout_ensemble_branch_new(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], eeg_channel=[0], eog_channel=[1, 2],
                 layer_filters=[64, 128, 256, 512], first_conv_small=[99,40,49],first_conv_big=[199, 40, 99], block_kernel_size=3, padding=1,
                 use_batchnorm=True,
                 num_classes=5, progress=True):
        super(resnet18_200hz_withDropout_ensemble_branch_new, self).__init__()
        self.eeg_channel = eeg_channel
        self.eog_channel = eog_channel
        self.eeg_featureExtract_small = ResNet_custom_FE(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv_small, layer_filters=layer_filters,
                                                                  in_channel=1,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)

        self.eeg_featureExtract_big = ResNet_custom_FE(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv_big, layer_filters=layer_filters,
                                                                  in_channel=1,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)


        self.eog_featureExtract_big = ResNet_custom_FE(block, layers,
                                                                      block_kernel_size=block_kernel_size,
                                                                      padding=padding,
                                                                      first_conv=first_conv_big,
                                                                      layer_filters=layer_filters,
                                                                      in_channel=2,
                                                                      num_classes=num_classes,
                                                                      use_batchnorm=use_batchnorm)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(layer_filters[3] * 3, num_classes)

    def forward(self, x):
        out_eeg_big = self.eeg_featureExtract_big(x[:, self.eeg_channel, :])
        out_eeg_small = self.eeg_featureExtract_small(x[:, self.eeg_channel, :])
        out_eog_big = self.eog_featureExtract_big(x[:, self.eog_channel, :])
        out = torch.cat((out_eeg_big,out_eeg_small,out_eog_big), dim=1)

        out = self.dropout(out)
        out = self.fc(out)
        return out

class resnet18_200hz_withDropout_ensemble_branch_new_FE(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], eeg_channel=[0], eog_channel=[1, 2],
                 layer_filters=[64, 128, 256, 512], first_conv_small=[99,40,49],first_conv_big=[199, 40, 99], block_kernel_size=3, padding=1,
                 use_batchnorm=True,
                 num_classes=5, progress=True):
        super(resnet18_200hz_withDropout_ensemble_branch_new_FE, self).__init__()
        self.eeg_channel = eeg_channel
        self.eog_channel = eog_channel
        self.eeg_featureExtract_small = ResNet_custom_FE(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv_small, layer_filters=layer_filters,
                                                                  in_channel=1,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)

        self.eeg_featureExtract_big = ResNet_custom_FE(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv_big, layer_filters=layer_filters,
                                                                  in_channel=1,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)


        self.eog_featureExtract_big = ResNet_custom_FE(block, layers,
                                                                      block_kernel_size=block_kernel_size,
                                                                      padding=padding,
                                                                      first_conv=first_conv_big,
                                                                      layer_filters=layer_filters,
                                                                      in_channel=2,
                                                                      num_classes=num_classes,
                                                                      use_batchnorm=use_batchnorm)


    def forward(self, x):
        # print('x shape : ',x.shape)
        out_eeg_big = self.eeg_featureExtract_big(x[:, self.eeg_channel, :])
        out_eeg_small = self.eeg_featureExtract_small(x[:, self.eeg_channel, :])
        out_eog_big = self.eog_featureExtract_big(x[:, self.eog_channel, :])
        out = torch.cat((out_eeg_big,out_eeg_small,out_eog_big), dim=1)

        
        return out


class resnet18_200hz_withDropout_temporal(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], eeg_channel1=[0],eeg_channel2=[1], eog_channel=[2],
                 layer_filters=[64, 128, 256, 512], first_conv_small=[99,10,49],first_conv_big=[399, 40, 199], block_kernel_size=3, padding=1,
                 use_batchnorm=True,
                 num_classes=5, progress=True):
        super(resnet18_200hz_withDropout_temporal, self).__init__()
        self.eeg_channel1 = eeg_channel1
        self.eeg_channel2 = eeg_channel2
        self.eog_channel = eog_channel
        self.eeg_featureExtract_small1 = ResNet_custom_FE(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv_small, layer_filters=layer_filters,
                                                                  in_channel=1,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)

        self.eeg_featureExtract_big1 = ResNet_custom_FE(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv_big, layer_filters=layer_filters,
                                                                  in_channel=1,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)

        self.eeg_featureExtract_small2 = ResNet_custom_FE(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv_small, layer_filters=layer_filters,
                                                                  in_channel=1,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)

        self.eeg_featureExtract_big2 = ResNet_custom_FE(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv_big, layer_filters=layer_filters,
                                                                  in_channel=1,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)


        self.eog_featureExtract = ResNet_custom_FE(block, layers,
                                                                      block_kernel_size=block_kernel_size,
                                                                      padding=padding,
                                                                      first_conv=first_conv_small,
                                                                      layer_filters=layer_filters,
                                                                      in_channel=1,
                                                                      num_classes=num_classes,
                                                                      use_batchnorm=use_batchnorm)
        
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(layer_filters[3] * 5, 1024) # 512 X 5
        self.classification = nn.Linear(1024,num_classes)

    def forward(self, x):
        out_eeg_big1 = self.eeg_featureExtract_big1(x[:, self.eeg_channel1, :])
        out_eeg_small1 = self.eeg_featureExtract_small1(x[:, self.eeg_channel1, :])
        out_eeg_big2 = self.eeg_featureExtract_big2(x[:, self.eeg_channel2, :])
        out_eeg_small2 = self.eeg_featureExtract_small2(x[:, self.eeg_channel2, :])
        out_eog = self.eog_featureExtract(x[:, self.eog_channel, :])
        out = torch.cat((out_eeg_big1,out_eeg_small1,out_eeg_big2,out_eeg_small2,out_eog), dim=1)

        out = self.dropout(out)
        out = self.fc(out)
        out = self.classification(out)

        return out



class resnet18_200hz_withDropout_temporal_FE(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], eeg_channel1=[0],eeg_channel2=[1], eog_channel=[2],
                 layer_filters=[64, 128, 256, 512], first_conv_small=[99,10,49],first_conv_big=[399, 40, 199], block_kernel_size=3, padding=1,
                 use_batchnorm=True,
                 num_classes=5, progress=True):
        super(resnet18_200hz_withDropout_temporal_FE, self).__init__()
        self.eeg_channel1 = eeg_channel1
        self.eeg_channel2 = eeg_channel2
        self.eog_channel = eog_channel
        self.eeg_featureExtract_small1 = ResNet_custom_FE(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv_small, layer_filters=layer_filters,
                                                                  in_channel=1,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)

        self.eeg_featureExtract_big1 = ResNet_custom_FE(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv_big, layer_filters=layer_filters,
                                                                  in_channel=1,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)

        self.eeg_featureExtract_small2 = ResNet_custom_FE(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv_small, layer_filters=layer_filters,
                                                                  in_channel=1,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)

        self.eeg_featureExtract_big2 = ResNet_custom_FE(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv_big, layer_filters=layer_filters,
                                                                  in_channel=1,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)


        self.eog_featureExtract = ResNet_custom_FE(block, layers,
                                                                      block_kernel_size=block_kernel_size,
                                                                      padding=padding,
                                                                      first_conv=first_conv_small,
                                                                      layer_filters=layer_filters,
                                                                      in_channel=1,
                                                                      num_classes=num_classes,
                                                                      use_batchnorm=use_batchnorm)

    def forward(self, x):
        out_eeg_big1 = self.eeg_featureExtract_big1(x[:, self.eeg_channel1, :])
        out_eeg_small1 = self.eeg_featureExtract_small1(x[:, self.eeg_channel1, :])
        out_eeg_big2 = self.eeg_featureExtract_big2(x[:, self.eeg_channel2, :])
        out_eeg_small2 = self.eeg_featureExtract_small2(x[:, self.eeg_channel2, :])
        out_eog = self.eog_featureExtract(x[:, self.eog_channel, :])
        out = torch.cat((out_eeg_big1,out_eeg_small1,out_eeg_big2,out_eeg_small2,out_eog), dim=1)

        return out


class ResNet_custom_FE_withoutGAP(nn.Module):
    
    def __init__(self, block, layers, first_conv=[200, 40, 100], layer_filters=[64, 128, 256, 512], in_channel=1,
                 block_kernel_size=3, padding=1, num_classes=5, use_batchnorm=False, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_custom_FE_withoutGAP, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer
        self.use_batchnorm = use_batchnorm
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1_1d = nn.Conv1d(in_channel, self.inplanes, kernel_size=first_conv[0], stride=first_conv[1],
                                  padding=first_conv[2],
                                  bias=False)
        # self.conv1_1d = nn.Conv1d(in_channel, self.inplanes, kernel_size=200, stride=40, padding=100,
        #                           bias=False)
        self.dropout = nn.Dropout(p=0.2)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.block_kernel_size = block_kernel_size
        self.padding = self.block_kernel_size // 2
        self.layer1 = self._make_layer(block, layer_filters[0], layers[0])
        self.layer2 = self._make_layer(block, layer_filters[1], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, layer_filters[2], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, layer_filters[3], layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                '''
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                '''

                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1_1d(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []

        layers.append(
            block(self.inplanes, planes, block_kernel_size=self.block_kernel_size, stride=stride, downsample=downsample,
                  groups=self.groups,
                  base_width=self.base_width, dilation=self.padding,
                  norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, block_kernel_size=self.block_kernel_size, groups=self.groups,
                                base_width=self.base_width, dilation=self.padding,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        # print(x.shape)
        x = self.conv1_1d(x)
        # print(x.shape)
        x = self.bn1(x)

        x = self.relu(x)
        x = self.dropout(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)  # -3
        # x = torch.flatten(x, 1)  # -2

        return x

    def forward(self, x):
        return self._forward_impl(x)

class resnet18_200hz_withDropout_ensemble_branch_twoChannel_attention(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], eeg_channel=[0], eog_channel=[1],
                 layer_filters=[64, 128, 128, 256], first_conv_small=[99, 10, 49],first_conv_big=[399, 40, 199], block_kernel_size=3, padding=1,
                 use_batchnorm=True,
                 num_classes=5,sample_rate = 200,epoch_size=30, progress=True):
        super(resnet18_200hz_withDropout_ensemble_branch_twoChannel_attention, self).__init__()
        self.eeg_channel = eeg_channel
        self.eog_channel = eog_channel
        self.eeg_featureExtract_small = ResNet_custom_FE_withoutGAP(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv_small, layer_filters=layer_filters,
                                                                  in_channel=1,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)

        self.eeg_featureExtract_big = ResNet_custom_FE_withoutGAP(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv_big, layer_filters=layer_filters,
                                                                  in_channel=1,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)


        self.eog_featureExtract = ResNet_custom_FE_withoutGAP(block, layers,
                                                                      block_kernel_size=block_kernel_size,
                                                                      padding=padding,
                                                                      first_conv=first_conv_small,
                                                                      layer_filters=layer_filters,
                                                                      in_channel=1,
                                                                      num_classes=num_classes,
                                                                      use_batchnorm=use_batchnorm)

        self.small_sizefilter = math.ceil(math.ceil(math.ceil(math.ceil(math.ceil((sample_rate * epoch_size)/first_conv_small[1])/2)/2)/2)/2)
        self.big_sizefilter = math.ceil(math.ceil(math.ceil(math.ceil(math.ceil((sample_rate * epoch_size)/first_conv_big[1])/2)/2)/2)/2)

        self.gap_eeg_small = nn.AdaptiveAvgPool1d(1) # GAP
        self.gap_eeg_big = nn.AdaptiveAvgPool1d(1) # GAP
        self.gap_eog = nn.AdaptiveAvgPool1d(1) # GAP

        self.conditional_layer1 = nn.Linear(layer_filters[3],layer_filters[3]//2)
        self.conditional_layer2 = nn.Linear(layer_filters[3]//2,3)
        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(p=0.5)
        
        self.fc = nn.Linear(self.big_sizefilter*layer_filters[3] + 2*layer_filters[3]*self.small_sizefilter, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.classification = nn.Linear(1024, num_classes)
        

        

    def forward(self, x):
        batch_size = x.size(0)
        out_eeg_small = self.eeg_featureExtract_small(x[:, self.eeg_channel, :]) # 38 X 512
        out_eeg_big = self.eeg_featureExtract_big(x[:, self.eeg_channel, :]) # 10 X 512
        out_eog = self.eog_featureExtract(x[:, self.eog_channel, :]) # 10 X 512
        
        feature_weight_eeg_small = self.gap_eeg_small(out_eeg_small) # 1 X 512
        feature_weight_eeg_big = self.gap_eeg_big(out_eeg_big) # 1 X 512
        feature_weight_eog = self.gap_eog(out_eog) # 1 X 512

        feature_weight = feature_weight_eeg_small+feature_weight_eeg_big+feature_weight_eog
        feature_weight = torch.flatten(feature_weight,1)

        feature_weight = self.conditional_layer1(feature_weight)
        feature_weight = self.relu(feature_weight)
        feature_weight = self.dropout(feature_weight)
        feature_weight = self.conditional_layer2(feature_weight)
        feature_weight = self.relu(feature_weight)
        feature_weight = self.dropout(feature_weight)

        feature_weight = self.softmax(feature_weight)

        #Attention
        out_eeg_small = feature_weight[:,0].unsqueeze(1).unsqueeze(2) * out_eeg_small
        out_eeg_big = feature_weight[:,1].unsqueeze(1).unsqueeze(2) * out_eeg_big
        out_eog = feature_weight[:,2].unsqueeze(1).unsqueeze(2) * out_eog

        # print('weight shape : ',feature_weight.shape)
        # out = torch.cat((out_eeg_big,out_eeg_small,out_eog_big), dim=1)
        out_eeg_small = torch.flatten(out_eeg_small,1)
        out_eeg_big   = torch.flatten(out_eeg_big,1)
        out_eog   = torch.flatten(out_eog,1)
        
        out = torch.cat((out_eeg_small,out_eeg_big,out_eog), dim=1)

        out = self.dropout(out)

        out = self.fc(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.classification(out)
        return out

class resnet18_200hz_withDropout_ensemble_branch_twoChannel_attention_new(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], eeg_channel=[0], eog_channel=[1],
                 layer_filters=[64, 128, 128, 256], first_conv_small=[99, 20, 49],first_conv_big=[399, 20, 199], block_kernel_size=3, padding=1,
                 use_batchnorm=True,embedding_size=512,
                 num_classes=5,sample_rate = 200,epoch_size=30, progress=True):
        super(resnet18_200hz_withDropout_ensemble_branch_twoChannel_attention_new, self).__init__()
        self.eeg_channel = eeg_channel
        self.eog_channel = eog_channel
        self.eeg_featureExtract_small = ResNet_custom_FE_withoutGAP(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv_small, layer_filters=layer_filters,
                                                                  in_channel=1,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)

        self.eeg_featureExtract_big = ResNet_custom_FE_withoutGAP(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv_big, layer_filters=layer_filters,
                                                                  in_channel=1,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)


        self.eog_featureExtract = ResNet_custom_FE_withoutGAP(block, layers,
                                                                      block_kernel_size=block_kernel_size,
                                                                      padding=padding,
                                                                      first_conv=first_conv_small,
                                                                      layer_filters=layer_filters,
                                                                      in_channel=1,
                                                                      num_classes=num_classes,
                                                                      use_batchnorm=use_batchnorm)

        self.small_sizefilter = math.ceil(math.ceil(math.ceil(math.ceil(math.ceil((sample_rate * epoch_size)/first_conv_small[1])/2)/2)/2)/2)
        self.big_sizefilter = math.ceil(math.ceil(math.ceil(math.ceil(math.ceil((sample_rate * epoch_size)/first_conv_big[1])/2)/2)/2)/2)

        self.gap_eeg_small = nn.AdaptiveAvgPool1d(1) # GAP
        self.gap_eeg_big = nn.AdaptiveAvgPool1d(1) # GAP
        self.gap_eog = nn.AdaptiveAvgPool1d(1) # GAP

        self.conditional_layer1 = nn.Linear(layer_filters[3],layer_filters[3]//2) 
        self.conditional_layer2 = nn.Linear(layer_filters[3]//2,3) 
        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(p=0.5)
        
        self.embedding_size = embedding_size

        self.conv = nn.Conv1d(layer_filters[3]*3, 512, kernel_size=self.big_sizefilter, stride=self.big_sizefilter, bias=False)
        self.bn = nn.BatchNorm1d(512)
        # self.fc = nn.Linear(self.big_sizefilter*layer_filters[3] + 2*layer_filters[3]*self.small_sizefilter, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.classification = nn.Linear(512, num_classes)
        

        

    def forward(self, x):
        batch_size = x.size(0)
        out_eeg_small = self.eeg_featureExtract_small(x[:, self.eeg_channel, :]) # 38 X 512
        out_eeg_big = self.eeg_featureExtract_big(x[:, self.eeg_channel, :]) # 10 X 512
        out_eog = self.eog_featureExtract(x[:, self.eog_channel, :]) # 10 X 512
        
        feature_weight_eeg_small = self.gap_eeg_small(out_eeg_small) # 1 X 512
        feature_weight_eeg_big = self.gap_eeg_big(out_eeg_big) # 1 X 512
        feature_weight_eog = self.gap_eog(out_eog) # 1 X 512

        feature_weight = feature_weight_eeg_small+feature_weight_eeg_big+feature_weight_eog
        feature_weight = torch.flatten(feature_weight,1)

        feature_weight = self.conditional_layer1(feature_weight)
        feature_weight = self.relu(feature_weight)
        feature_weight = self.dropout(feature_weight)
        feature_weight = self.conditional_layer2(feature_weight)
        feature_weight = self.relu(feature_weight)
        feature_weight = self.dropout(feature_weight)

        feature_weight = self.softmax(feature_weight)

        #Attention
        out_eeg_small = feature_weight[:,0].unsqueeze(1).unsqueeze(2) * out_eeg_small # 19 X 512
        out_eeg_big = feature_weight[:,1].unsqueeze(1).unsqueeze(2) * out_eeg_big # 19 X 512
        out_eog = feature_weight[:,2].unsqueeze(1).unsqueeze(2) * out_eog # 19 X 512

        # print('weight shape : ',feature_weight.shape)
        # out = torch.cat((out_eeg_big,out_eeg_small,out_eog_big), dim=1)
        # out_eeg_small = torch.flatten(out_eeg_small,1)
        # out_eeg_big   = torch.flatten(out_eeg_big,1)
        # out_eog   = torch.flatten(out_eog,1)
        
        out =  torch.cat((out_eeg_big,out_eeg_small,out_eog), dim=1)
        # print(out.shape)
        # out = out_eeg_small + out_eeg_big + out_eog

        out = self.dropout(out)

        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        
        out = torch.flatten(out,1)
        out = self.dropout(out)

        out = self.classification(out)
        return out

class resnet18_200hz_withDropout_ensemble_branch_twoChannel_attention_new1(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], eeg_channel=[0], eog_channel=[1],
                 layer_filters=[64, 128, 128, 256], first_conv_small=[99, 20, 49],first_conv_big=[399, 20, 199], block_kernel_size=3, padding=1,
                 use_batchnorm=True,embedding_size=512,
                 num_classes=5,sample_rate = 200,epoch_size=30, progress=True):
        super(resnet18_200hz_withDropout_ensemble_branch_twoChannel_attention_new1, self).__init__()
        self.eeg_channel = eeg_channel
        self.eog_channel = eog_channel
        self.eeg_featureExtract_small = ResNet_custom_FE_withoutGAP(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv_small, layer_filters=layer_filters,
                                                                  in_channel=1,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)

        self.eeg_featureExtract_big = ResNet_custom_FE_withoutGAP(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv_big, layer_filters=layer_filters,
                                                                  in_channel=1,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)


        self.eog_featureExtract = ResNet_custom_FE_withoutGAP(block, layers,
                                                                      block_kernel_size=block_kernel_size,
                                                                      padding=padding,
                                                                      first_conv=first_conv_small,
                                                                      layer_filters=layer_filters,
                                                                      in_channel=1,
                                                                      num_classes=num_classes,
                                                                      use_batchnorm=use_batchnorm)

        self.small_sizefilter = math.ceil(math.ceil(math.ceil(math.ceil(math.ceil((sample_rate * epoch_size)/first_conv_small[1])/2)/2)/2)/2)
        self.big_sizefilter = math.ceil(math.ceil(math.ceil(math.ceil(math.ceil((sample_rate * epoch_size)/first_conv_big[1])/2)/2)/2)/2)

        self.gap_eeg_small = nn.AdaptiveAvgPool1d(1) # GAP
        self.gap_eeg_big = nn.AdaptiveAvgPool1d(1) # GAP
        self.gap_eog = nn.AdaptiveAvgPool1d(1) # GAP

        self.conditional_layer1 = nn.Linear(layer_filters[3],layer_filters[3]//2) 
        self.conditional_layer2 = nn.Linear(layer_filters[3]//2,3) 
        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(p=0.5)
        
        self.embedding_size = embedding_size

        self.conv = nn.Conv1d(layer_filters[3]*3, 512, kernel_size=1, stride=1, bias=False)
        
        self.bn = nn.BatchNorm1d(512)

        # self.fc = nn.Linear(self.big_sizefilter*layer_filters[3] + 2*layer_filters[3]*self.small_sizefilter, 1024)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool1d(1)  #
        self.classification = nn.Linear(512, num_classes)
        

        

    def forward(self, x):
        batch_size = x.size(0)
        out_eeg_small = self.eeg_featureExtract_small(x[:, self.eeg_channel, :]) # 38 X 512
        out_eeg_big = self.eeg_featureExtract_big(x[:, self.eeg_channel, :]) # 10 X 512
        out_eog = self.eog_featureExtract(x[:, self.eog_channel, :]) # 10 X 512
        
        feature_weight_eeg_small = self.gap_eeg_small(out_eeg_small) # 1 X 512
        feature_weight_eeg_big = self.gap_eeg_big(out_eeg_big) # 1 X 512
        feature_weight_eog = self.gap_eog(out_eog) # 1 X 512

        feature_weight = feature_weight_eeg_small+feature_weight_eeg_big+feature_weight_eog
        feature_weight = torch.flatten(feature_weight,1)

        feature_weight = self.conditional_layer1(feature_weight)
        feature_weight = self.relu(feature_weight)
        feature_weight = self.dropout(feature_weight)
        feature_weight = self.conditional_layer2(feature_weight)
        feature_weight = self.relu(feature_weight)
        feature_weight = self.dropout(feature_weight)

        feature_weight = self.softmax(feature_weight)

        #Attention
        out_eeg_small = feature_weight[:,0].unsqueeze(1).unsqueeze(2) * out_eeg_small # 19 X 512
        out_eeg_big = feature_weight[:,1].unsqueeze(1).unsqueeze(2) * out_eeg_big # 19 X 512
        out_eog = feature_weight[:,2].unsqueeze(1).unsqueeze(2) * out_eog # 19 X 512

        # print('weight shape : ',feature_weight.shape)
        # out = torch.cat((out_eeg_big,out_eeg_small,out_eog_big), dim=1)
        # out_eeg_small = torch.flatten(out_eeg_small,1)
        # out_eeg_big   = torch.flatten(out_eeg_big,1)
        # out_eog   = torch.flatten(out_eog,1)
        
        out =  torch.cat((out_eeg_big,out_eeg_small,out_eog), dim=1)
        # print(out.shape)
        # out = out_eeg_small + out_eeg_big + out_eog

        out = self.dropout(out)

        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.avgpool(out)
        out = torch.flatten(out,1)
        out = self.dropout(out)

        out = self.classification(out)
        return out

class resnet18_200hz_withDropout_ensemble_branch_twoChannel_attention_new1(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], eeg_channel=[0], eog_channel=[1],
                 layer_filters=[64, 128, 128, 256], first_conv_small=[99, 20, 49],first_conv_big=[399, 20, 199], block_kernel_size=3, padding=1,
                 use_batchnorm=True,embedding_size=512,
                 num_classes=5,sample_rate = 200,epoch_size=30, progress=True):
        super(resnet18_200hz_withDropout_ensemble_branch_twoChannel_attention_new1, self).__init__()
        self.eeg_channel = eeg_channel
        self.eog_channel = eog_channel
        self.eeg_featureExtract_small = ResNet_custom_FE_withoutGAP(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv_small, layer_filters=layer_filters,
                                                                  in_channel=1,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)

        self.eeg_featureExtract_big = ResNet_custom_FE_withoutGAP(block, layers, block_kernel_size=block_kernel_size,
                                                                  padding=padding,
                                                                  first_conv=first_conv_big, layer_filters=layer_filters,
                                                                  in_channel=1,
                                                                  num_classes=num_classes, use_batchnorm=use_batchnorm)


        self.eog_featureExtract = ResNet_custom_FE_withoutGAP(block, layers,
                                                                      block_kernel_size=block_kernel_size,
                                                                      padding=padding,
                                                                      first_conv=first_conv_small,
                                                                      layer_filters=layer_filters,
                                                                      in_channel=1,
                                                                      num_classes=num_classes,
                                                                      use_batchnorm=use_batchnorm)

        self.small_sizefilter = math.ceil(math.ceil(math.ceil(math.ceil(math.ceil((sample_rate * epoch_size)/first_conv_small[1])/2)/2)/2)/2)
        self.big_sizefilter = math.ceil(math.ceil(math.ceil(math.ceil(math.ceil((sample_rate * epoch_size)/first_conv_big[1])/2)/2)/2)/2)

        self.gap_eeg_small = nn.AdaptiveAvgPool1d(1) # GAP
        self.gap_eeg_big = nn.AdaptiveAvgPool1d(1) # GAP
        self.gap_eog = nn.AdaptiveAvgPool1d(1) # GAP

        self.conditional_layer1 = nn.Linear(layer_filters[3],layer_filters[3]//2) 
        self.conditional_layer2 = nn.Linear(layer_filters[3]//2,3) 
        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(p=0.5)
        
        self.embedding_size = embedding_size

        self.conv = nn.Conv1d(layer_filters[3]*3, 512, kernel_size=1, stride=1, bias=False)
        
        self.bn = nn.BatchNorm1d(512)

        # self.fc = nn.Linear(self.big_sizefilter*layer_filters[3] + 2*layer_filters[3]*self.small_sizefilter, 1024)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool1d(1)  #
        self.classification = nn.Linear(512, num_classes)
        

        

    def forward(self, x):
        batch_size = x.size(0)
        out_eeg_small = self.eeg_featureExtract_small(x[:, self.eeg_channel, :]) # 38 X 512
        out_eeg_big = self.eeg_featureExtract_big(x[:, self.eeg_channel, :]) # 10 X 512
        out_eog = self.eog_featureExtract(x[:, self.eog_channel, :]) # 10 X 512
        
        feature_weight_eeg_small = self.gap_eeg_small(out_eeg_small) # 1 X 512
        feature_weight_eeg_big = self.gap_eeg_big(out_eeg_big) # 1 X 512
        feature_weight_eog = self.gap_eog(out_eog) # 1 X 512

        feature_weight = feature_weight_eeg_small+feature_weight_eeg_big+feature_weight_eog
        feature_weight = torch.flatten(feature_weight,1)

        feature_weight = self.conditional_layer1(feature_weight)
        feature_weight = self.relu(feature_weight)
        feature_weight = self.dropout(feature_weight)
        feature_weight = self.conditional_layer2(feature_weight)
        feature_weight = self.relu(feature_weight)
        feature_weight = self.dropout(feature_weight)

        feature_weight = self.softmax(feature_weight)

        #Attention
        out_eeg_small = feature_weight[:,0].unsqueeze(1).unsqueeze(2) * out_eeg_small # 19 X 512
        out_eeg_big = feature_weight[:,1].unsqueeze(1).unsqueeze(2) * out_eeg_big # 19 X 512
        out_eog = feature_weight[:,2].unsqueeze(1).unsqueeze(2) * out_eog # 19 X 512

        # print('weight shape : ',feature_weight.shape)
        # out = torch.cat((out_eeg_big,out_eeg_small,out_eog_big), dim=1)
        # out_eeg_small = torch.flatten(out_eeg_small,1)
        # out_eeg_big   = torch.flatten(out_eeg_big,1)
        # out_eog   = torch.flatten(out_eog,1)
        
        out =  torch.cat((out_eeg_big,out_eeg_small,out_eog), dim=1)
        # print(out.shape)
        # out = out_eeg_small + out_eeg_big + out_eog

        out = self.dropout(out)

        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.avgpool(out)
        out = torch.flatten(out,1)
        out = self.dropout(out)

        out = self.classification(out)
        return out