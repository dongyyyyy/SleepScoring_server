from include.header import *

class ResBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return out

class SEBlock(nn.Module):
    def __init__(self, planes, r=16):
        super(SEBlock, self).__init__()
        self.adap_pool = nn.AdaptiveAvgPool1d(1)
        self.linear1 = nn.Linear(planes*4, int(planes*4//r))
        self.linear2 = nn.Linear(int(planes*4//r), planes*4)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        se = x

        se = self.adap_pool(se) # Global pooling
        se = se.squeeze()
        se = self.linear1(se) # 1 X C -> 1 X C/r
        se = self.relu(se) 
        se = self.linear2(se) # 1 X C/r -> 1 X C
        se = self.sigmoid(se) 

        se = torch.unsqueeze(se, 2) # (batch , channel) -> (batch , channel, 1)
        
        x = x * se # Scale

        return x

class ResSEBlock(nn.Module):
    expansion = 4
    def __init__(self,inplanes, planes,stride=1, downsample=None):
        super(ResSEBlock, self).__init__()
        self.res_b = ResBlock(inplanes, planes, stride) # ResNet bottleNeck Block
        self.downsample = downsample # down sample
        self.se_b = SEBlock(planes) # SE block ( Attention )
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x
        out = self.res_b(x)
        out = self.se_b(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
class ResBlock_withDropout(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(ResBlock_withDropout, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        
        self.dropout = nn.Dropout(p=0.3)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)
        

        return out

class ResSEBlock_withDropout(nn.Module):
    expansion = 4
    def __init__(self,inplanes, planes,stride=1, downsample=None):
        super(ResSEBlock_withDropout, self).__init__()
        self.res_b = ResBlock_withDropout(inplanes, planes, stride) # ResNet bottleNeck Block
        self.downsample = downsample # down sample
        self.se_b = SEBlock(planes) # SE block ( Attention )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        residual = x
        out = self.res_b(x)
        out = self.se_b(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.dropout(out)

        return out

class ResSENet(nn.Module):

    def __init__(self, block, layers=[3,4,6,3], in_channel=1,first_conv=[7,3,3],layers_num=[64,128,256,512], num_classes=5):
        self.inplanes = 64
        super(ResSENet, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, 64, kernel_size=first_conv[0], stride=first_conv[1], padding=first_conv[2],
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(block, layers_num[0], layers[0])
        self.layer2 = self._make_layer(block, layers_num[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, layers_num[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, layers_num[3], layers[3], stride=2)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def ressenet50(block=ResSEBlock,blocks=[3, 4, 6, 3],pretrained=False, **kwargs):
    model = ResSENet(block, blocks, **kwargs)
    return model

class resnet50se_200hz_withoutDropout_ensemble_branch_twoChannel(nn.Module):
    def __init__(self, block=ResSEBlock, layers=[3,4,6,3], eeg_channel=[0], eog_channel=[1],
                 layer_filters=[64, 128, 256, 512], first_conv_small=[99, 20, 49],first_conv_big=[399, 20, 199], block_kernel_size=3, padding=1,
                 use_batchnorm=True,embedding_size=512,
                 num_classes=5,sample_rate = 200,epoch_size=30, progress=True):
        super(resnet50se_200hz_withoutDropout_ensemble_branch_twoChannel, self).__init__()
        self.eeg_channel = eeg_channel
        self.eog_channel = eog_channel
        self.eeg_featureExtract_small = ResSENet(block=block, layers=layers,in_channel=len(eeg_channel), first_conv=first_conv_small,layers_num=layer_filters)

        self.eeg_featureExtract_big = ResSENet(block=block, layers=layers,in_channel=len(eeg_channel), first_conv=first_conv_big,layers_num=layer_filters)


        self.eog_featureExtract = ResSENet(block=block, layers=layers,in_channel=len(eog_channel), first_conv=first_conv_big,layers_num=layer_filters)

        self.small_sizefilter = math.ceil(math.ceil(math.ceil(math.ceil(math.ceil((sample_rate * epoch_size)/first_conv_small[1])/2)/2)/2)/2)
        self.big_sizefilter = math.ceil(math.ceil(math.ceil(math.ceil(math.ceil((sample_rate * epoch_size)/first_conv_big[1])/2)/2)/2)/2)
        
        self.dropout = nn.Dropout(p=0.5)

        self.conv = nn.Conv1d(layer_filters[3]*4*3, 1024, kernel_size=self.big_sizefilter, stride=self.big_sizefilter, bias=False)
        self.bn = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.classification = nn.Linear(1024, num_classes)
    
        

    def forward(self, x):
        
        out_eeg_small = self.eeg_featureExtract_small(x[:, self.eeg_channel, :]) # 38 X 512
        out_eeg_big = self.eeg_featureExtract_big(x[:, self.eeg_channel, :]) # 10 X 512
        out_eog = self.eog_featureExtract(x[:, self.eog_channel, :]) # 10 X 512
        
        
        # channel concatenate
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

class resnet50se_200hz_withDropout_ensemble_branch_twoChannel(nn.Module):
    def __init__(self, block=ResSEBlock_withDropout, layers=[3,4,6,3], eeg_channel=[0], eog_channel=[1],
                 layer_filters=[64, 128, 256, 512], first_conv_small=[99, 20, 49],first_conv_big=[399, 20, 199], block_kernel_size=3, padding=1,
                 use_batchnorm=True,embedding_size=512,
                 num_classes=5,sample_rate = 200,epoch_size=30, progress=True):
        super(resnet50se_200hz_withDropout_ensemble_branch_twoChannel, self).__init__()
        self.eeg_channel = eeg_channel
        self.eog_channel = eog_channel
        self.eeg_featureExtract_small = ResSENet(block=block, layers=layers,in_channel=len(eeg_channel), first_conv=first_conv_small,layers_num=layer_filters)

        self.eeg_featureExtract_big = ResSENet(block=block, layers=layers,in_channel=len(eeg_channel), first_conv=first_conv_big,layers_num=layer_filters)


        self.eog_featureExtract = ResSENet(block=block, layers=layers,in_channel=len(eog_channel), first_conv=first_conv_big,layers_num=layer_filters)

        self.small_sizefilter = math.ceil(math.ceil(math.ceil(math.ceil(math.ceil((sample_rate * epoch_size)/first_conv_small[1])/2)/2)/2)/2)
        self.big_sizefilter = math.ceil(math.ceil(math.ceil(math.ceil(math.ceil((sample_rate * epoch_size)/first_conv_big[1])/2)/2)/2)/2)
        
        self.dropout = nn.Dropout(p=0.5)

        self.conv = nn.Conv1d(layer_filters[3]*4*3, 1024, kernel_size=self.big_sizefilter, stride=self.big_sizefilter, bias=False)
        self.bn = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.classification = nn.Linear(1024, num_classes)
    
        

    def forward(self, x):
        
        out_eeg_small = self.eeg_featureExtract_small(x[:, self.eeg_channel, :]) # 38 X 512
        out_eeg_big = self.eeg_featureExtract_big(x[:, self.eeg_channel, :]) # 10 X 512
        out_eog = self.eog_featureExtract(x[:, self.eog_channel, :]) # 10 X 512
        
        
        # channel concatenate
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