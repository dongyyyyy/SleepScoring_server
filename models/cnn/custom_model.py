from include.header import *

# common 1d convolutional layer
def conv_1d(in_planes, out_planes, kernel_size=3, stride=1, groups=1, padding=1, dilation=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=dilation)


# 3X3 1d convolutional layer ( Basic block & Bottleneck block)
def conv3x3_1d(in_planes, out_planes, kernel_size=3, stride=1, groups=1, padding=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv2x2_1d(in_planes, out_planes):
    return nn.Conv1d(in_planes,out_planes, kernel_size =2, stride =2, bias = False)
# 1Xd 1d convolutional layer ( Bottleneck block )
def conv1x1_1d(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Block_basic(nn.Module):
    def __init__(self, inplanes, planes, stride=1, block_kernel_size=3, downsample=None, 
                 activation_func='relu'):
        super(Block_basic, self).__init__()

        self.conv1 = conv_1d(in_planes=inplanes, out_planes=planes, kernel_size=block_kernel_size, stride=stride,
                             padding=block_kernel_size//2)
        self.bn1 = nn.BatchNorm1d(planes)

        if activation_func == 'relu':
            self.activation_func = nn.ReLU(inplace=True)
        elif activation_func == 'sigmoid':
            self.activation_func = nn.Sigmoid()
        elif activation_func == 'Lrelu':
            self.activation_func = nn.LeakyReLU(inplace=True)
        elif activation_func == 'Prelu':
            self.activation_func = nn.PReLU()

        self.conv2 = conv_1d(in_planes=planes, out_planes=planes, kernel_size=block_kernel_size,padding=block_kernel_size//2)
        self.bn2 = nn.BatchNorm1d(planes)
        self.dropout = nn.Dropout(p=0.2)

        self.downsample = downsample
        

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation_func(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None: # down sample
            identity = self.downsample(x)

        out += identity # skip connection

        out = self.activation_func(out)
        out = self.dropout(out)

        return out

class custom_featureExtract(nn.Module):
    def __init__(self, block=Block_basic,first_conv = [49,20,29],in_channel=1,blocks=[2,2,2,2],block_kernel_size=3,layers_num=[64,128,256,512]
                ,activation_func='relu'):
        super(custom_featureExtract, self).__init__()

        self.first_conv = nn.Conv1d(in_channels=in_channel,out_channels=layers_num[0],
        kernel_size=first_conv[0],stride=1,padding=first_conv[2],bias=False)
        
        self.bn1 = nn.BatchNorm1d(layers_num[0])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)

        self.maxpool1 = nn.MaxPool1d(kernel_size=first_conv[1],stride=first_conv[1])
        
        self.block1 = self._make_layer(block=block,inplanes=layers_num[0],planes=layers_num[0],
                                    blocks=blocks[0],stride=1,block_kernel_size=block_kernel_size,activation_func=activation_func)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2,stride=2)

        self.block2 = self._make_layer(block=block,inplanes=layers_num[0],planes=layers_num[1],
                                    blocks=blocks[0],stride=1,block_kernel_size=block_kernel_size,activation_func=activation_func)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2,stride=2)

        self.block3 = self._make_layer(block=block,inplanes=layers_num[1],planes=layers_num[2],
                                    blocks=blocks[0],stride=1,block_kernel_size=block_kernel_size,activation_func=activation_func)
        self.maxpool4 = nn.MaxPool1d(kernel_size=2,stride=2)

        self.block4 = self._make_layer(block=block,inplanes=layers_num[2],planes=layers_num[3],
                                    blocks=blocks[0],stride=1,block_kernel_size=block_kernel_size,activation_func=activation_func)
        self.maxpool5 = nn.MaxPool1d(kernel_size=2,stride=2)

    def _make_layer(self, block, inplanes,planes, blocks, stride=1,block_kernel_size=3,activation_func='relu'):
        downsample = None
        if inplanes != planes and stride ==1: # stride =1 but channel is different !
            downsample = nn.Sequential(
                conv1x1_1d(inplanes, planes, stride),
                nn.BatchNorm1d(planes),
            )
        elif stride != 1 and inplanes != planes: # stride and input channel & output channel is different !
            downsample = nn.Sequential(
                conv2x2_1d(inplace,planes,stride), # original is 1x1 conv but change to 2x2 (using more input info)
                nn.BatchNorm1d(planes),
            ) 

        layers = []

        layers.append(
            block(inplanes, planes, block_kernel_size=block_kernel_size, stride=stride,downsample=downsample,activation_func=activation_func))
        
        for _ in range(1, blocks):
            layers.append(block(planes, planes, block_kernel_size=block_kernel_size,stride=1,downsample=False,activation_func=activation_func))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.first_conv(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.maxpool1(out) # 300
        

        out = self.block1(out)
        out = self.maxpool2(out) # 150
        

        out = self.block2(out)
        out = self.maxpool3(out) # 75
        

        out = self.block3(out)
        out = self.maxpool4(out) # 37
        

        out = self.block4(out)
        out = self.maxpool5(out) # 18

        
        return out

class custom_model1(nn.Module):
    def __init__(self, block=Block_basic,num_classes=5,eeg_channel=[0],eog_channel=[1],first_conv_small = [99,20,49],first_conv_big=[199,20,99],
                        in_channel=1,blocks=[1,1,1,1],block_kernel_size=3,layers_num=[64,64,128,128],activation_func='relu',sample_rate=200,epoch_size=30):
        super(custom_model1, self).__init__()

        self.eeg_channel = eeg_channel
        self.eog_channel = eog_channel

        self.featureExtract_eeg_small = custom_featureExtract(block=block,first_conv = first_conv_small,
            in_channel=1,blocks=blocks,block_kernel_size=block_kernel_size,layers_num=layers_num
            ,activation_func=activation_func)

        self.featureExtract_eeg_big = custom_featureExtract(block=block,first_conv = first_conv_big,
            in_channel=1,blocks=blocks,block_kernel_size=block_kernel_size,layers_num=layers_num
            ,activation_func=activation_func)

        self.featureExtract_eog = custom_featureExtract(block=block,first_conv = first_conv_big,
            in_channel=1,blocks=blocks,block_kernel_size=block_kernel_size,layers_num=layers_num
            ,activation_func=activation_func)

        # self.small_sizefilter = math.ceil(math.ceil(math.ceil(math.ceil(math.ceil((sample_rate * epoch_size)/first_conv_small[1])/2)/2)/2)/2)
        # self.big_sizefilter = math.ceil(math.ceil(math.ceil(math.ceil(math.ceil((sample_rate * epoch_size)/first_conv_big[1])/2)/2)/2)/2)
        self.small_sizefilter = 18
        self.big_sizefilter = 18

        # using Attention
        self.gap_eeg_small = nn.AdaptiveAvgPool1d(1) # GAP
        self.gap_eeg_big = nn.AdaptiveAvgPool1d(1) # GAP
        self.gap_eog = nn.AdaptiveAvgPool1d(1) # GAP

        self.conditional_layer1 = nn.Linear(layers_num[3],layers_num[3]//2) 
        self.conditional_layer2 = nn.Linear(layers_num[3]//2,3) 
        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(p=0.5)
        
        self.conv = nn.Conv1d(layers_num[3]*3, 512, kernel_size=self.big_sizefilter, stride=self.big_sizefilter, bias=False)
        self.bn = nn.BatchNorm1d(512)
        self.relu = nn.ReLU(inplace=True)

        self.classification = nn.Linear(512, num_classes)

    def forward(self, x):
        out_eeg_small = self.featureExtract_eeg_small(x[:, self.eeg_channel, :]) # 38 X 512
        out_eeg_big = self.featureExtract_eeg_big(x[:, self.eeg_channel, :]) # 10 X 512
        out_eog = self.featureExtract_eog(x[:, self.eog_channel, :]) # 10 X 512
        
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
        
        out =  torch.cat((out_eeg_big,out_eeg_small,out_eog), dim=1)


        out = self.dropout(out)

        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        
        out = torch.flatten(out,1)
        out = self.dropout(out)

        out = self.classification(out)
        return out
