from include.header import *

class DeepSleepNet_block(nn.Module):  # 2D 없이 1D로 통일하기
    def __init__(self,in_channel,conv1,conv2,mp1,mp2):
        super(DeepSleepNet_block, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channel,out_channels=conv1[3],kernel_size=conv1[0],stride=conv1[1],padding=conv1[2],
                               bias=False)

        self.conv2 = nn.Conv1d(in_channels=conv1[3],out_channels=conv2[3],kernel_size=conv2[0],stride=conv2[1],padding=conv2[2],
                               bias=False)

        self.conv3 = nn.Conv1d(in_channels=conv2[3], out_channels=conv2[3], kernel_size=conv2[0], stride=conv2[1],
                               padding=conv2[2],
                               bias=False)
        self.conv4 = nn.Conv1d(in_channels=conv2[3], out_channels=conv2[3], kernel_size=conv2[0], stride=conv2[1],
                               padding=conv2[2],
                               bias=False)

        self.maxpool1 = nn.MaxPool1d(kernel_size=mp1[0],stride=mp1[1],padding=mp1[2])
        self.maxpool2 = nn.MaxPool1d(kernel_size=mp2[0], stride=mp2[1], padding=mp2[2])

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool1(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.relu(out)

        out = self.maxpool2(out)

        return out



class DeepSleepNet_FE(nn.Module):  # 2D 없이 1D로 통일하기
    def __init__(self,in_channel=1):
        super(DeepSleepNet_FE, self).__init__()
        self.bigCNN = DeepSleepNet_block(in_channel,conv1=[800,100,400,64],conv2=[6,1,3,128],mp1=[4,4,2],mp2=[2,2,1])
        self.smallCNN = DeepSleepNet_block(in_channel,conv1=[100, 12, 50, 64], conv2=[8, 1, 4, 128], mp1=[8, 8, 4], mp2=[4, 4, 2])

    def forward(self,x):
        feature_big = self.bigCNN(x)
        feature_small = self.smallCNN(x)

        feature_big = torch.flatten(feature_big, 1)
        feature_small = torch.flatten(feature_small, 1)

        out = torch.cat((feature_big,feature_small),dim=1)

        return out



class DeepSleepNet_classification(nn.Module):  # 2D 없이 1D로 통일하기
    def __init__(self,in_channel=1,class_num=5):
        super(DeepSleepNet_classification, self).__init__()
        self.feature_extract = DeepSleepNet_FE(in_channel=in_channel)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(3456,class_num)

    def forward(self,x):
        out = self.feature_extract(x)
        out = self.dropout(out)
        out = self.fc(out)

        return out

class DeepSleepNet_multiChannel_FE(nn.Module):
    def __init__(self):
        super(DeepSleepNet_multiChannel_FE,self).__init__()
        self.FeatireExtract_eeg = DeepSleepNet_FE(in_channel=1)
        self.FeatureExtract_emg = DeepSleepNet_FE(in_channel=1)
        self.FeatureExtract_eog = DeepSleepNet_FE(in_channel=2)

    def forward(self,channel0,channel1,channel2):
        out_channel0 = self.FeatireExtract_eeg(channel0)
        out_channel1 = self.FeatureExtract_emg(channel1)
        out_channel2 = self.FeatureExtract_eog(channel2)
        out = torch.cat((out_channel0,out_channel1,out_channel2),dim=1)

        return out


class DeepSleepNet_eeg_eog_FE(nn.Module):
    def __init__(self,eeg_channel=1,eog_channel=2):
        super(DeepSleepNet_eeg_eog_FE,self).__init__()
        self.FeatireExtract_eeg = DeepSleepNet_FE(in_channel=eeg_channel)
        self.FeatureExtract_eog = DeepSleepNet_FE(in_channel=eog_channel)

    def forward(self,channel0,channel1):
        out_channel0 = self.FeatireExtract_eeg(channel0)
        out_channel1 = self.FeatureExtract_eog(channel1)
        out = torch.cat((out_channel0,out_channel1),dim=1)

        return out

class DeepSleepNet_multiChannel_Classification(nn.Module):
    def __init__(self,class_num=5):
        super(DeepSleepNet_multiChannel_Classification, self).__init__()
        self.FeatureExtract = DeepSleepNet_multiChannel_FE()
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(3456*3, class_num)

    def forward(self, x):
        out = self.FeatureExtract(x[:,0,:].unsqueeze(1),x[:,1,:].unsqueeze(1),x[:,[2,3],:])
        out = self.dropout(out)
        out = self.fc(out)
        return out

class DeepSleepNet_eeg_eog_Classification(nn.Module):
    def __init__(self,class_num=5):
        super(DeepSleepNet_eeg_eog_Classification, self).__init__()
        self.FeatureExtract = DeepSleepNet_eeg_eog_FE()
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(3456*2, class_num)

    def forward(self, x):
        out = self.FeatureExtract(x[:,0,:].unsqueeze(1),x[:,[1,2],:])
        out = self.dropout(out)
        out = self.fc(out)

        return out


class featureExtract_pretrained(nn.Module):
    def __init__(self, in_channel=1, hidden_dim=512, class_num=5):
        super(featureExtract_pretrained, self).__init__()
        self.featureExtract = DeepSleepNet_eeg_eog_FE()
        self.dropout = nn.Dropout(p=0.5)
        self.flat = 3456 * 2

    def forward(self, input):
        out = self.featureExtract(input[:, 0, :].unsqueeze(1), input[:, [1, 2], :])

        out = out.unsqueeze(0)

        return out

class featureExtract_oneModel_pretrained(nn.Module):
    def __init__(self, in_channel=1, hidden_dim=512, class_num=5):
        super(featureExtract_oneModel_pretrained, self).__init__()
        self.featureExtract = DeepSleepNet_FE(in_channel=in_channel)
        self.dropout = nn.Dropout(p=0.5)
        self.flat = 3456

    def forward(self, input):
        out = self.featureExtract(input)

        out = out.unsqueeze(0)

        return out
