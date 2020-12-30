from include.header import *
from collections import OrderedDict

def makeFeatureExtract_savefile():
    #model = resnet18(in_channel=3,use_dropout=True)

    load_path = '/home/eslab/kdy/git/SleepScoring_server/saved_model/Hallym_dataset/seoul/cnn/'
    
    load_file_name = 'ResNet_ensemble_branch_dataloaer_seoul_3channel_0.0100_AdamW_CE_WarmUp_restart_gamma.pth'
    save_file_name = 'ResNet_ensemble_branch_dataloaer_seoul_3channel_0.0100_AdamW_CE_WarmUp_restart_gamma_FE.pth'
    load_file = load_path + load_file_name
    state_dict = torch.load(load_file)

    new_state_dict = OrderedDict()
    for key,value in state_dict.items():
        if(key[7:] != 'classification.bias' and key[7:] != 'classification.weight' and key[7:] != 'fc.bias' and key[7:] != 'fc.weight'):
            print('key : ',key)
            #key = key[15:]
            # print('key : ', key[15:])
            new_state_dict[key[7:]] = value
    print(new_state_dict.keys())
    save_file = load_path+save_file_name
    torch.save(new_state_dict, save_file)

# makeFeatureExtract_savefile()