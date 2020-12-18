from include.header import *
from collections import OrderedDict

def makeFeatureExtract_savefile():
    #model = resnet18(in_channel=3,use_dropout=True)

    load_path = '/home/eslab/kdy/git/Hallym_SleepStage/saved_model/Hallym_dataset/cnn/'
    learning_rate = 0.001
    optim='Adam'
    loss_function='CE'
    scheduler= 'WarmUp_restart_gamma'
    load_file_name = 'ResNet_ensemble_branch_dataloaer_hallym_%.4f_%s_%s_%s.txt'%(learning_rate,optim,loss_function,scheduler)
    save_file_name = 'ResNet_ensemble_branch_dataloaer_hallym_%.4f_%s_%s_%s_FE.pth'%(learning_rate,optim,loss_function,scheduler)
    load_file = load_path + load_file_name
    state_dict = torch.load(load_file)

    new_state_dict = OrderedDict()
    for key,value in state_dict.items():
        if(key[7:] != 'fc.bias' and key[7:] != 'fc.weight'):
            print('key : ',key)
            #key = key[15:]
            # print('key : ', key[15:])
            new_state_dict[key[7:]] = value
    print(new_state_dict.keys())
    save_file = load_path+save_file_name
    torch.save(new_state_dict, save_file)

# makeFeatureExtract_savefile()