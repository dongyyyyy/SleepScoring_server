from include.header import *
from utils.makeDataset.select_channel import *
from utils.makeDataset.makeDataset_dataloader import *
from utils.function.dataloader_custom import *
from train.transformer.train_transformer import *
from utils.function.MakeFeatureExtract import *
from include.header import *
from train.cnn.train_cnn import *
from train.transformer.train_transformer import *
from train.rnn.train_rnn import *
from train.rnn.train_rnn_cuda1 import *
from utils.makeDataset.SeoulUniv.make_edf_to_npy import *
from utils.makeDataset.SeoulUniv.make_dataloader import *
from utils.makeDataset.SeoulUniv.check_dataset_info import *
use_cudaNum = 1

if __name__=='__main__':
    make_dataloader_dataset()
    # make_edf_to_npy_usingmne()
    # check_severity()
    # make_selectChannel_npy()
    # makeFeatureExtract_savefile()
    # make_edf_to_npy_usingmne()
    # signals_path = '/home/eslab/dataset/Seoul_dataset/3channel_dataloader/A2019-NX-01-0254_3_/'
    # data_list = os.listdir(signals_path)
    # print(data_list[0])

    # signals = np.load(signals_path + data_list[0])

    # print(signals.shape)
    # plt.plot(signals[0,:])
    # plt.savefig('./fig')
    # plt.cla()
    # make_edf_to_npy_usingmne()
    # make_dataloader_dataset()
    # make_selectChannel_npy()
    # check_severity()
    # make_seoul_dataloader_data()
    # check_dataset_truth()
    # makeFeatureExtract_savefile()

    # training_cnn_dataloader()
    # training_detr_dataloader()

    # training_detr_dataloader()
    # signals_path = '/home/ssd1/dataset/Hallym_npy/3channel_info_shuffle/train_dataloader/'
    
    # #test_signal_dir = '/home/jglee/medical_dataset/Seoul_medicalDataset_npy/C3M2/EEG/test_each_preprocessing/'
    # k_fold = 10

    # dataset_list = os.listdir(signals_path)
    # training_fold_list = []
    # validation_fold_list = []
    # for i in range(0,1):
    #     for folder_index, folder_name in enumerate(dataset_list):
    #         if folder_index % k_fold != 0 or folder_index == 0:
    #             training_fold_list.append(folder_name)
    #         else:
    #             validation_fold_list.append(folder_name)
    # print(len(training_fold_list))
    # print(len(validation_fold_list))



    # s_t = time.time()
    # for index, data in enumerate(train_dataloader):
    #     signals,label = data
    #     signals = signals.reshape(-1,3,6000)
    #     print('index : ',index,signals.shape)
    #     if index == 100:
    #         break


    # for index, data in enumerate(val_dataloader):
    #     signals,label = data
    #     signals = signals.reshape(-1,3,6000)
    #     print('index : ',index,signals.shape)