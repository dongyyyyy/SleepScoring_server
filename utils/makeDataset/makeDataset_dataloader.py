from include.header import *
from utils.function.function import *

def func_make_dataloader_dataset(file_list,signals_path,annotations_path,save_path):
    for filename in file_list:
        current_save_path = save_path + filename.split('.npy')[0]+'/'
        os.makedirs(current_save_path,exist_ok=True)
        print(current_save_path)
        signals = np.load(signals_path+filename)
        signals = data_preprocessing_numpy(signals)
        annotations = np.load(annotations_path+filename)
        width = 200 * 30
        signals_len = len(signals[0])// width
        if signals_len == len(annotations):
            for index in range(signals_len):
                save_signals = signals[:,index*width : (index+1)*width]
                if index < 10:
                    save_index = '000%d'%(index)
                elif index < 100:
                    save_index = '00%d'%index
                elif index < 1000:
                    save_index = '0%d'%index
                else:
                    save_index = '%d'%index
                save_filename = current_save_path+'%s_%d.npy'%(save_index,annotations[index])
                print(save_filename, annotations[index])
                # exit(1)
                np.save(save_filename,save_signals)

def make_dataloader_dataset(signals_path,save_path,annotations_path):

    os.makedirs(save_path, exist_ok=True)
    file_list = os.listdir(signals_path)

    func_make_dataloader_dataset(file_list, signals_path, annotations_path, save_path)

def make_dataloader_data():
    signals_path = '/home/ssd1/dataset/Hallym_npy/3channel_info_shuffle/train/'
    save_path = '/home/ssd1/dataset/Hallym_npy/3channel_info_shuffle/train_dataloader/'
    annotations_path = '/home/ssd1/dataset/Hallym_npy/annotations_info/'
    make_dataloader_dataset(signals_path,save_path,annotations_path)
    signals_path = '/home/ssd1/dataset/Hallym_npy/3channel_info_shuffle/test/'
    save_path = '/home/ssd1/dataset/Hallym_npy/3channel_info_shuffle/test_dataloader/'
    annotations_path = '/home/ssd1/dataset/Hallym_npy/annotations_info/'
    make_dataloader_dataset(signals_path,save_path,annotations_path)