from include.header import *
from utils.function.function import *
def func_make_dataloader_dataset(filename):
    signals_path = '/data/hdd1/dataset/Seoul_dataset/9channel_prefilter_butter/signals/'
    annotations_path = '/data/hdd1/dataset/Seoul_dataset/annotations/'
    # annotations_path = '/home/ssd1/dataset/Seoul_dataset/9channel_prefilter/annotations/'
    save_path = '/data/ssd1/dataset/Seoul_dataset/9channel_prefilter_butter/signals_dataloader/'
    # print(file_list)
    # for filename in file_list:
    current_save_path = save_path + filename.split('.npy')[0]+'/'
    os.makedirs(current_save_path,exist_ok=True)
    # print(current_save_path)
    
    signals = np.load(signals_path+filename)
    if signals.shape[1] == 0:
        return
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
            # print(save_filename, annotations[index])
            # exit(1)
            np.save(save_filename,save_signals)
        print(f'finish : {current_save_path}')
        # exit(1)

def make_dataloader_dataset():
    signals_path = '/data/hdd1/dataset/Seoul_dataset/9channel_prefilter_butter/signals/'
    annotations_path = '/data/hdd1/dataset/Seoul_dataset/annotations/'
    # annotations_path = '/home/ssd1/dataset/Seoul_dataset/9channel_prefilter/annotations/'
    save_path = '/data/ssd1/dataset/Seoul_dataset/9channel_prefilter_butter/signals_dataloader/'
    
    os.makedirs(save_path,exist_ok=True)

    file_list = os.listdir(signals_path)
    # print(file_list)
    cpu_num = multiprocessing.cpu_count()
    print('cpu_num : ',cpu_num)
    
    
    start = time.time()
    pool = Pool(cpu_num)

    pool.map(func_make_dataloader_dataset,file_list)
    pool.close()
    pool.join()





select_channel = [0,1,2,3]

def make_np_selectChannel(filename):
    save_path = '/home/eslab/dataset/seoulDataset/4channel_prefilter/signals_dataloader/' + filename.split('/')[-2]+'/'
    print(save_path)
    # print(filename)
    # save_path = filename.split('/')
    # save_path[-4] = '3channel_prefilter'
    # save_path = '/'.join(save_path)
    # print(save_path)
    os.makedirs(save_path,exist_ok=True)

    file_list = os.listdir(filename)

    for file_name in file_list:
        signals = np.load(filename+file_name)
        signals = signals[select_channel]
        
        np.save(save_path+file_name,signals)
        # exit(1)


def make_selectChannel_npy():
    cpu_num = multiprocessing.cpu_count()


    path = '/home/eslab/dataset/seoulDataset/9channel_prefilter/signals_dataloader/'
    file_list = os.listdir(path)
    filepath_list = [ path+f+'/' for f in file_list]

    # make_np_selectChannel(filepath_list[0])
    # exit(1)
    pool = Pool(cpu_num)
    pool.map(make_np_selectChannel,filepath_list)
    pool.close()
    pool.join()

