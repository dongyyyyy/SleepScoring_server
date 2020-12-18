from include.header import *

def func_select_channel(filelist,signals_path,annotations_path,save_path,select_channel):
    # print(filelist)
    for filename in filelist:
        signals = np.load(signals_path+filename)
        signals = signals[select_channel]
        annotations = np.load(annotations_path+filename)
        if len(annotations) == len(signals[0])/6000:
            print(f'filename : {filename} shape : {signals.shape}')
            np.save(save_path+filename,signals)
        else:
            print(f'Fault filename : {filename} shape : {signals.shape}')

def func_check_channel(filelist,signals_path,annotations_path):
    for filename in filelist:
        signals = np.load(signals_path+filename)
        annotations = np.load(annotations_path+filename)
        if len(annotations) == len(signals[0])/6000:
            print(f'filename : {filename} shape : {signals.shape}')
        else:
            print(f'filename : {filename} is fault file!')


def make_select_channel():
    # cpu_num = multiprocessing.cpu_count()
    # lock = Lock()
    select_channel = [0,1,2,3,4]
    signals_path = 'D:/dataset/Hallym_npy/signals_info/'
    save_path = 'D:/dataset/Hallym_npy/5channel_info/'
    annotations_path = 'D:/dataset/Hallym_npy/annotations_info/'
    os.makedirs(save_path,exist_ok=True)
    file_list = os.listdir(signals_path)

    # devide_num = len(file_list) // cpu_num

    func_select_channel(file_list,signals_path,annotations_path,save_path,select_channel)

def make_select_channel_new(signals_path,save_path,annotations_path):
    cpu_num = multiprocessing.cpu_count()
    lock = Lock()
    select_channel = [1, 3, 4]
    os.makedirs(save_path, exist_ok=True)
    file_list = os.listdir(signals_path)

    devide_num = len(file_list) // cpu_num

    func_select_channel(file_list, signals_path, annotations_path, save_path, select_channel)
    # dif_num = len(file_list) - devide_num * cpu_num
    
    # start_num = 0
    # for num in range(cpu_num):
    #     if num < dif_num:
    #         Process(target=func_select_channel, args=(file_list[(start_num):(start_num+devide_num+1)],signals_path,save_path,select_channel)).start()
    #         start_num += (devide_num + 1)
    #     else:
    #         Process(target=func_select_channel, args=(file_list[(start_num):(start_num+devide_num)],signals_path,save_path,select_channel)).start()
    #         start_num += devide_num

def check_select_channel(signals_path,annotations_path):
    cpu_num = multiprocessing.cpu_count()
    lock = Lock()
    file_list = os.listdir(signals_path)

    devide_num = len(file_list) // cpu_num
    dif_num = len(file_list) - devide_num * cpu_num
    
    start_num = 0
    for num in range(cpu_num):
        if num < dif_num:
            Process(target=func_check_channel, args=(file_list[(start_num):(start_num+devide_num+1)],signals_path,annotations_path)).start()
            start_num += (devide_num + 1)
        else:
            Process(target=func_check_channel, args=(file_list[(start_num):(start_num+devide_num)],signals_path,annotations_path)).start()
            start_num += devide_num




def func_select_channel_plot(select_channel=[0]):
    # print(filelist)
    signals_path = 'C:/dataset/Hallym_dataset/5channel_info_shuffle/test/'
    signals_list = os.listdir(signals_path)
    signals = np.load(signals_path+signals_list[0])
    signals = signals[select_channel]

    plt.plot(signals[0,:6000])
    plt.show()



# make_select_channel_new()
