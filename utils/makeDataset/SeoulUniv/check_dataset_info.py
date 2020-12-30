from include.header import *

def check_severity():
    signals_path = '/home/eslab/dataset/seoulDataset/4channel_prefilter/signals_dataloader/'
    file_list = os.listdir(signals_path)

    check_severitys = np.array((0,0,0,0))

    for filename in file_list:
        severity = int(filename.split('_')[1])
        check_severitys[severity] += 1

    print(check_severitys)