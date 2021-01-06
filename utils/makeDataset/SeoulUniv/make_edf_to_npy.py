from include.header import *
from utils.function.function import *

def check_dataset_thread(filename):
    signals_save_path = '/mnt/ssd1/dataset/Seoul_dataset/9channel_prefilter/signals/'
    annotations_save_path = '/mnt/ssd1/dataset/Seoul_dataset/9channel_prefilter/annotations/'

    signals = np.load(signals_save_path+filename)
    annotations = np.load(annotations_save_path+filename)

    if len(signals[0])// 200 // 30 != len(annotations):
        print(f'filename : {filename} is fault!!!')

    # signals = torch.from_numpy(signals).float()
    # annotations = torch.from_numpy(annotations).long()
    #print(signals.shape)


def my_thread(file_list):
    signals_path = file_list[0]
    annotations_path = file_list[1]
    
    signals_mne_save_path = '/data/hdd1/dataset/Seoul_dataset/9channel_prefilter_mne/signals/'
    # signals_sos_save_path = '/mnt/hdd1/dataset/Seoul_dataset/9channel_prefilter_sos/signals/'
    signals_butter_save_path = '/data/hdd1/dataset/Seoul_dataset/9channel_prefilter_butter/signals/'
    signals_ellip_save_path = '/data/hdd1/dataset/Seoul_dataset/9channel_prefilter_ellip/signals/'
    annotations_save_path = '/data/hdd1/dataset/Seoul_dataset/annotations/'

    save_filename = '%s%s'%(signals_path.split('/')[-1].split('_')[0] , '.npy')
    # file_list = os.listdir(signals_save_path)

    # if save_filename in file_list:
    #     print('This file is exist!')
    # else:
    print(signals_path)
    print(annotations_path)
    select_channel=['F3-M2','F4-M1','C3-M2','C4-M1','O1-M2','O2-M1','E1-M2','E2-M1','1-2']
    eeg_channel = ['C3-M2', 'C4-M1', 'F4-M1', 'F3-M2', 'O2-M1', 'O1-M2']
    eog_channel = ['E1-M2', 'E2-M1']
    emg_channel = ['1-2']
    eeg_lowcut = 0.5
    eeg_highcut = 35
    eog_lowcut = 0.3
    eog_highcut = 35
    emg_lowcut = 10
    annotations = pd.read_csv(annotations_path)

    # mne로 사용하여 시작 시간을 찾을 경우 정상적이지 못해 highlevel을 활용
    # info = highlevel.read_edf_header(signals_path)

    # pyedflib를 활용하여 edf 데이터 읽기
    signals_pyedf, signals_info_pyedf, info = highlevel.read_edf(signals_path)
    # mne library를 활용한 데이터 읽기
    signals = mne.io.read_raw_edf(signals_path, preload=True)
    # 필요없는 line 제거
    annotations = annotations.dropna(axis=1)
    annotations = annotations.values.tolist()
    annotations = annotations[1:]

    # numpy 형태로 저장할 list
    apnea_duration = 0
    # 첫번째 sleep stage 위치를 판단하기 위한 변수
    first = 0
    sleep_start = 0
    start_epoch = 0
    end_epoch = 0
    sleep_stage = ['Wake','N1','N2','N3','REM']

    for index in range(0,len(annotations),1):
        if annotations[index][0] in sleep_stage:
            start_epoch = int(annotations[index][4])
            break

    for index in range(len(annotations)-1,-1,-1):
        if annotations[index][0] in sleep_stage:
            end_epoch = int(annotations[index][4])
            break
    if start_epoch == 0 and end_epoch == 0 : # false labeling!
        return
    
    # print(start_epoch,end_epoch)
    annotations_np = np.zeros(end_epoch-start_epoch+1)
    # print(annotations_np.shape)
    # sleep stage 판단 후 사용할 stage numpy 형태로 저장히기 위해 list에 추가
    for annotations_info in annotations:
        if (annotations_info[0] == 'Wake'):
            if first == 0:
                start_time = annotations_info[2]
                first += 1
                # print(int(annotations_info[4])-start_epoch)
            if sleep_start != 0:
                sleep_start += 1
            annotations_np[int(annotations_info[4])-start_epoch] = 0
        elif (annotations_info[0] == 'N1'):
            if first == 0:
                start_time = annotations_info[2]
                first += 1
            
            sleep_start += 1
            annotations_np[int(annotations_info[4])-start_epoch] = 1
        elif (annotations_info[0] == 'N2'):
            if first == 0:
                start_time = annotations_info[2]
                first += 1

            sleep_start += 1
            annotations_np[int(annotations_info[4])-start_epoch] = 2
        elif (annotations_info[0] == 'N3'):
            if first == 0:
                start_time = annotations_info[2]
                first += 1
            
            sleep_start += 1
            annotations_np[int(annotations_info[4])-start_epoch] = 3
        elif (annotations_info[0] == 'REM'):
            if first == 0:
                start_time = annotations_info[2]
                first += 1
            
            sleep_start += 1
            annotations_np[int(annotations_info[4])-start_epoch] = 4
        elif (annotations_info[0] == 'Hypopnea'):
            if sleep_start != 0:
                apnea_duration += 1
        elif (annotations_info[0] == 'A. Obstructive'):
            if sleep_start != 0:
                apnea_duration += 1
        elif (annotations_info[0] == 'A. Mixed'):
            if sleep_start != 0:
                apnea_duration += 1
        elif (annotations_info[0] == 'A. Central'):
            if sleep_start != 0:
                apnea_duration += 1
    if len(annotations_np) < 10:
        print('%s file label is too small' % annotations_path)
    else:
        annotations_np = np.array(annotations_np)
        ahi_index = apnea_duration / (sleep_start*30)*3600 # AHI = (Apnea + Hypopnea) / sleep time * 100
        print(start_epoch,end_epoch)
        
        if ahi_index < 5:
            severity = 0
        elif ahi_index < 15:
            severity = 1
        elif ahi_index < 30:
            severity = 2
        else:
            severity = 3

        # signals 시작 시간
        signals_start_time = info['startdate']

        print('start_time : ', start_time)
        annotations_split = start_time.split(' ')

        # annotations의 시작 시간의 형태가 signals와 다르기 때문에 일치시키기 위한 작업
        if (annotations_split[-1] == 'PM'):
            annotations_split[-2] = '%s:%s:%s' % (
                str(int(annotations_split[-2].split(':')[0]) + 12), annotations_split[-2].split(':')[1],
                annotations_split[-2].split(':')[2])
            if annotations_split[-2] == '24:00:00':
                annotations_split[0] = '%s/%s/%s'%(str(annotations_split[0].split('/')[0]), int(annotations_split[0].split('/')[1])+1,
                annotations_split[0].split('/')[2])
                annotations_split[-2] = '00:00:00'

        start_time = ' '.join(annotations_split[:-1])
        print('start_time : ', start_time)
        
        annotations_start_time = datetime.datetime.strptime(start_time, '%m/%d/%Y %H:%M:%S')

        print(signals_start_time)
        print(annotations_start_time)

        dif_sec = annotations_start_time - signals_start_time  # annotations 시작 시간 - signals 시작 시간

        print(dif_sec)
        
        dif_sec = int(str(dif_sec).split(':')[0]) * 3600 + int(str(dif_sec).split(':')[1]) * 60 + int(
            str(dif_sec).split(':')[2])

        # check channel
        count = 0
        order = 4
        rp = 5 # berndporr default
        rs = 60
        # signals_pyedf

        for index, signals_info in enumerate(signals_info_pyedf):
            if signals_info['label'] in select_channel and count == 0:
                print('select channel : ', signals_info['label'])
                if signals_info['label'] in eeg_channel:
                    new_signals_pyedf = butter_bandpass_filter(signals=signals_pyedf[index].reshape(1,-1),lowcut=eeg_lowcut,highcut=eeg_highcut,fs=200,order=order)
                    # new_signals_pyedf_sos = butter_filter_sos(signals=signals_pyedf[index].reshape(1,-1), lowcut=eeg_lowcut, highcut=eeg_highcut, fs=200 , order =order)
                    new_signals_pyedf_ellip = ellip_filter_sos(signals=signals_pyedf[index].reshape(1,-1),rp=rp,rs=rs, lowcut=eeg_lowcut, highcut=eeg_highcut, fs = 200 , order = order)
                elif signals_info['label'] in eog_channel:
                    new_signals_pyedf = butter_bandpass_filter(signals=signals_pyedf[index].reshape(1,-1),lowcut=eog_lowcut,highcut=eog_highcut,fs=200,order=order)
                    # new_signals_pyedf_sos = butter_filter_sos(signals=signals_pyedf[index].reshape(1,-1), lowcut=eog_lowcut, highcut=eog_highcut, fs=200 , order =order)
                    new_signals_pyedf_ellip = ellip_filter_sos(signals=signals_pyedf[index].reshape(1,-1),rp=rp,rs=rs ,lowcut=eog_lowcut, highcut=eog_highcut, fs = 200 , order = order)
                elif signals_info['label'] in emg_channel:
                    new_signals_pyedf = butter_highpass_filter(data=signals_pyedf[index].reshape(1,-1), cutoff=emg_lowcut, order=order,fs=200)
                    # new_signals_pyedf_sos = butter_filter_sos(signals=signals_pyedf[index].reshape(1,-1), lowcut=emg_lowcut, highcut=None, fs=200 , order =order)
                    new_signals_pyedf_ellip = ellip_filter_sos(signals=signals_pyedf[index].reshape(1,-1),rp=rp,rs=rs, lowcut=emg_lowcut, highcut=None, fs = 200 , order = order)
                count += 1
            elif signals_info['label'] in select_channel and count != 0:
                if signals_info['label'] in eeg_channel:
                    new_signals_pyedf_behind = butter_bandpass_filter(signals=signals_pyedf[index].reshape(1,-1),lowcut=eeg_lowcut,highcut=eeg_highcut,fs=200,order=order)
                    # new_signals_pyedf_behind_sos = butter_filter_sos(signals=signals_pyedf[index].reshape(1,-1), lowcut=eeg_lowcut, highcut=eeg_highcut, fs=200 , order =order)
                    new_signals_pyedf_behind_ellip = ellip_filter_sos(signals=signals_pyedf[index].reshape(1,-1),rp=rp,rs=rs, lowcut=eeg_lowcut, highcut=eeg_highcut, fs = 200 , order = order)
                elif signals_info['label'] in eog_channel:
                    new_signals_pyedf_behind = butter_bandpass_filter(signals=signals_pyedf[index].reshape(1,-1),lowcut=eog_lowcut,highcut=eog_highcut,fs=200,order=order)
                    # new_signals_pyedf_behind_sos = butter_filter_sos(signals=signals_pyedf[index].reshape(1,-1), lowcut=eog_lowcut, highcut=eog_highcut, fs=200 , order =order)
                    new_signals_pyedf_behind_ellip = ellip_filter_sos(signals=signals_pyedf[index].reshape(1,-1),rp=rp,rs=rs ,lowcut=eog_lowcut, highcut=eog_highcut, fs = 200 , order = order)
                elif signals_info['label'] in emg_channel:
                    new_signals_pyedf_behind = butter_highpass_filter(data=signals_pyedf[index].reshape(1,-1), cutoff=emg_lowcut, order=order,fs=200)
                    # new_signals_pyedf_behind_sos = butter_filter_sos(signals=signals_pyedf[index].reshape(1,-1), lowcut=emg_lowcut, highcut=None, fs=200 , order =order)
                    new_signals_pyedf_behind_ellip = ellip_filter_sos(signals=signals_pyedf[index].reshape(1,-1),rp=rp,rs=rs ,lowcut=emg_lowcut, highcut=None, fs = 200 , order = order)
                
                new_signals_pyedf = np.concatenate([new_signals_pyedf,new_signals_pyedf_behind],axis=0)
                # new_signals_pyedf_sos = np.concatenate([new_signals_pyedf_sos,new_signals_pyedf_behind_sos],axis=0)

                new_signals_pyedf_ellip = np.concatenate([new_signals_pyedf_ellip,new_signals_pyedf_behind_ellip],axis=0)
                count += 1

        print(new_signals_pyedf.shape)
        
        signals.pick_channels(['C3-M2', 'C4-M1', 'F4-M1', 'F3-M2', 'O2-M1', 'O1-M2', 'E1-M2', 'E2-M1', '1-2'])
        

        
        print('is same?? : ' , (new_signals_pyedf[0,:] == signals[:][0][0,:]).all())


        # each channel bandpass filter ( EEG & EMG )
        signals.filter(eeg_lowcut, eeg_highcut, picks=eeg_channel)
        signals.filter(eog_lowcut, eog_highcut, picks=eog_channel)
        # 1-2(EMG) highpass filter

        signals.filter(emg_lowcut, h_freq=None, picks=emg_channel)

        # tuple to numpy
        new_signals = signals[:][0]

        print(new_signals.shape)
        if len(new_signals) != 9:
            print('This file is fault!')
        else:
            if dif_sec > 0:
                print('Annotations is longer than Signals')

                new_signals = new_signals[:, dif_sec * 200:]
                new_signals_pyedf = new_signals_pyedf[:,dif_sec *200:]
                # new_signals_pyedf_sos = new_signals_pyedf_sos[:,dif_sec *200:]
                new_signals_pyedf_ellip = new_signals_pyedf_ellip[:,dif_sec *200:]

                tail_dif_len = len(new_signals[0]) - len(annotations_np) * 200 * 30

                print('tail dif : ', tail_dif_len)

                if tail_dif_len > 0:
                    new_signals = new_signals[:, :-tail_dif_len]
                    new_signals_pyedf = new_signals_pyedf[:, :-tail_dif_len]
                    # new_signals_pyedf_sos = new_signals_pyedf_sos[:, :-tail_dif_len]
                    new_signals_pyedf_ellip = new_signals_pyedf_ellip[:, :-tail_dif_len]
                    
                    print('signals len : ', len(new_signals[0]) / 200 / 30)
                    print('annotations len : ', len(annotations_np))
                else:
                    signals_len = len(new_signals[0]) // 30 // 200
                    signals_len = signals_len * 30 * 200

                    new_signals = new_signals[:, :signals_len]
                    new_signals_pyedf = new_signals_pyedf[:, :signals_len]
                    # new_signals_pyedf_sos = new_signals_pyedf_sos[:, :signals_len]
                    new_signals_pyedf_ellip = new_signals_pyedf_ellip[:, :signals_len]

                    annotations_np = annotations_np[:len(new_signals[0]) // 30 // 200]
                    print('signals len : ', len(new_signals[0]) / 200 / 30)
                    print('annotations len : ', len(annotations_np))
                if len(new_signals[0]) / 200 / 30 == len(annotations_np):
                    print('Truth file')
                    print(new_signals.shape)
                    print(annotations_np.shape)
                    print('signals_filename : %s ' % (
                            signals_mne_save_path + signals_path.split('/')[-1].split('_')[0] + '_%d_.npy'%severity))
                    print('annotations_filename : %s' % (
                            signals_mne_save_path + signals_path.split('/')[-1].split('_')[
                        0] + '_%d_.npy'%severity))
                    # signals
                    np.save(signals_mne_save_path + signals_path.split('/')[-1].split('_')[0] + '_%d_.npy'%severity,
                            new_signals)
                    np.save(signals_butter_save_path + signals_path.split('/')[-1].split('_')[0] + '_%d_.npy'%severity,
                            new_signals_pyedf)
                    # np.save(signals_sos_save_path + signals_path.split('/')[-1].split('_')[0] + '_%d_.npy'%severity,
                    #         new_signals_pyedf_sos)
                    np.save(signals_ellip_save_path + signals_path.split('/')[-1].split('_')[0] + '_%d_.npy'%severity,
                            new_signals_pyedf_ellip)
                    #annotations
                    np.save(annotations_save_path + signals_path.split('/')[-1].split('_')[0] + '_%d_.npy'%severity,
                            annotations_np)

            elif dif_sec < 0:
                print('Signals is longer than Annotations')
            else:
                tail_dif_len = len(new_signals[0]) - len(annotations_np) * 200 * 30

                print('tail dif : ', tail_dif_len)

                if tail_dif_len > 0:
                    new_signals = new_signals[:, :-tail_dif_len]
                    new_signals_pyedf = new_signals_pyedf[:, :-tail_dif_len]
                    # new_signals_pyedf_sos = new_signals_pyedf_sos[:, :-tail_dif_len]
                    new_signals_pyedf_ellip = new_signals_pyedf_ellip[:, :-tail_dif_len]
                    
                    print('signals len : ', len(new_signals[0]) / 200 / 30)
                    print('annotations len : ', len(annotations_np))
                else:
                    signals_len = len(new_signals[0]) // 30 // 200
                    signals_len = signals_len * 30 * 200

                    new_signals = new_signals[:, :signals_len]
                    new_signals_pyedf = new_signals_pyedf[:, :signals_len]
                    # new_signals_pyedf_sos = new_signals_pyedf_sos[:, :signals_len]
                    new_signals_pyedf_ellip = new_signals_pyedf_ellip[:, :signals_len]

                    annotations_np = annotations_np[:len(new_signals[0]) // 30 // 200]
                    print('signals len : ', len(new_signals[0]) / 200 / 30)
                    print('annotations len : ', len(annotations_np))
                if len(new_signals[0]) / 200 / 30 == len(annotations_np):
                    print('Truth file')
                    print(new_signals.shape)
                    print(annotations_np.shape)
                    print('signals_filename : %s ' % (
                            signals_mne_save_path + signals_path.split('/')[-1].split('_')[0] + '_%d_.npy'%severity))
                    print('annotations_filename : %s' % (
                            signals_mne_save_path + signals_path.split('/')[-1].split('_')[
                        0] + '_%d_.npy'%severity))
                    # signals
                    np.save(signals_mne_save_path + signals_path.split('/')[-1].split('_')[0] + '_%d_.npy'%severity,
                            new_signals)
                    np.save(signals_butter_save_path + signals_path.split('/')[-1].split('_')[0] + '_%d_.npy'%severity,
                            new_signals_pyedf)
                    # np.save(signals_sos_save_path + signals_path.split('/')[-1].split('_')[0] + '_%d_.npy'%severity,
                    #         new_signals_pyedf_sos)
                    np.save(signals_ellip_save_path + signals_path.split('/')[-1].split('_')[0] + '_%d_.npy'%severity,
                            new_signals_pyedf_ellip)
                    #annotations
                    np.save(annotations_save_path + signals_path.split('/')[-1].split('_')[0] + '_%d_.npy'%severity,
                            annotations_np)


def make_edf_to_npy_usingmne(directory_path='/data/ssd1/SeoulDataset_edf/origin_edf/seoulDataset/'):
    signals_list = []
    annotations_list = []
    path_list = []
    cpu_num = multiprocessing.cpu_count()
    signals_mne_save_path = '/data/hdd1/dataset/Seoul_dataset/9channel_prefilter_mne/signals/'
    # signals_sos_save_path = '/mnt/hdd1/dataset/Seoul_dataset/9channel_prefilter_sos/signals/'
    signals_butter_save_path = '/data/hdd1/dataset/Seoul_dataset/9channel_prefilter_butter/signals/'
    signals_ellip_save_path = '/data/hdd1/dataset/Seoul_dataset/9channel_prefilter_ellip/signals/'
    annotations_save_path = '/data/hdd1/dataset/Seoul_dataset/annotations/'
    os.makedirs(signals_mne_save_path,exist_ok=True)
    # os.makedirs(signals_sos_save_path,exist_ok=True)
    os.makedirs(signals_butter_save_path,exist_ok=True)
    os.makedirs(signals_ellip_save_path,exist_ok=True)
    os.makedirs(annotations_save_path,exist_ok=True)
    print('cpu_num : ',cpu_num)
    for (path, dir, files) in os.walk(directory_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.edf':
                signals_list.append('%s/%s' % (path, filename))
                annotations_filename = filename[:-7] + 'event.csv'
                if path.split('/')[-1] == '1. EDF':
                    annotations_path = '/'.join(path.split('/')[:-1]) + '/2. Event/'
                    annotations_list.append(annotations_path + annotations_filename)
                else:
                    annotations_list.append('%s/%s'%(path,annotations_filename))

    cpu_num = multiprocessing.cpu_count()

    for i in range(len(signals_list)):
        path_list.append([signals_list[i], annotations_list[i]])
    print(len(path_list))

    for i in range(len(path_list)):
        if path_list[i][0].split('/')[-1].split('_')[0] != path_list[i][1].split('/')[-1].split('_')[0]:
            print('This is fault')
        
    # my_thread(path_list[0])    
    start = time.time()
    pool = Pool(cpu_num)

    pool.map(my_thread,path_list)
    pool.close()
    pool.join()




def check_dataset_truth():
    cpu_num = multiprocessing.cpu_count()
    signals_mne_save_path = '/mnt/ssd1/dataset/Seoul_dataset/9channel_prefilter_mne/signals/'
    # signals_sos_save_path = '/mnt/ssd1/dataset/Seoul_dataset/9channel_prefilter_sos/signals/'
    signals_butter_save_path = '/mnt/ssd1/dataset/Seoul_dataset/9channel_prefilter_butter/signals/'
    signals_ellip_save_path = '/mnt/ssd1/dataset/Seoul_dataset/9channel_prefilter_ellip/signals/'
    annotations_save_path = '/mnt/ssd1/dataset/Seoul_dataset/9channel_prefilter/annotations/'
    file_list = os.listdir(signals_mne_save_path)

    
    signals_mne = np.load(signals_mne_save_path+file_list[0])
    signals_butter = np.load(signals_butter_save_path+file_list[0])
    # signals_sos = np.load(signals_sos_save_path+file_list[0])
    signals_ellip = np.load(signals_ellip_save_path+file_list[0])

    print(signals_mne.shape)
    print(signals_butter.shape)
    print(signals_sos.shape)
    print(signals_ellip.shape)
    plt.plot(signals_mne[1,6000:12000])
    plt.savefig('/home/eslab/mne')
    plt.cla()

    plt.plot(signals_butter[1,6000:12000])
    plt.savefig('/home/eslab/butter')
    plt.cla()

    plt.plot(signals_sos[1,6000:12000])
    plt.savefig('/home/eslab/sos')
    plt.cla()

    plt.plot(signals_ellip[1,6000:12000])
    plt.savefig('/home/eslab/ellip')
    plt.cla()
    

    # for i in range(batch_size):
    #     if i == 0:
    #         batch_signals = batch_data[i][0]
    #         batch_labels = batch_data[i][1]
            
    #     else:
    #         batch_signals = torch.cat((batch_signals,batch_data[i][0]),dim=1)
    #         batch_labels = torch.cat((batch_labels,batch_data[i][1]))
       
    # print(batch_signals[0,:10])
    # print(batch_data[0][0][0,:10])
    # batch_signals = batch_signals.unsqueeze(0)
    # print(batch_signals.shape)
    # batch_signals = batch_signals.transpose(1,2)
    # print(batch_signals.shape)
    # batch_signals = batch_signals.view(-1,6000,9)
    # print(batch_signals.shape)
    # batch_signals = batch_signals.transpose(1,2)

    # print(batch_signals.shape)
    # print(batch_labels.shape)
    # print(time.time()-start_time)

