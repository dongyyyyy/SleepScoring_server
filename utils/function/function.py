from include.header import *


#Standard Scaler torch
def data_preprocessing_torch(signals): # 하나의 데이터셋에 대한 data_preprocessing (using torch)
    signals = (signals - signals.mean(dim=1).unsqueeze(1))/signals.std(dim=1).unsqueeze(1)
    return signals

#Standard Scaler npy
def data_preprocessing_numpy(signals): # zero mean unit variance 한 환자에 대한 signal 전체에 대한 normalize
    signals = (signals - np.expand_dims(signals.mean(axis=1), axis=1)) / np.expand_dims(signals.std(axis=1), axis=1)
    return signals

#MinMax Scaler torch
def data_preprocessing_oneToOne_torch(signals,min,max,max_value):
    signals_std = (signals + max_value) / (2*max_value)
    signals_scaled = signals_std * (max - min) + min
    return signals_scaled

def get_dataset_selectChannel(signals_path,annotations_path,filename,select_channel=[0,1,2],use_noise=False,epsilon=0.5,noise_scale=2e-6,preprocessing=False,norm_methods='Standard',cut_value=200,device='cpu'):
    signals = np.load(signals_path+filename)

    annotations = np.load(annotations_path+filename)
    # print(signals.shape)
    signals = signals[select_channel]

    signals = torch.from_numpy(signals).float().to(device)
    annotations = torch.from_numpy(annotations).long().to(device)

    if preprocessing:
        if norm_methods=='Standard':
            signals = data_preprocessing_torch(signals)
        elif norm_methods=='OneToOne':
            signals = torch.where(signals < -cut_value, -cut_value, signals)
            signals = torch.where(signals > cut_value, cut_value, signals)
            signals = data_preprocessing_oneToOne_torch(signals,-1,1,cut_value)
        elif norm_methods=='MinMax':
            signals = torch.where(signals < -cut_value, -cut_value, signals)
            signals = torch.where(signals > cut_value, cut_value, signals)
            signals = data_preprocessing_oneToOne_torch(signals,0,1,cut_value)

    return signals,annotations

def expand_signals_torch(signals,channel_len,sample_rate=200,epoch_sec=30):
    signals = signals.unsqueeze(0)
    #print(signals.shape)
    signals = signals.transpose(1,2)
    #print(batch_signals.shape)
    signals = signals.view(-1,sample_rate*epoch_sec,channel_len)
    #print(batch_signals.shape)
    signals = signals.transpose(1,2)
    return signals

def suffle_dataset_list(dataset_list): # 데이터 셔플
    random.shuffle(dataset_list)
    return dataset_list

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:         # Conv weight init
        torch.nn.init.xavier_uniform_(m.weight.data)

def int_to_string(num):
    str_num = str(num).zfill(4)
    return str_num

# lowpass filter
def butter_lowpass_filter(data, cutoff, order=4,nyq=100):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(N=order, Wn=normal_cutoff, btype='low', analog=False,output='ba')
    y = filtfilt(b, a, data)
    return y

# highpass filter
def butter_highpass_filter(data, cutoff, order=4,fs=200):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(N=order, Wn=normal_cutoff, btype='high', analog=False,output='ba')

    y = filtfilt(b, a, data)
    # b = The numerator coefficient vector of the filter (분자)
    # a = The denominator coefficient vector of the filter (분모)

    return y

def butter_bandpass(lowcut, highcut, fs=200 , order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b,a = butter(N=order,Wn=[low,high],btype='bandpass', analog=False,output='ba')
    return b,a

# bandpass filter
def butter_bandpass_filter(signals, lowcut, highcut, fs , order = 4):
    b,a = butter_bandpass(lowcut,highcut,fs,order=order)

    y = lfilter(b,a,signals)
    return y

def butter_filter_sos(signals, lowcut=None, highcut=None, fs=200 , order =4):
    if lowcut != None and highcut != None: # bandpass filter
        sos = signal.butter(N=order,Wn=[lowcut,highcut],btype='bandpass',analog=False,output='sos',fs=fs)
        filtered = signal.sosfilt(sos,signals)
    elif lowcut != None and highcut == None: # highpass filter
        sos = signal.butter(N=order,Wn=lowcut,btype='highpass',analog=False,output='sos',fs=fs)
    elif lowcut == None and highcut != None: 
        sos = signal.butter(N=order,Wn=highcut,btype='lowpass',analog=False,output='sos',fs=fs)
    else: # None filtering
        return signals 
    filtered = signal.sosfilt(sos,signals)
    return filtered

def ellip_filter_sos(signals,rp=6,rs=53, lowcut=None, highcut=None, fs = 200 , order = 4):
    if lowcut != None and highcut != None: # bandpass filter
        sos = signal.ellip(N=order,rp=rp,rs=rs,Wn=[lowcut,highcut],btype='bandpass',analog=False,output='sos',fs=fs)
    elif lowcut != None and highcut == None: # highpass filter
        sos = signal.ellip(N=order,rp=rp,rs=rs,Wn=lowcut,btype='highpass',analog=False,output='sos',fs=fs)
    elif lowcut == None and highcut != None: 
        sos = signal.ellip(N=order,rp=rp,rs=rs,Wn=highcut,btype='lowpass',analog=False,output='sos',fs=fs)
    else: # None filtering
        return signals 
    filtered = signal.sosfilt(sos,signals)
    return filtered