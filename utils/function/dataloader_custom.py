from utils.function.function import *


def make_weights_for_balanced_classes(data_list, nclasses=5):
    count = [0] * nclasses
    
    for data in data_list:
        count[int(data.split('.npy')[0].split('_')[-1])] += 1

    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(data_list)
    for idx, val in enumerate(data_list):
        weight[idx] = weight_per_class[int(val.split('.npy')[0].split('_')[-1])]
    return weight , count


class Sleep_Dataset_cnn(object):
    def read_dataset(self):
        all_signals_files = []
        all_labels = []

        for dataset_folder in self.dataset_list:
            signals_path = self.data_path + dataset_folder+'/'
            signals_list = os.listdir(signals_path)
            signals_list.sort()
            for signals_filename in signals_list:
                signals_file = signals_path+signals_filename
                all_signals_files.append(signals_file)
                all_labels.append(int(signals_filename.split('.npy')[0].split('_')[-1]))
                
        return all_signals_files, all_labels, len(all_signals_files)

    def __init__(self,
                 data_path,
                 dataset_list,
                 class_num=5,
                 use_scaling=False,
                 scaling=1e+6,
                 use_noise=False,
                 epsilon=0.8,
                 noise_scale=1e-6,
                 preprocessing=False,
                 preprocessing_type = 'Standard',
                 cut = False,
                 cut_value = 192e-6,
                 use_channel=[0,1,2],
                 use_cuda = True,
                 ):

        self.class_num = class_num
        self.data_path = data_path
        self.dataset_list = dataset_list
        self.signals_files_path, self.labels, self.length = self.read_dataset()

        self.preprocessing = preprocessing
        self.preprocessing_type = preprocessing_type

        self.use_noise = use_noise
        self.epsilon = epsilon
        self.noise_scale = noise_scale
        self.cut = cut
        self.use_scaling = use_scaling
        self.scaling = scaling
        
        if use_cuda:
            self.cut_value = torch.tensor(cut_value)
        else:
            self.cut_value = cut_value

        self.use_channel = use_channel
        self.use_cuda = use_cuda

    def __getitem__(self, index):
        signals = np.load(self.signals_files_path[index])
        label = self.labels[index]
        
        if len(signals) > 10: # 단일채널인 경우
            signals = signals.reshape(1,-1)
        if self.use_cuda:
            signals = torch.from_numpy(signals).float()

        signals = signals[self.use_channel]

        if self.use_scaling:
            signals = signals * self.scaling

        if self.use_noise:
            if np.random.rand() < self.epsilon:
                noise = torch.normal(mean=0., std=self.noise_scale, size=signals.shape)
                signals = signals + noise

        return signals,label
    def __len__(self):
        return self.length


class Sleep_Dataset_sequence(object):
    def read_dataset(self):
        all_signals_files = []
        all_labels = []

        for dataset_folder in self.dataset_list:
            signals_path = self.data_path + dataset_folder+'/'
            signals_list = os.listdir(signals_path)

            signals_list.sort()
            for index in range(0,len(signals_list)//self.sequence_length*self.sequence_length,self.window_size):
                signals_file = signals_path+signals_list[index]
                all_signals_files.append(signals_file)
            # for signals_filename in signals_list:
            #     signals_file = signals_path+signals_filename
            #     all_signals_files.append(signals_file)
                
        return all_signals_files, len(all_signals_files)

    def __init__(self,
                 data_path,
                 dataset_list,
                 class_num=5,
                 use_scaling=False,
                 scaling=1e+6,
                 use_noise=False,
                 epsilon=0.8,
                 noise_scale=1e-6,
                 preprocessing=False,
                 preprocessing_type = 'Standard',
                 cut = False,
                 cut_value = 192e-6,
                 use_channel=[0,1,2],
                 use_cuda = True,
                 sequence_length=10,
                 len_divide=2,
                 window_size=1
                 ):

        self.class_num = class_num
        self.data_path = data_path
        self.dataset_list = dataset_list
        # self.window_slide = window_slide

        self.preprocessing = preprocessing
        self.preprocessing_type = preprocessing_type

        self.use_noise = use_noise
        self.epsilon = epsilon
        self.noise_scale = noise_scale
        self.cut = cut
        self.use_scaling = use_scaling
        self.scaling = scaling

        if use_cuda:
            self.cut_value = torch.tensor(cut_value)
        else:
            self.cut_value = cut_value
        self.use_channel = use_channel
        self.use_cuda = use_cuda

        self.sequence_length = sequence_length

        self.len_divide = len_divide
        self.window_size = window_size

        self.signals_files_path, self.length = self.read_dataset()
        
        # self.train_batch_epoch = train_batch_epoch

    def __iter__(self):
        return iter(range(0,self.train_batch_epoch))

    def __getitem__(self, index):
        folder_name = '/'.join(self.signals_files_path[index].split('/')[:-1]) + '/'
        folder_list = os.listdir(folder_name)
        folder_list.sort()
        folder_length = len(folder_list)-1
        current_length = int(self.signals_files_path[index].split('/')[-1].split('_')[0])
        dif_len = folder_length-current_length
        
        if dif_len < self.sequence_length:
            start_length = current_length - (self.sequence_length - (dif_len+1))
        else:
            start_length = current_length
        start_num = int_to_string(start_length)
        # print(f'folder name : {folder_name} / filename : {self.signals_files_path[index]} / folder length : {folder_length} / dif len : {dif_len } / start_num : {start_num}')
        count = 0
        signals = None
        label = None
        # print(f'dif len : {dif_len}')
        for filename in folder_list:
            if start_num in filename:
                c_signals = np.load(self.signals_files_path[index])
                c_label = int(filename.split('.npy')[0].split('_')[-1])
                c_signals = c_signals[self.use_channel]
                if len(c_signals) > 10: # 단일채널인 경우
                    c_signals = c_signals.reshape(1,-1)
                if self.use_cuda:
                    c_signals = torch.from_numpy(c_signals).float()

                if self.use_scaling:
                    c_signals = c_signals * self.scaling

                if self.use_noise:
                    if np.random.rand() < self.epsilon:
                        noise = torch.normal(mean=0., std=self.noise_scale, size=c_signals.shape)
                        c_signals = c_signals + noise
                start_num = int_to_string(int(start_num)+1)

                if count == 0:
                    signals = c_signals.unsqueeze(0)
                    label = c_label
                else:
                    signals = torch.cat((signals,c_signals.unsqueeze(0)))
                    label = np.append(label,c_label)
                count += 1
            if count == self.sequence_length:
                break
        assert len(label) == self.sequence_length, print(f'dif_len : {dif_len } /current_length : {current_length} /start_length : {start_length} / start_num : {start_num} / folder_length : {folder_length} / signals shape : {signals.shape} / label shape : {label.shape}')

        return signals,label
    def __len__(self):
        return self.length // self.len_divide


class Sleep_Dataset_sequence_val(object):
    def read_dataset(self):
        all_signals_files = []

        for dataset_folder in self.dataset_list:
            signals_path = self.data_path + dataset_folder+'/'
            signals_list = os.listdir(signals_path)

            signals_list.sort()
            
            signals_length = len(signals_list)

            for index in range(signals_length//self.sequence_length):
                signals_files = []
                for file_index in range(self.sequence_length):
                    signals_file = signals_path + signals_list[file_index + index*self.sequence_length]
                    signals_files.append(signals_file)
                
                all_signals_files.append(signals_files)
                
        return all_signals_files, len(all_signals_files)

    def __init__(self,
                 data_path,
                 dataset_list,
                 class_num=5,
                 use_scaling=False,
                 scaling=1e+6,
                 use_noise=False,
                 epsilon=0.8,
                 noise_scale=1e-6,
                 preprocessing=False,
                 preprocessing_type = 'Standard',
                 cut = False,
                 cut_value = 192e-6,
                 use_channel=[0,1,2],
                 use_cuda = True,
                 sequence_length=10,
                 ):

        self.class_num = class_num
        self.data_path = data_path
        self.dataset_list = dataset_list
        
        self.preprocessing = preprocessing
        self.preprocessing_type = preprocessing_type

        self.use_noise = use_noise
        self.epsilon = epsilon
        self.noise_scale = noise_scale
        self.cut = cut
        self.use_scaling = use_scaling
        self.scaling = scaling

        if use_cuda:
            self.cut_value = torch.tensor(cut_value)
        else:
            self.cut_value = cut_value
        self.use_channel = use_channel
        self.use_cuda = use_cuda

        self.sequence_length = sequence_length

        self.signals_files_path, self.length = self.read_dataset()
        # print('file list : ',self.signals_files_path)

    def __getitem__(self, index):
        for index, filename in enumerate(self.signals_files_path[index]):
            c_signals = np.load(filename)
            c_label = int(filename.split('.npy')[0].split('_')[-1])
            c_signals = c_signals[self.use_channel]
            
            if len(c_signals) > 10: # 단일채널인 경우
                c_signals = c_signals.reshape(1,-1)
            if self.use_cuda:
                c_signals = torch.from_numpy(c_signals).float()

            if self.use_scaling:
                c_signals = c_signals * self.scaling

            if self.use_noise:
                if np.random.rand() < self.epsilon:
                    noise = torch.normal(mean=0., std=self.noise_scale, size=c_signals.shape)
                    c_signals = c_signals + noise

            if index == 0:
                signals = c_signals.unsqueeze(0)
                label = c_label
            else:
                signals = torch.cat((signals,c_signals.unsqueeze(0)))
                label = np.append(label,c_label)

        assert len(label) == self.sequence_length, 'fault!'

        return signals,label

    def __len__(self):
        return self.length



class Sleep_Dataset_sequence_onePerson(object):
    def read_dataset(self):
        all_signals_files = []

        for dataset_folder in self.dataset_list:
            signals_path = self.data_path + dataset_folder+'/'

            all_signals_files.append(signals_path)
            # for signals_filename in signals_list:
            #     signals_file = signals_path+signals_filename
            #     all_signals_files.append(signals_file)
                
        return all_signals_files, len(all_signals_files)

    def __init__(self,
                 data_path,
                 dataset_list,
                 class_num=5,
                 use_scaling=False,
                 scaling=1e+6,
                 use_noise=False,
                 epsilon=0.8,
                 noise_scale=1e-6,
                 preprocessing=False,
                 preprocessing_type = 'Standard',
                 cut = False,
                 cut_value = 192e-6,
                 use_channel=[0,1,2],
                 use_cuda = True,
                 sequence_length=10,
                 len_divide=1,
                 window_size=1
                 ):

        self.class_num = class_num
        self.data_path = data_path
        self.dataset_list = dataset_list
        # self.window_slide = window_slide

        self.preprocessing = preprocessing
        self.preprocessing_type = preprocessing_type

        self.use_noise = use_noise
        self.epsilon = epsilon
        self.noise_scale = noise_scale
        self.cut = cut
        self.use_scaling = use_scaling
        self.scaling = scaling

        if use_cuda:
            self.cut_value = torch.tensor(cut_value)
        else:
            self.cut_value = cut_value
        self.use_channel = use_channel
        self.use_cuda = use_cuda

        self.sequence_length = sequence_length

        self.len_divide = len_divide
        self.window_size = window_size

        self.signals_files_path, self.length = self.read_dataset()
        
        # self.train_batch_epoch = train_batch_epoch

    def __iter__(self):
        return iter(range(0,self.train_batch_epoch))

    def __getitem__(self, index):
        folder_name = self.signals_files_path[index]
        folder_list = os.listdir(folder_name)
        folder_list.sort()
        # print('folder len : ', len(folder_list))
        if self.sequence_length != 0: # sequence length가 존재
            max_len = (len(folder_list)-1)-self.sequence_length
            
            # print(f'dif len : {dif_len}')
            for index in range(0,max_len,self.window_size):
                for current_index in range(self.sequence_length):
                    # print('current index : ',index+current_index)
                    c_signals = np.load(folder_name+folder_list[index+current_index])
                    c_label = int(folder_list[index+current_index].split('.npy')[0].split('_')[-1])
                    c_signals = c_signals[self.use_channel]
                    if len(c_signals) > 10: # 단일채널인 경우
                        c_signals = c_signals.reshape(1,-1)
                    if self.use_cuda:
                        c_signals = torch.from_numpy(c_signals).float()

                    if self.use_scaling:
                        c_signals = c_signals * self.scaling

                    if self.use_noise:
                        if np.random.rand() < self.epsilon:
                            noise = torch.normal(mean=0., std=self.noise_scale, size=c_signals.shape)
                            c_signals = c_signals + noise


                    if current_index == 0:
                        in_signals = c_signals.unsqueeze(0) # (1,channel, vec)
                        in_label = c_label
                    else:
                        in_signals = torch.cat((in_signals,c_signals.unsqueeze(0))) # (n,channel,vec)
                        in_label = np.append(in_label,c_label)
                if index == 0:
                    signals = in_signals.unsqueeze(0)
                    label = in_label
                else:
                    signals = torch.cat((signals,in_signals.unsqueeze(0)))
                    label = np.append(label,in_label)
            #assert len(label) == self.sequence_length, print(f'dif_len : {dif_len } /current_length : {current_length} /start_length : {start_length} / start_num : {start_num} / folder_length : {folder_length} / signals shape : {signals.shape} / label shape : {label.shape}')
        else: # 하나의 환자 데이터로 학습
            # print('One person')
            for index in range(0,len(folder_list),1):
                c_signals = np.load(folder_name+folder_list[index])
                c_label = int(folder_list[index].split('.npy')[0].split('_')[-1])
                c_signals = c_signals[self.use_channel]

                if len(c_signals) > 10: # 단일채널인 경우
                    c_signals = c_signals.reshape(1,-1)

                if self.use_cuda:
                    c_signals = torch.from_numpy(c_signals).float()

                if self.use_scaling:
                    c_signals = c_signals * self.scaling

                if self.use_noise:
                    if np.random.rand() < self.epsilon:
                        noise = torch.normal(mean=0., std=self.noise_scale, size=c_signals.shape)
                        c_signals = c_signals + noise

                if index == 0:
                    signals = c_signals.unsqueeze(0) # (1,channel, vec)
                    label = c_label
                else:
                    signals = torch.cat((signals,c_signals.unsqueeze(0))) # (n,channel,vec)
                    label = np.append(label,c_label)
        # print(signals.shape)
        return signals,label

    def __len__(self):
        return self.length


class Sleep_Dataset_sequence_val_onePerson(object):
    def read_dataset(self):
        all_signals_files = []

        for dataset_folder in self.dataset_list:
            signals_path = self.data_path + dataset_folder+'/'

            all_signals_files.append(signals_path)

        return all_signals_files, len(all_signals_files)

    def __init__(self,
                 data_path,
                 dataset_list,
                 class_num=5,
                 use_scaling=False,
                 scaling=1e+6,
                 use_noise=False,
                 epsilon=0.8,
                 noise_scale=1e-6,
                 preprocessing=False,
                 preprocessing_type = 'Standard',
                 cut = False,
                 cut_value = 192e-6,
                 use_channel=[0,1,2],
                 use_cuda = True,
                 sequence_length=10,
                 ):

        self.class_num = class_num
        self.data_path = data_path
        self.dataset_list = dataset_list
        
        self.preprocessing = preprocessing
        self.preprocessing_type = preprocessing_type

        self.use_noise = use_noise
        self.epsilon = epsilon
        self.noise_scale = noise_scale
        self.cut = cut
        self.use_scaling = use_scaling
        self.scaling = scaling

        if use_cuda:
            self.cut_value = torch.tensor(cut_value)
        else:
            self.cut_value = cut_value
        self.use_channel = use_channel
        self.use_cuda = use_cuda

        self.sequence_length = sequence_length

        self.signals_files_path, self.length = self.read_dataset()
        # print('file list : ',self.signals_files_path)

    def __getitem__(self, index):
        folder_name = self.signals_files_path[index]
        folder_list = os.listdir(folder_name)
        folder_list.sort()
        max_len = (len(folder_list)-1)-self.sequence_length
        # print(f'dif len : {dif_len}')
        if self.sequence_length != 0: # sequence length가 존재
            for index in range(0,max_len,self.sequence_length):
                for current_index in range(self.sequence_length):
                    c_signals = np.load(folder_name+folder_list[index+current_index])
                    c_label = int(folder_list[index+current_index].split('.npy')[0].split('_')[-1])
                    c_signals = c_signals[self.use_channel]
                    if len(c_signals) > 10: # 단일채널인 경우
                        c_signals = c_signals.reshape(1,-1)
                    if self.use_cuda:
                        c_signals = torch.from_numpy(c_signals).float()

                    if self.use_scaling:
                        c_signals = c_signals * self.scaling

                    if self.use_noise:
                        if np.random.rand() < self.epsilon:
                            noise = torch.normal(mean=0., std=self.noise_scale, size=c_signals.shape)
                            c_signals = c_signals + noise
                    

                    if current_index == 0:
                        in_signals = c_signals.unsqueeze(0) # (1,channel, vec)
                        in_label = c_label
                    else:
                        in_signals = torch.cat((in_signals,c_signals.unsqueeze(0))) # (n,channel,vec)
                        in_label = np.append(in_label,c_label)
                if index == 0:
                    signals = in_signals.unsqueeze(0)
                    label = in_label
                else:
                    signals = torch.cat((signals,in_signals.unsqueeze(0)))
                    label = np.append(label,in_label)
        else:
            for index in range(0,len(folder_list),1):
                c_signals = np.load(folder_name+folder_list[index])
                c_label = int(folder_list[index].split('.npy')[0].split('_')[-1])
                c_signals = c_signals[self.use_channel]

                if len(c_signals) > 10: # 단일채널인 경우
                    c_signals = c_signals.reshape(1,-1)

                if self.use_cuda:
                    c_signals = torch.from_numpy(c_signals).float()

                if self.use_scaling:
                    c_signals = c_signals * self.scaling

                if self.use_noise:
                    if np.random.rand() < self.epsilon:
                        noise = torch.normal(mean=0., std=self.noise_scale, size=c_signals.shape)
                        c_signals = c_signals + noise

                if index == 0:
                    signals = c_signals.unsqueeze(0) # (1,channel, vec)
                    label = c_label
                else:
                    signals = torch.cat((signals,c_signals.unsqueeze(0))) # (n,channel,vec)
                    label = np.append(label,c_label)

        #assert len(label) == self.sequence_length, print(f'dif_len : {dif_len } /current_length : {current_length} /start_length : {start_length} / start_num : {start_num} / folder_length : {folder_length} / signals shape : {signals.shape} / label shape : {label.shape}')

        return signals,label

    def __len__(self):
        return self.length