from include.header import *

from utils.function.function import *
from utils.function.loss_fn import *
from utils.function.dataloader_custom import *
from utils.function.scheduler import *

from models.cnn.ResNet import *
from models.rnn.lstm import *


select_channel = [1,2,3]

def train_rnn_dataloader_cuda1(save_filename,logging_filename,load_filename,signals_path, train_dataset_list,val_dataset_list,batch_size = 10000,
                                                 epochs=2000,learning_rate=0.001,use_scaling=False,scaling=1e+6,
                                          optim='Adam',loss_function='CE',epsilon=0.7,noise_scale=2e-6,
                                          use_noise=True,preprocessing=False,preprocessing_methods='Standard',use_cut = False, cut_value = 192e-6,use_channel=[0,1,2],scheduler=None,warmup_iter=10,cosine_decay_iter=40,stop_iter=300,gamma=0.8,sequence_length=10):
    # cpu processor num
    cpu_num = multiprocessing.cpu_count()
    train_batch_epoch = 1000
    print('train dataset len : ',len(train_dataset_list))
    print('val dataset len : ',len(val_dataset_list))

    #dataload Training Dataset
    train_dataset = Sleep_Dataset_sequence(data_path=signals_path,dataset_list=train_dataset_list,class_num=5,
    use_scaling=use_scaling,scaling=scaling,use_noise=use_noise,epsilon=epsilon,noise_scale=noise_scale,
    preprocessing=preprocessing,preprocessing_type = preprocessing_methods,cut = use_cut,cut_value = cut_value,use_channel=use_channel,use_cuda = True,sequence_length=sequence_length)
    
    weights,count = make_weights_for_balanced_classes(train_dataset.signals_files_path)
    # print(f'weights : {weights} / count : {count}')
    

    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights,len(weights))
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=True, num_workers=cpu_num)
    #train_dataloader = DataLoader(dataset=train_dataset,batch_size=10000,sampler=sampler,num_workers=20)

    #dataload Validation Dataset
    val_dataset = Sleep_Dataset_sequence_val(data_path=signals_path,dataset_list=val_dataset_list,class_num=5,
    use_scaling=use_scaling,scaling=scaling,use_noise=use_noise,epsilon=epsilon,noise_scale=noise_scale,
    preprocessing=preprocessing,preprocessing_type = preprocessing_methods,cut = use_cut,cut_value = cut_value,use_channel=use_channel,use_cuda = True,sequence_length=sequence_length)

    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=cpu_num//2)
    print(train_dataset.length,val_dataset.length)
    # Adam optimizer paramQ
    b1 = 0.5
    b2 = 0.999

    beta = 0.001
    norm_square = 2

    check_file = open(logging_filename, 'w')  # logging file

    best_accuracy = 0.
    best_epoch = 0

    #model = DeepSleepNet_classification(class_num=5)
    # model = resnet18_custom_withoutDropout(in_channel=1,layer_filters=[64,128,128,128],first_conv=[200,10,100],block_kernel_size=3,
    #                                        padding=1,use_batchnorm=True)
    # model = resnet18_custom_withoutDropout(in_channel=3, layer_filters=[64, 128, 256, 512], first_conv=[7, 2, 3],
    #                                        block_kernel_size=3,
    #                                        padding=1, use_batchnorm=True, num_classes=5)
    FeatureExtract = resnet18_200hz_withDropout_ensemble_branch_new_FE()
    Classification = lstm(flat=512*3,sequence_length=sequence_length)
    # model = resnet18_withoutDropout_200hz(in_channel=1,layer_filters=layer_filters,first_conv=first_conv,num_classes=5)
    device2 = device3
    torch.cuda.set_device(device2)
    print('current gpu num : ', torch.cuda.current_device())
    # model.apply(weights_init)  # weight init
    # print(torch.load(load_filename))
    FeatureExtract = FeatureExtract.cpu()
    FeatureExtract.load_state_dict(torch.load(load_filename))
    
    cuda = torch.cuda.is_available()

    if cuda:
        print('can use CUDA!!!')
        FeatureExtract = FeatureExtract.to(device2)
        Classification = Classification.to(device2)
    # summary(model,[1,6000])
    print('torch.cuda.device_count() : ', torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        print('Multi GPU Activation !!!', torch.cuda.device_count())
        # model = nn.DataParallel(model)

    # summary(model, (3, 6000))

    print('loss function : %s' % loss_function)
    if loss_function == 'CE':
        loss_fn = nn.CrossEntropyLoss().to(device2)
    elif loss_function == 'CEW':
        samples_per_cls = count / np.sum(count)
        no_of_classes = 5
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * no_of_classes
        weights = torch.tensor(weights).float()
        weights = weights.to(device2)
        loss_fn = nn.CrossEntropyLoss(weight=weights).to(device2)
    elif loss_function == 'FL':
        loss_fn = FocalLoss(gamma=2).to(device2)
    elif loss_function == 'CBL':
        samples_per_cls = count / np.sum(count)
        loss_fn = CB_loss(samples_per_cls=samples_per_cls, no_of_classes=5, loss_type='focal', beta=0.9999,
                          gamma=2.0)
    # loss_fn = FocalLoss(gamma=2).to(device)

    # optimizer ADAM (SGD의 경우에는 정상적으로 학습이 진행되지 않았음)
    if optim == 'Adam':
        print('Optimizer : Adam')
        optimizer = torch.optim.Adam([{'params': FeatureExtract.parameters(), 'lr': 0},
                                     {'params': Classification.parameters()}], lr=learning_rate, betas=(b1, b2))
    elif optim == 'RMS':
        print('Optimizer : RMSprop')
        optimizer = torch.optim.RMSprop([{'params': FeatureExtract.parameters(), 'lr': 0},
                                     {'params': Classification.parameters()}], lr=learning_rate)
    elif optim == 'SGD':
        print('Optimizer : SGD')
        optimizer = torch.optim.SGD([{'params': FeatureExtract.parameters(), 'lr': 0},
                                     {'params': Classification.parameters()}], lr=learning_rate, momentum=0.9)


    gamma = 0.8

    lr = learning_rate
    epochs = epochs
    if scheduler == 'WarmUp_restart_gamma':
        print(f'target lr : {learning_rate} / warmup_iter : {warmup_iter} / cosine_decay_iter : {cosine_decay_iter} / gamma : {gamma}')
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, warmup_iter)
        scheduler = LearningRateWarmUP_restart_changeMax(optimizer=optimizer,
                                    warmup_iteration=warmup_iter,
                                    cosine_decay_iter=cosine_decay_iter,
                                    target_lr=lr,
                                    after_scheduler=scheduler_cosine,gamma=gamma,two_param=1) # rnn learning rate만 scheduling
    elif scheduler == 'WarmUp_restart':
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, warmup_iter)
        scheduler = LearningRateWarmUP_restart(optimizer=optimizer,
                                    warmup_iteration=warmup_iter,
                                    cosine_decay_iter=cosine_decay_iter,
                                    target_lr=lr,
                                    after_scheduler=scheduler_cosine)
    elif scheduler == 'WarmUp':
        print(f'target lr : {learning_rate} / warmup_iter : {warmup_iter} / gamma : {gamma}')
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs-warmup_iter)
        scheduler = LearningRateWarmUP(optimizer=optimizer,
                                    warmup_iteration=warmup_iter,
                                    target_lr=lr,
                                    after_scheduler=scheduler_cosine)
 
    # scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=.5)
    # loss의 값이 최소가 되도록 하며, 50번 동안 loss의 값이 감소가 되지 않을 경우 factor값 만큼
    # learning_rate의 값을 줄이고, 최저 1e-6까지 줄어들 수 있게 설정
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.5, patience=20,
    #                                                        min_lr=1e-6)
    best_accuracy = 0.
    stop_count = 0
    for epoch in range(epochs):
        train_count = 0
        if epoch == 0:
            scheduler.step(epoch)
        else:
            scheduler.step(epoch)
            train_total_loss = 0.0
            train_total_count = 0
            train_total_data = 0

            val_total_loss = 0.0
            val_total_count = 0
            val_total_data = 0

            start_time = time.time()
            FeatureExtract.eval()
            Classification.train()

            output_str = 'FeatureExtract_lr : %f\nclassification_lr : %f\n'%(optimizer.state_dict()['param_groups'][0]['lr'],optimizer.state_dict()['param_groups'][1]['lr'])
            sys.stdout.write(output_str)
            check_file.write(output_str)
            for index,data in enumerate(train_dataloader):
                batch_signal,batch_label = data

                batch_signal = batch_signal.to(device2)
                batch_label = batch_label.long().to(device2)
                # print('batch signals shape : ',batch_signal.shape)

                batch_signal = batch_signal.reshape(-1,3,6000)

                # print('batch signals shape : ',batch_signal.shape)
                optimizer.zero_grad()

                feature = FeatureExtract(batch_signal)
                

                pred = Classification(feature)

                # norm = 0
                # for parameter in model.parameters():
                #     norm += torch.norm(parameter, p=norm_square)
                # pred = [batch_size, seq_length, class_num]
                pred = pred.view(-1,5)

                #batch_label = [batch_size, seq_length]
                batch_label = batch_label.view(-1)

                loss = loss_fn(pred, batch_label) # + beta * norm

                _, predict = torch.max(pred, 1)
                check_count = (predict == batch_label).sum().item()

                train_total_loss += loss.item()

                train_total_count += check_count
                train_total_data += len(batch_signal)
                loss.backward()
                optimizer.step()

                # del (batch_signal)
                # del (batch_label)
                # del (loss)
                # del (pred)
                # torch.cuda.empty_cache()
                train_count += 1
                if train_count >= train_batch_epoch:
                    break
            train_total_loss /= index
            train_accuracy = train_total_count / train_total_data * 100

            output_str = 'train dataset : %d/%d epochs spend time : %.4f sec / total_loss : %.4f correct : %d/%d -> %.4f%%\n' \
                        % (epoch + 1, epochs, time.time() - start_time, train_total_loss,
                            train_total_count, train_total_data, train_accuracy)
            sys.stdout.write(output_str)
            check_file.write(output_str)

            # check validation dataset
            start_time = time.time()
            FeatureExtract.eval()
            Classification.eval()

            for index, data in enumerate(val_dataloader):
                batch_signal, batch_label = data
                batch_signal = batch_signal.to(device2)
                batch_label = batch_label.long().to(device2)

                with torch.no_grad():
                    pred = model(batch_signal)

                    loss = loss_fn(pred, batch_label)

                    # acc
                    _, predict = torch.max(pred, 1)
                    check_count = (predict == batch_label).sum().item()

                    val_total_loss += loss.item()
                    val_total_count += check_count
                    val_total_data += len(batch_signal)

                    # # 사용하지 않는 변수 제거
                    # del (batch_signal)
                    # del (batch_label)
                    # del (loss)
                    # del (pred)
                    # torch.cuda.empty_cache()

            val_total_loss /= index
            val_accuracy = val_total_count / val_total_data * 100

            output_str = 'val dataset : %d/%d epochs spend time : %.4f sec  / total_loss : %.4f correct : %d/%d -> %.4f%%\n' \
                        % (epoch + 1, epochs, time.time() - start_time, val_total_loss,
                            val_total_count, val_total_data, val_accuracy)
            sys.stdout.write(output_str)
            check_file.write(output_str)

            # scheduler.step(float(val_total_loss))

            if epoch == 0:
                best_accuracy = val_accuracy
                best_epoch = epoch
                save_file = save_filename
                torch.save(model.state_dict(), save_file)
                stop_count = 0
            else:
                if best_accuracy < val_accuracy:
                    best_accuracy = val_accuracy
                    best_epoch = epoch
                    save_file = save_filename
                    torch.save(model.state_dict(), save_file)
                    stop_count = 0
                else:
                    stop_count += 1
            if stop_count > stop_iter:
                print('Early Stopping')
                break

            output_str = 'best epoch : %d/%d / val accuracy : %f%%\n' \
                        % (best_epoch + 1, epochs, best_accuracy)
            sys.stdout.write(output_str)
            print('=' * 30)

    output_str = 'best epoch : %d/%d / accuracy : %f%%\n' \
                 % (best_epoch + 1, epochs, best_accuracy)
    sys.stdout.write(output_str)
    check_file.write(output_str)
    print('=' * 30)

    check_file.close()


def training_rnn_dataloader_cuda1():
    model_load_path = '/home/eslab/kdy/git/Hallym_SleepStage/saved_model/Hallym_dataset/cnn/'
    model_save_path = '/home/eslab/kdy/git/Hallym_SleepStage/saved_model/Hallym_dataset/rnn/'
    logging_save_path = '/home/eslab/kdy/git/Hallym_SleepStage/log/Hallym_dataset/rnn/'

    os.makedirs(model_save_path,exist_ok=True)
    os.makedirs(logging_save_path,exist_ok=True)

    signals_path = '/home/ssd1/dataset/Hallym_npy/3channel_info_shuffle/train_dataloader/'
    
    #test_signal_dir = '/home/jglee/medical_dataset/Seoul_medicalDataset_npy/C3M2/EEG/test_each_preprocessing/'
    k_fold = 10

    dataset_list = os.listdir(signals_path)
    training_fold_list = []
    validation_fold_list = []
    for i in range(0,1):
        for folder_index, folder_name in enumerate(dataset_list):
            if folder_index % k_fold != 0 or folder_index == 0:
                training_fold_list.append(folder_name)
            else:
                validation_fold_list.append(folder_name)
    print(len(training_fold_list))
    print(len(validation_fold_list))
    # exit(1)
    epochs = 3000
    batch_size = 400

    preprocessing=False
    preprocessing_methods = 'Standard'

    learning_rate = 0.001
    loss_function = 'CE'
    optim = 'Adam'
    sequence_length = 20

    #sequence_length = 10 / batch_size = 800 -> 23.3GB (GPU Memory)
    use_noise = False
    epsilon = 0.8
    noise_scale = 2e-6
    scheduler_list = ['WarmUp_restart_gamma'] # 'WarmUp_restart'
    for scheduler in scheduler_list:
        load_filename = model_load_path + 'ResNet_ensemble_branch_dataloaer_hallym_%.4f_%s_%s_%s_FE.pth'%(learning_rate,optim,loss_function,scheduler)
        save_filename = model_save_path + 'ResNet_ensemble_branch_dataloaer_hallym_%.4f_%s_%s_%s_lstm_%d.pth'%(learning_rate,optim,loss_function,scheduler,sequence_length)
        logging_filename = logging_save_path + 'ResNet_ensemble_branch_dataloaer_hallym_%.4f_%s_%s_%s_lstm_%d..txt'%(learning_rate,optim,loss_function,scheduler,sequence_length)

        train_rnn_dataloader_cuda1(save_filename=save_filename,logging_filename=logging_filename,load_filename=load_filename,signals_path=signals_path, train_dataset_list=training_fold_list,val_dataset_list=validation_fold_list,
                                                    epochs=epochs,batch_size=batch_size,learning_rate=learning_rate,
                                            optim=optim,loss_function=loss_function,epsilon=epsilon,noise_scale=noise_scale,
                                            use_noise=use_noise,preprocessing=preprocessing,preprocessing_methods=preprocessing_methods,scheduler=scheduler,warmup_iter=5,cosine_decay_iter=15,stop_iter=50,sequence_length=sequence_length
                                        )

