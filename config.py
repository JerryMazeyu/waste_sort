class Config(object):
    data_path = './data/'
    num_workers = 2
    batch_size = 5
    max_epoch = 1
    lr = 0.002
    weight_decay = 1e-4
    use_gpu = False
    print_freq = 20
    vis = True
    env = 'waste'
    net_path = './checkpoints/resnet34_finetune_wt.pth'
    id2class = {'cardboard': 0, 'glass': 1, 'metal': 2, 'paper': 3, 'plastic': 4, 'trash': 5}

opt = Config()