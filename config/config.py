PARA = dict(
    GA_params=dict(
        MAXGEN = 10, #进化代数
        maxormins = -1, #表示目标函数最小最大化标记，1表示最小化，-1表示最大化
        xov_rate = 0.7, #交叉概率
        mut_rate = 0.05, #变异概率
        Nind = 10, #种群大小
        epoch=20,
    ),

    CNN_params = dict(
        conv_num=[1,5],
        conv_output=[1,4],
        fc_num=[1,4],
        fc_output=[1,8],

        save_x_mat='./cache/data/MyCNN/X.mat',
        save_y_mat='./cache/data/MyCNN/Y.mat',
        save_rand_x_mat='./cache/data/MyCNN/Rand_X.mat',
        save_rand_y_mat='./cache/data/MyCNN/Rand_Y.mat',
        save_data_path='./cache/data/MyCNN/',
        save_data_txt='./cache/data/MyCNN/SaveData.txt',
        save_bestdata_txt='./cache/data/MyCNN/BestData.txt',
        checkpoint_path='./cache/checkpoint/MyCNN/',
    ),

    mnist_params=dict(
        root = '../../DATASET/Mnist',
        in_dim = 28*28,
        out_dim = 10,
        hidden_layers = 3,
        hidden_neurons = [16,32,64,64],
    ),
    train=dict(
        epochs = 20,
        batch_size = 64,
        lr = 0.01,
        momentum=0.9,
        wd = 5e-4,
        num_workers = 2,
        divice_ids = [1],
        gpu_id = 0,
        num_classes=10,
    ),
    test=dict(
        batch_size=64
    ),
    cifar10_paths = dict(
        validation_rate = 0.05,

        root = '../../DATASET/cifar10/',

        original_trainset_path = '../../../DATASET/cifar10/cifar-10-python/',#train_batch_path
        original_testset_path = '../../../DATASET/cifar10/cifar-10-python/',

        after_trainset_path = '../../../DATASET/cifar10/trainset/',
        after_testset_path = '../../../DATASET/cifar10/testset/',
        after_validset_path = '../../../DATASET/cifar10/validset/',

        train_data_txt = '../../../DATASET/cifar10/train.txt',
        test_data_txt = '../../../DATASET/cifar10/test.txt',
        valid_data_txt = '../../../DATASET/cifar10/valid.txt',
    ),
    utils_paths = dict(
        checkpoint_path = './cache/checkpoint/',
        log_path = './cache/log/',
        visual_path = './cache/visual/',
        params_path = './cache/params/',
    ),
)