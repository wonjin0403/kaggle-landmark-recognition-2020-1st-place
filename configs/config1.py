
import os
import albumentations as A
abs_path = os.path.dirname(__file__)

args = {
    'model_path':'../models/',
    'data_path':'/kaggle/input/landmark-recognition-2020/',
    'data_path_2019':'/kaggle/input/landmark-recognition-2020/',
    'valid_csv_fn':'train.csv',
    'train_csv_fn':'train.csv',

    'gpus':'0',
    'filter_warnings':True,
    'logger': 'neptune',
    'num_sanity_val_steps': 0,

    'distributed_backend': 'ddp',
    'channels_last':False,

    'gradient_accumulation_steps':2,
    'precision':16,
    'sync_batchnorm':False,
    
    'seed':1138,
    'num_workers':4,
    'save_weights_only':True,

    'p_trainable': True,

    'resume_from_checkpoint': None,
    'pretrained_weights': None,

    'normalization':'imagenet',
    'crop_size':224,

    'backbone':'gluon_seresnext101_32x4d',
    'embedding_size': 512,
    'pool': 'gem',
    'arcface_s': 45,
    'arcface_m': 0.4,

    'neck': 'option-D',
    'head':'arc_margin',

    'crit': "bce",
    'loss':'arcface',
    #'focal_loss_gamma': 2,
    'class_weights': "log",
    'class_weights_norm' :'batch',
    
    'optimizer': "sgd",
    'weight_decay':1e-4,
    'lr': 0.05,
    'batch_size': 16,
    'max_epochs': 10,
    'scheduler': {"method":"cosine","warmup_epochs": 1},
    

    'n_classes':81313,
    'data_frac':1.,

    'neptune_project':'xx/kaggle-landmark',
}

args['tr_aug'] = A.Compose([
    A.SmallestMaxSize(224),
    A.RandomCrop(height=args['crop_size'],width=args['crop_size'],p=1.),
    A.HorizontalFlip(p=0.5),
    ])

args['val_aug'] = A.Compose([
    A.SmallestMaxSize(224),
    A.CenterCrop(height=args['crop_size'],width=args['crop_size'],p=1.)
])
