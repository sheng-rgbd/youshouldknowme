

import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

C.seed = 12345

remoteip = os.popen('pwd').read()
C.root_dir = os.path.abspath(os.path.join(os.getcwd(), './'))
C.abs_dir = osp.realpath(".")





# # SUNRGBD_Dataset config
# """Dataset Path"""
# C.dataset_name = 'SUNRGBD'
# C.dataset_path = osp.join(C.root_dir, 'datasets', 'SUNRGBD')
# C.rgb_root_folder = osp.join(C.dataset_path, 'RGB')
# C.rgb_format = '.jpg'
# C.gt_root_folder = osp.join(C.dataset_path, 'labels')
# C.gt_format = '.png'
# C.gt_transform = True
# # True when label 0 is invalid, you can also modify the function _transform_gt in dataloader.RGBXDataset
# # True for most dataset valid, Faslse for MFNet(?)
# C.x_root_folder = osp.join(C.dataset_path, 'Depth')
# C.x_format = '.png'
# C.x_is_single_channel = False # True for raw depth, thermal and aolp/dolp(not aolp/dolp tri) input
# C.train_source = osp.join(C.dataset_path, "train1.txt")
# C.eval_source = osp.join(C.dataset_path, "test1.txt")
# C.is_test = False
# C.num_train_imgs = 5284
# C.num_eval_imgs = 5049
# C.num_classes = 37
# C.class_names =  ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter',
#                    'blinds','desk','shelves','curtain','dresser','pillow','mirror','floor_mat','clothes','ceiling','books',
#                    'fridge','tv','paper','towel','shower_curtain','box','whiteboard','person','night_stand','toilet','sink','lamp','bathtub','bag']



# # NYU_Dataset config
# """Dataset Path"""
# C.dataset_name = 'NYUDepthv2'
# C.dataset_path = osp.join(C.root_dir, 'datasets', 'NYUDepthv2')
# C.rgb_root_folder = osp.join(C.dataset_path, 'RGB')
# C.rgb_format = '.jpg'
# C.gt_root_folder = osp.join(C.dataset_path, 'Label')
# C.gt_format = '.png'
# C.gt_transform = True
# # True when label 0 is invalid, you can also modify the function _transform_gt in dataloader.RGBXDataset
# # True for most dataset valid, Faslse for MFNet(?)
# C.x_root_folder = osp.join(C.dataset_path, 'HHA')
# C.x_format = '.jpg'
# C.x_is_single_channel = False # True for raw depth, thermal and aolp/dolp(not aolp/dolp tri) input
# C.train_source = osp.join(C.dataset_path, "train.txt")
# C.eval_source = osp.join(C.dataset_path, "test.txt")
# C.is_test = False
# C.num_train_imgs = 794
# C.num_eval_imgs = 654   
# C.num_classes = 40
# C.class_names =  ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds',
#     'desk','shelves','curtain','dresser','pillow','mirror','floor mat','clothes','ceiling','books','refridgerator',
#     'television','paper','towel','shower curtain','box','whiteboard','person','night stand','toilet',
#     'sink','lamp','bathtub','bag','otherstructure','otherfurniture','otherprop']


# # MFnet_Dataset config
"""Dataset Path"""
C.dataset_name = 'MFnet'
C.dataset_path = osp.join(C.root_dir, 'datasets', 'MFnet')
C.rgb_root_folder = osp.join(C.dataset_path, 'MF_rgb')
C.rgb_format = '.png'
C.gt_root_folder = osp.join(C.dataset_path, 'labels')
C.gt_format = '.png'
C.gt_transform = False
# True when label 0 is invalid, you can also modify the function _transform_gt in dataloader.RGBXDataset
# True for most dataset valid, Faslse for MFNet(?)
C.x_root_folder = osp.join(C.dataset_path, 'MF_depth3')
C.x_format = '.png'
C.x_is_single_channel = False # True for raw depth, thermal and aolp/dolp(not aolp/dolp tri) input
C.train_source = osp.join(C.dataset_path, "train_flip.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = False
C.num_train_imgs = 1568
C.num_eval_imgs = 393
C.num_classes = 9
C.class_names =  ['unlabeled','car','person','bike','curve','car_stop','guardrail','color_cone','bump']





# # Dataset config
# C.dataset_name = 'MFnet'
# C.dataset_path = osp.join(C.root_dir, 'datasets', 'MFnet')
# # C.dataset_name = 'NYUDepthv2'
# # C.dataset_path = osp.join(C.root_dir, 'datasets', 'NYUDepthv2')
# # C.rgb_root_folder = osp.join(C.dataset_path, 'RGB')
# C.rgb_root_folder = osp.join(C.dataset_path, 'MF_rgb')
# C.rgb_format = '.png'
# # C.gt_root_folder = osp.join(C.dataset_path, 'Label')
# C.gt_root_folder = osp.join(C.dataset_path, 'labels')
# C.gt_format = '.png'
# C.gt_transform = True
# C.x_root_folder = osp.join(C.dataset_path, 'HHA')
# C.x_format = '.jpg'
# C.x_is_single_channel = False
# C.train_source = osp.join(C.dataset_path, "train.txt")
# C.eval_source = osp.join(C.dataset_path, "test.txt")
# C.is_test = False
# C.num_train_imgs = 795
# C.num_eval_imgs = 654
# C.num_classes = 40
# C.class_names = [
#     'wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf',
#     'picture','counter','blinds','desk','shelves','curtain','dresser','pillow','mirror','floor mat',
#     'clothes','ceiling','books','refridgerator','television','paper','towel','shower curtain','box',
#     'whiteboard','person','night stand','toilet','sink','lamp','bathtub','bag','otherstructure',
#     'otherfurniture','otherprop']

# Image Config
C.background = 255
C.image_height = 480
C.image_width = 640
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

# Network Config
C.backbone = 'mit_b2'
C.pretrained_model = C.root_dir + '/pretrained/segformer/mit_b2.pth'
C.decoder = 'MLPDecoder'
C.decoder_embed_dim = 512
C.optimizer = 'AdamW'

# Train Config
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 8
C.nepochs = 500
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 16
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.warm_up_epoch = 10

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

# Eval Config
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1]
C.eval_flip = False
C.eval_crop_size = [480, 640]

# Store Config
C.checkpoint_start_epoch = 250
C.checkpoint_step = 25

# Dynamic Path Config

def get_env_or_default(env_name, default_path):
    return os.environ.get(env_name, default_path)

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
exp_dir = get_env_or_default('EXP_DIR', osp.abspath(f'./EXP/default_exp'))

C.exp_dir = exp_dir
C.log_dir = exp_dir
C.tb_dir = get_env_or_default('TB_DIR', osp.join(exp_dir, 'TB'))
C.checkpoint_dir = get_env_or_default('CKPT_DIR', osp.join(exp_dir, 'checkpoint'))
C.log_dir_link = C.log_dir

C.log_file = osp.join(exp_dir, 'logs', f'log_{exp_time}.log')
C.link_log_file = osp.join(exp_dir, 'logs', 'log_last.log')
C.val_log_file = osp.join(exp_dir, 'logs', f'val_{exp_time}.log')
C.link_val_log_file = osp.join(exp_dir, 'logs', 'val_last.log')

# Path Config

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(C.root_dir)

if __name__ == '__main__':
    print(config.nepochs)
    parser = argparse.ArgumentParser()
    parser.add_argument('-tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()

    if args.tensorboard:
        open_tensorboard()