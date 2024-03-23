#CUDA_VISIBLE_DEVICES=1 python run.py --config log_TH1K/finetune-th1k-spade.yml --device_ids 0 --checkpoint log_TH1K/00000001-checkpoint.pth.tar
import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread
from functools import partial
from skimage.transform import resize


import torch
import random
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from augmentation import AllAugmentationTransform
import glob
import math

import pickle
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor



def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """
    
    if os.path.isdir(name):
        
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
        
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None):
        self.root_dir = root_dir
        
        tmp_file = open(root_dir + 'train_file_list.pickle','rb')
        self.train_files_list = pickle.load(tmp_file)
        
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
                # train_videos = {os.path.basename(video).split('#')[0] for video in
                #                 os.listdir(os.path.join(root_dir, 'train'))}
                # train_videos = list(train_videos)
                train_videos = list(self.train_files_list.keys())
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            test_videos = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)

            #### for degradation ####
            

            self.kernel_range = [2 * v + 1 for v in range(1,3)]
            self.pulse_tensor = torch.zeros(11, 11).float()
            self.pulse_tensor[5, 5] = 1

            self.resize_range = [0.15, 1.5]

            # blur settings for the first degradation
            self.blur_kernel_size = 7
            self.kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
            self.kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]  # a list for each kernel probability
            self.blur_sigma = [0.1, 0.5]
            self.betag_range = [0.2, 1] # betag used in generalized Gaussian blur kernels
            self.betap_range = [0.5, 1.2] # betap used in plateau blur kernels
            self.sinc_prob = 0.1  # the probability for sinc filters

            # blur settings for the second degradation
            self.blur_kernel_size2 = 7
            self.kernel_list2 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
            self.kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
            self.blur_sigma2 = [0.1, 0.5]
            self.betag_range2 = [0.2, 1]
            self.betap_range2 = [1, 1.2]
            self.sinc_prob2 = 0.1
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
            # name = self.videos[idx]
            # path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
            name = self.videos[idx]
            choice_list = self.train_files_list[name]
            # if len(choice_list) == 0:
            #     name = self.videos[idx-1]
            #     choice_list = self.train_files_list[name]
            paths = np.random.choice(choice_list)
        else:
            name = self.videos[idx]
            paths = os.path.join(self.root_dir, name)

        video_name = os.path.basename(paths)
        if self.is_train and os.path.isdir(paths):
            frames = os.listdir(paths)
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))


            if self.frame_shape is not None:
                resize_fn = partial(resize, output_shape=self.frame_shape)
            else:
                resize_fn = img_as_float32
            video_array = [resize_fn(img_as_float32(io.imread(paths + '/' + '%06d.jpg'%(idx) ))) for idx in frame_idx]


        else:
            video_array = read_video(paths, frame_shape=self.frame_shape)
            num_frames = len(video_array)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(
                num_frames)
            video_array = video_array[frame_idx]

        if self.transform is not None:
            video_array = self.transform(video_array)

        out = {}
        if self.is_train:
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')
            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))

            # if self.degradation:
            ############ run degradation ############
            # ---- Generate kernels (used in the first degradation) ---- #
            kernel_size = random.choice(self.kernel_range)
            if np.random.uniform() <  0.1:
                # this sinc filter setting is for kernels ranging from [7, 21]
                if kernel_size < 11:
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            else:
                kernel = random_mixed_kernels(
                    self.kernel_list,
                    self.kernel_prob,
                    kernel_size,
                    self.blur_sigma,
                    self.blur_sigma, [-math.pi, math.pi],
                    self.betag_range,
                    self.betap_range,
                    noise_range=None)
            # pad kernel
            pad_size = (21 - kernel_size) // 2
            kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
            

            # ----- Generate kernels (used in the second degradation) ---- #
            kernel_size = random.choice(self.kernel_range)
            if np.random.uniform() < 0.1:
                if kernel_size < 13:
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            else:
                kernel2 = random_mixed_kernels(
                    self.kernel_list2,
                    self.kernel_prob2,
                    kernel_size,
                    self.blur_sigma2,
                    self.blur_sigma2, [-math.pi, math.pi],
                    self.betag_range2,
                    self.betap_range2,
                    noise_range=None)
            # pad kernel
            pad_size = (21 - kernel_size) // 2
            kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))
            
            # ---- the final sinc kernel ---- #
            if np.random.uniform() < 0.8:
                kernel_size = random.choice(self.kernel_range)
                omega_c = np.random.uniform(np.pi / 3, np.pi)
                sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=11)
                sinc_kernel = torch.FloatTensor(sinc_kernel)
            else:
                sinc_kernel = self.pulse_tensor

            # BGR to RGB, HWC to CHW, numpy to tensor
            # img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
            kernel = torch.FloatTensor(kernel)
            kernel2 = torch.FloatTensor(kernel2)
            #########################################

            out['kernel'] = kernel
            out['kernel2']= kernel2
            out['sinc_kernel'] = sinc_kernel

        else:
            video = np.array(video_array, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))

        out['name'] = video_name

        return out


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]
