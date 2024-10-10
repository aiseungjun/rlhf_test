import os
import glob
import numpy as np
import torch
from torch.utils import data
import random

class denodata(data.Dataset):
    def __init__(self, config, mode='train') -> None:
        self.dataset_spec = config['dataset']
        self.mode = mode
        self.data_dir = self.dataset_spec['data_dir']
        self.dt_list = []
        self.gt_list = []

        self._scan()
        print(f"Total Images: {len(self.gt_list)}")

    def _scan(self):
        """Dataset path scanning logic"""
        if self.mode == 'train':
            patient_id = ['L067', 'L096', 'L143', 'L192', 'L291', 'L310', 'L333', 'L506']
            for idx in patient_id:
                gt_list = glob.glob(os.path.join(self.data_dir, idx, 'gt', '*.npy'))
                gt_list.sort()
                self.gt_list.extend(gt_list)
                
                dt_list = glob.glob(os.path.join(self.data_dir, idx, 'distorted', '*.npy'))
                dt_list.sort()
                self.dt_list.extend(dt_list)

        elif self.mode == 'test':
            patient_id = ['L109', 'L286']
            distortion_list = ['10_180', '10_360', '10_720', '25_180', '25_360', '25_720', '50_180', '50_360', '50_720', '100_180', '100_360', '100_720']
            
            for idx in patient_id:
                gt_list = glob.glob(os.path.join(self.data_dir, idx, 'gt', '*.npy'))
                gt_list.sort()
                self.gt_list.extend(gt_list)
                
                for gt_img in gt_list:
                    dt_per_gt = []
                    for distortion in distortion_list:
                        distortion_file = os.path.join(self.data_dir, idx, distortion, os.path.basename(gt_img))
                        if not os.path.exists(distortion_file):
                            raise FileNotFoundError(f"Distortion file not found: {distortion_file}")
                        dt_per_gt.append(distortion_file)
                    self.dt_list.append(dt_per_gt)

        else:
            raise ValueError(f"Invalid mode: {self.mode}. Expected 'train' or 'test'.")

    def normalize(self, img, width=350, level=40):
        disp_window = [level - width/2, level + width/2]
        img = img.clip(disp_window[0], disp_window[1])
        img = (img - disp_window[0]) / (disp_window[1] - disp_window[0])
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        return img

    def augment(self, ldct, ndct, hflip=True, vflip=True, rot=True):
        hflip = self.dataset_spec['augment'] and hflip and (random.random() < 0.5)
        vflip = self.dataset_spec['augment'] and vflip and (random.random() < 0.5)
        rot90 = self.dataset_spec['augment'] and rot * random.randint(0, 4)

        if hflip:
            ldct = torch.flip(ldct, [2])
            ndct = torch.flip(ndct, [2])
        if vflip:
            ldct = torch.flip(ldct, [1])
            ndct = torch.flip(ndct, [1])

        ldct = torch.rot90(ldct, rot90, [1, 2])
        ndct = torch.rot90(ndct, rot90, [1, 2])
        return ldct, ndct

    def __getitem__(self, index):
        if self.mode == 'train':
            gt_img = self.normalize(np.load(self.gt_list[index]))
            dis_img = self.normalize(np.load(self.dt_list[index]))
            dis_img, gt_img = self.augment(dis_img, gt_img)
             # Negative stride 문제 해결
            dis_img = dis_img.contiguous()
            gt_img = gt_img.contiguous()
            return dis_img, gt_img
        
        elif self.mode == 'test':
            gt_img = self.normalize(np.load(self.gt_list[index]))
            distorted_list = self.dt_list[index]
            dis_imgs = [self.normalize(np.load(dis_img_path)) for dis_img_path in distorted_list]
            return dis_imgs, gt_img

    def __len__(self):
        return len(self.gt_list)
