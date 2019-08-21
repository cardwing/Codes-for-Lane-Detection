import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class VOCAugDataSet(Dataset):
    def __init__(self, dataset_path='/home/houyuenan/remote/ApolloScapes/Codes-for-Lane-Detection/ENet-SAD-Pytorch/data/CULane/list', data_list='train', transform=None):

        with open(os.path.join(dataset_path, data_list + '.txt')) as f:
            self.img_list = []
            self.img = []
            self.label_list = []
            self.exist_list = []
            for line in f:
                self.img.append(line.strip().split(" ")[0])
                self.img_list.append(dataset_path.replace('/list', '') + line.strip().split(" ")[0])
                self.label_list.append(dataset_path.replace('/list', '') + line.strip().split(" ")[1])
                self.exist_list.append(np.array([int(line.strip().split(" ")[2]), int(line.strip().split(" ")[3]), int(line.strip().split(" ")[4]), int(line.strip().split(" ")[5])]))

        self.img_path = dataset_path
        self.gt_path = dataset_path
        self.transform = transform
        self.is_testing = data_list == 'test_img' # 'val'

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.img_path, self.img_list[idx])).astype(np.float32)
        label = cv2.imread(os.path.join(self.gt_path, self.label_list[idx]), cv2.IMREAD_UNCHANGED)
        exist = self.exist_list[idx]
        image = image[240:, :, :]
        label = label[240:, :]
        label = label.squeeze()
        if self.transform:
            image, label = self.transform((image, label))
            image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
            label = torch.from_numpy(label).contiguous().long()
        if self.is_testing:
            return image, label, self.img[idx]
        else:
            return image, label, exist
