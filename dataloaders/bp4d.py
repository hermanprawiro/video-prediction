import os
import glob
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class BP4D_Micro(Dataset):
    def __init__(self, train=True, grayscale=False, frame_length=4, frame_stride=1, frame_dilation=0, transform=None):
        self.train = train
        self.transform = transform
        self.root = R"E:\Datasets\BP4D+\Thermal_Frames_Align"
        if grayscale:
            self.root += "_G"

        self.frame_length = frame_length
        self.frame_stride = frame_stride
        self.frame_dilation = frame_dilation

        self.subject_dict = self.__get_subject_dict()
        self.task_dict = self.__get_task_dict()
        self.filenames = self.__get_list()

    def __getitem__(self, idx):
        img_paths = self.filenames[idx]
        imgs = []
        for img_path in img_paths:
            img = Image.open(img_path)
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)

        subject = os.path.basename(os.path.dirname(os.path.dirname(img_paths[0])))
        subject_id = self.subject_dict[subject]
        gender_id = 0 if subject[0] == "F" else 1
        task_id = int(os.path.basename(os.path.dirname(img_paths[0])).replace("T", "")) - 1
        task_id = self.task_dict[task_id]

        return torch.stack(imgs, dim=1), task_id, subject_id, gender_id

    def __len__(self):
        return len(self.filenames)

    def __get_list(self):
        # Train : F011 - F015 and M011 - M015
        # Test : F016 - F020 and M016 - M020
        if self.train:
            subjects = ['F%03d' % (i+1) for i in range(5, 15)] + ['M%03d' % (i+1) for i in range(5, 15)]
        else:
            subjects = ['F%03d' % (i+1) for i in range(15, 20)] + ['M%03d' % (i+1) for i in range(15, 20)]
        tasks = ['T%d' % (i+1) for i in range(10)]
        
        filenames = []
        for subject in subjects:
            for task in tasks:
                files_in_folder = np.array(glob.glob(os.path.join(self.root, subject, task, "*.jpg")))
                max_files = len(files_in_folder)

                for j in self._getstartindices(max_files):
                    filenames.append(files_in_folder[self._getindices(j)].tolist())
        return filenames
    
    def _getstartindices(self, max_images):
        return list(range(0, max_images - self._getdilatedlength() + 1, self.frame_stride))

    def _getdilatedlength(self):
        return self.frame_length * (self.frame_dilation + 1) - self.frame_dilation

    def _getindices(self, start_id):
        return list(range(start_id, start_id + self._getdilatedlength(), self.frame_dilation + 1))

    def __get_subject_dict(self):
        subject_list = [f"F{item+1:03d}" for item in range(5, 20)] + [f"M{item+1:03d}" for item in range(5, 20)]
        return {value: i for i, value in enumerate(subject_list)}

    def __get_task_dict(self):
        return {
            0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 
            6: 1, 7: 2, 8: 3, 9: 4
        }


class BP4D_Single(Dataset):
    def __init__(self, subject='F001', task=0, grayscale=False, frame_length=4, frame_stride=1, frame_dilation=0, transform=None):
        self.subject_to_id, self.id_to_subject = self.__get_subject_dict()
        self.task_dict = self.__get_task_dict()
        self.root = R"E:\Datasets\BP4D+\Thermal_Frames_Align"
        if grayscale:
            self.root += "_G"
        if isinstance(subject, int):
            self.subject = self.id_to_subject[subject]
        else:
            self.subject = subject

        assert task < 10 and task >= 0
        self.task_id = task
        self.task = "T" + str(task + 1)
        
        self.transform = transform
        self.filenames = self.__get_list()

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        img = Image.open(img_path)
        subject_id = self.subject_to_id[self.subject]
        gender_id = 0 if self.subject[0] == "F" else 1
        task_id = self.task_dict[self.task_id]

        if self.transform is not None:
            img = self.transform(img)

        return img, task_id, subject_id, gender_id

    def __len__(self):
        return len(self.filenames)

    def __get_list(self):
        filenames = glob.glob(os.path.join(self.root, self.subject, self.task, "*.jpg"))
        return filenames

    def __get_subject_dict(self):
        subject_list = [f"F{item+1:03d}" for item in range(82)] + [f"M{item+1:03d}" for item in range(48)] + [f"M{item+1:03d}" for item in range(49, 58)]
        return {value: i for i, value in enumerate(subject_list)}, dict(enumerate(subject_list))

    def __get_task_dict(self):
        return {
            0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 
            6: 1, 7: 2, 8: 3, 9: 4
        }

if __name__ == "__main__":
    dset = BP4D_Micro(train=True, transform=torchvision.transforms.ToTensor())
    print(len(dset.filenames))
    print(dset[0][0].shape)