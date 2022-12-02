from email.mime import image
import torch
from augmentations import randaugment
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import random
import torch.nn.functional as F
import numpy as np
from torchvision.io import read_image
from PIL import Image
from torchvision.datasets import ImageFolder
from pathlib import Path, PurePath
import numpy as np
import random
import cv2
import os
from cgi import test
import json
from config import load_config_from_yaml

# Divides the input image into patches of fixed size.
def extract_patches(image, patch_size_x, patch_size_y, num_patches):
    patches = []
    for i in range(0, image.shape[1], patch_size_x):
        for j in range(0, image.shape[2], patch_size_y):
            patch = image[:, i:i + patch_size_x, j:j + patch_size_y] 
            if patch.shape[1] != patch_size_x or patch.shape[2] != patch_size_y or len(patches) == num_patches:
                break
            else:
                patches.append(patch)
        if len(patches) == num_patches:
            break
    return patches

def show_image_cv2(image):
    cv2.imshow('image', image)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()
    for i in range(1, 5):
        cv2.waitKey(1)

def create_datasets(train_data, val_data, test_data, hyper_params):
    train_dataset = ArtHistoricalDocumentsDataset(
        data=train_data, 
        patch_size_x=hyper_params['patch_size_x'],
        patch_size_y=hyper_params['patch_size_y'],
        mode="train",
        ops=hyper_params['aug_num_ops'],
        mag=hyper_params['aug_mag'],
        numb_magnitudes=hyper_params['aug_num_mag'],
        num_classes = hyper_params['num_languages'],
        aug_prob=hyper_params['aug_prob'],
        num_patches=hyper_params['patch_size']
    ) if hyper_params['mode'].split('_')[0] == "training" else None  # only needs to be created when training

    val_dataset = ArtHistoricalDocumentsDataset(
        data=val_data, 
        patch_size_x=hyper_params['patch_size_x'],
        patch_size_y=hyper_params['patch_size_y'],
        mode="val",
        num_patches=hyper_params['patch_size'],
        num_classes = hyper_params['num_languages'],
    ) if hyper_params['mode'].split('_')[0] == "training" else None

    test_dataset = ArtHistoricalDocumentsDataset(
        data=test_data,
        patch_size_x=hyper_params['patch_size_x'],
        patch_size_y=hyper_params['patch_size_y'],
        mode="test",
        num_patches=hyper_params['patch_size'],
        num_classes = hyper_params['num_languages'],
    )

    print(f'SIZE of TRAIN      DATASET: {len(train_dataset) if hyper_params["mode"].split("_")[0] == "training" else None}')
    print(f'SIZE of VALIDATION DATASET: {len(val_dataset) if hyper_params["mode"].split("_")[0] == "training" else None}')
    print(f'SIZE of TEST       DATASET: {len(test_dataset)}')

    return train_dataset, val_dataset, test_dataset
    
def get_datasets(path, hyper_params):
    data = ImageFolder(root=path)
    wpi_classes = ['de', 'en', 'fr', 'nl']
    full_idxs = np.array(data.targets)
    classes = range(hyper_params['num_languages'])

    train_idxs = []
    val_idxs = []
    test_idxs = []

    # for few-shot training, use the list of unique filenames that belong to the few-shot
    # samples and assign the samples to the train idxs list
    if hyper_params['mode'] == 'training_few_shot' or hyper_params['mode'] == 'training_meta':
        few_shot_filenames = hyper_params[hyper_params['few_shot_n']]
        data_filenames = np.array([x[0].split('/')[-1] for x in data.imgs])
        
        for filename in few_shot_filenames:
            train_idxs.extend(np.where(data_filenames == filename)[0])

    for cls in classes:
        cls_idxs = np.where(full_idxs == cls)[0]
        num_samples = len(cls_idxs)

        # if we are doing few-shot training, remove the idxs that belong to the few-shot samples
        # otherwise remove the portion of idxs calculated via the train/val splitting
        # if we are doing meta training, we only want to extend the training id's by id's that belong to 
        # samples coming from the source domain, i.e. IMPACT. We make this sure by comparing the class number with the
        # corresponding class name and checking if the class name belongs to a WPI class.
        if hyper_params['mode'] != 'training_few_shot':
            if hyper_params['mode'] == 'training_meta':
                if list(data.class_to_idx.keys())[cls] not in wpi_classes:
                    train_ = np.random.choice(cls_idxs, replace=False, size=((int(hyper_params['split_train'] * num_samples)),))
                    train_idxs.extend(train_) 
            else:
                train_ = np.random.choice(cls_idxs, replace=False, size=((int(hyper_params['split_train'] * num_samples)),))
                train_idxs.extend(train_) 

        cls_idxs = [x for x in cls_idxs if x not in train_idxs]
        val_ = np.random.choice(cls_idxs, replace=False, size=((int(hyper_params['split_val'] * num_samples)),))
        test_ = np.array([x for x in cls_idxs if x not in val_])
        
        val_idxs.extend(val_)
        test_idxs.extend(test_)
    
    # use the _idxs lists to retrieve the actual data (image + label) from the original ImageFolder object
    train_data = [data.imgs[i] for i in train_idxs]
    val_data = [data.imgs[i] for i in val_idxs]
    test_data = [data.imgs[i] for i in test_idxs]

    train_dataset, val_dataset, test_dataset = create_datasets(train_data, val_data, test_data, hyper_params)
    return train_dataset, val_dataset, test_dataset, data.classes

# create a dataset class used for pytorch lightning
class ArtHistoricalDocumentsDataset(Dataset):
    def __init__(
        self, 
        data, 
        mode, 
        patch_size_x, 
        patch_size_y, 
        num_patches,
        num_classes, 
        aug_prob=None, 
        ops=None, 
        mag=None, 
        numb_magnitudes=None):

        self.data = data
        self.mode = mode
        self.patch_size_x = patch_size_x
        self.patch_size_y = patch_size_y
        self.num_patches = num_patches
        self.labels_one_hot = F.one_hot(torch.arange(num_classes)).float()
        self.ops = ops,
        self.mag = mag,
        self.numb_magnitudes = numb_magnitudes
        self.aug_prob = aug_prob

    def __len__(self):
        return len(self.data)

    def loader(self, path):
        """image = torch.FloatTensor(
            np.array(
                read_image(
                    path, mode=torchvision.io.ImageReadMode.RGB
                ),
                dtype=np.float32
            )
        )"""

        image = Image.open(path).convert('RGB')
        return image

    def __getitem__(self, index):
        # composition of transformations applied on each image
        transformations = transforms.Compose([
            # transform the PIL images to tensor.
            # ToTensor() Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
            transforms.ToTensor(),
            transforms.CenterCrop((1024,1024)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        image = self.loader(self.data[index][0])
        image_transformed = transformations(image)
        filename = self.data[index][0].split('/')[-1]
        label = self.labels_one_hot[self.data[index][1]]

        # augmentations will only be applied on train data
        if self.mode == "train":
            if random.random() < self.aug_prob:
                image_augmented = randaugment(
                    image, 
                    ops=self.ops, 
                    mag=self.mag, 
                    numb_magnitudes=self.numb_magnitudes
                )
                image_transformed = transformations(image_augmented)

        patches = extract_patches(
            image_transformed, 
            patch_size_x=self.patch_size_x, 
            patch_size_y=self.patch_size_y, 
            num_patches=self.num_patches
        )
        sample = {
            "image": torch.stack(patches, 0), 
            "label": label,
            "image_transformed": image_transformed, 
            "filename": filename
        }
        return sample
        