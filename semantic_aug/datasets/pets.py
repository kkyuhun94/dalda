from semantic_aug.few_shot_dataset import FewShotDataset
from semantic_aug.generative_augmentation import GenerativeAugmentation
from typing import Tuple

import numpy as np
import torchvision.transforms as transforms
import torch
import os

from PIL import Image
from collections import defaultdict


PET_DIR = "datasets/oxford_pet"

TRAIN_LABEL_SET = os.path.join(
    PET_DIR, "annotations/trainval.txt")
VAL_LABEL_SET = os.path.join(
    PET_DIR, "annotations/test.txt")


class PetsDataset(FewShotDataset):

    class_names = ['Abyssinian',
                    'american bulldog',
                    'american pit bull terrier',
                    'basset hound',
                    'beagle',
                    'Bengal',
                    'Birman',
                    'Bombay',
                    'boxer',
                    'British Shorthair',
                    'chihuahua',
                    'Egyptian Mau',
                    'english cocker spaniel',
                    'english setter',
                    'german shorthaired',
                    'great pyrenees',
                    'havanese',
                    'japanese chin',
                    'keeshond',
                    'leonberger',
                    'Maine Coon',
                    'miniature pinscher',
                    'newfoundland',
                    'Persian',
                    'pomeranian',
                    'pug',
                    'Ragdoll',
                    'Russian Blue',
                    'saint bernard',
                    'samoyed',
                    'scottish terrier',
                    'shiba inu',
                    'Siamese',
                    'Sphynx',
                    'staffordshire bull terrier',
                    'wheaten terrier',
                    'yorkshire terrier'] 

    num_classes: int = len(class_names)

    def __init__(self, *args, split: str = "train", seed: int = 0, 
                 train_image_set: str = TRAIN_LABEL_SET, 
                 val_image_set: str = VAL_LABEL_SET, 
                 image_dir: str = PET_DIR, 
                 examples_per_class: int = None, 
                 generative_aug: GenerativeAugmentation = None, 
                 synthetic_probability: float = 0.5,
                 image_size: Tuple[int] = (256, 256), 
                 inference = False, **kwargs):

        super(PetsDataset, self).__init__(
            *args, examples_per_class=examples_per_class,
            synthetic_probability=synthetic_probability, 
            generative_aug=generative_aug, inference = inference, **kwargs)

        image_set = {"train": train_image_set, "val": val_image_set}[split]

        class_to_images = defaultdict(list)

        with open(image_set, 'r') as file:
            for line in file:
                image_path = os.path.join(image_dir, line.strip().split(' ')[0] + '.jpg')
                class_idx = int(line.strip().split(' ')[1])  - 1
                class_name = self.class_names[class_idx]

                class_to_images[class_name].append(image_path)


        rng = np.random.default_rng(seed)
        class_to_ids = {key: rng.permutation(
            len(class_to_images[key])) for key in self.class_names}
        
        class_to_ids = {key: np.array_split(class_to_ids[key], 2)[0 if split == "train" else 1] for key in self.class_names}

        if examples_per_class is not None:
            class_to_ids = {key: ids[:examples_per_class] 
                            for key, ids in class_to_ids.items()}

        self.class_to_images = {
            key: [class_to_images[key][i] for i in ids] 
            for key, ids in class_to_ids.items()}

        self.all_images = sum([
            self.class_to_images[key] 
            for key in self.class_names], [])

        self.all_labels = [i for i, key in enumerate(
            self.class_names) for _ in self.class_to_images[key]]

        self.all_labels = [i for i, key in enumerate(
            self.class_names) for _ in self.class_to_images[key]]

        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15.0),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x.expand(3, *image_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5])
        ])

        val_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x.expand(3, *image_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5])
        ])

        self.transform = {"train": train_transform, "val": val_transform}[split]

    def __len__(self):
        
        return len(self.all_images)

    def get_image_by_idx(self, idx: int) -> Image.Image:

        return Image.open(self.all_images[idx]).convert('RGB')

    def get_label_by_idx(self, idx: int) -> int:

        return self.all_labels[idx]
    
    def get_metadata_by_idx(self, idx: int) -> dict:

        return dict(name=self.class_names[self.all_labels[idx]])