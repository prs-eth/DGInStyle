# Author: Yuru Jia
# Last Modified: 2023-10-19

import random
import json
import os.path as osp

import numpy as np
from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset

from .tools.training_classes import get_class_stacks, make_one_hot, get_label_stats, get_rcs_class_probs

class CityScapesDataset(Dataset):
    def __init__(self, args, tokenizer):
        super(CityScapesDataset, self).__init__()

        self.file_path = args.dataset_file
        self.tokenizer = tokenizer
        self.crop_size = args.resolution
        self.rcs_enabled = args.rcs_enabled
        self.random_crop_enabled = args.random_crop_enabled
        self.resize_ratios = args.resize_ratios
    
        if self.rcs_enabled:
            self.rcs_class_temp = 0.01
            self.rcs_min_crop_ratio = 0.5
            self.rcs_min_pixels = 3000

            self.rcs_classes, self.rcs_classprob = get_rcs_class_probs(
                args.rcs_data_root, self.rcs_class_temp)

            with open(
                osp.join(args.rcs_data_root, 'samples_with_class.json'), 'r') as of:
                samples_with_class_and_n = json.load(of)
            
            samples_with_class_and_n = { int(k): v 
                                        for k, v in samples_with_class_and_n.items() 
                                        if int(k) in self.rcs_classes
            }

            self.samples_with_class = {}
            for c in self.rcs_classes:
                self.samples_with_class[c] = []
                for file, pixels in samples_with_class_and_n[c]:
                    if pixels > self.rcs_min_pixels:
                        self.samples_with_class[c].append(file)
                assert len(self.samples_with_class[c]) > 0


        self.img_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.conditioning_img_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.data = []

        with open(self.file_path, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def get_crop(self, img, label_map):
        img = Image.open(img).convert("RGB")
        label_map = Image.open(label_map)

        w, h = img.size
        # resize image
        if self.resize_ratios is not None:
            resize_ratio = random.choice(self.resize_ratios)
            new_w, new_h = int(w * resize_ratio), int(h * resize_ratio)
            img = img.resize((new_w, new_h), Image.BILINEAR)
            label_map = label_map.resize((new_w, new_h), Image.NEAREST)
            w, h = new_w, new_h
        
        if self.random_crop_enabled:
            x1 = random.randint(0, w -  self.crop_size)
            y1 = random.randint(0, h -  self.crop_size)
        else:
            x1 = (w -  self.crop_size) // 2
            y1 = (h -  self.crop_size) // 2

        crop_coords = (x1, y1, x1 +  self.crop_size, y1 +  self.crop_size)

        new_img = img.crop(crop_coords)
        new_condition_img = label_map.crop(crop_coords)          
        new_texts = get_class_stacks(new_condition_img)
            
        return new_img, new_condition_img, new_texts
    

    def get_crop_rcs(self, img, label_map, c):
        img = Image.open(img).convert("RGB")
        label_map = Image.open(label_map)
        
        w, h = img.size
        # resize image
        if self.resize_ratios is not None:
            resize_ratio = random.choice(self.resize_ratios)
            new_w, new_h = int(w * resize_ratio), int(h * resize_ratio)
            img = img.resize((new_w, new_h), Image.BILINEAR)
            label_map = label_map.resize((new_w, new_h), Image.NEAREST)
            w, h = new_w, new_h

        indices = np.where(np.array(label_map) == c)

        idx = random.randint(0, len(indices[0]) - 1)
        y, x = indices[0][idx], indices[1][idx]
        x1 = min(max(0, x - self.crop_size // 2), w - self.crop_size)
        y1 = min(max(0, y - self.crop_size // 2), h - self.crop_size)
        x2 = x1 + self.crop_size
        y2 = y1 + self.crop_size
        
        new_img = img.crop((x1, y1, x2, y2))
        new_condition_img = label_map.crop((x1, y1, x2, y2))   
        new_texts = get_class_stacks(new_condition_img)        
          
        return new_img, new_condition_img, new_texts
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.rcs_enabled:
            c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
            label_id_file = np.random.choice(self.samples_with_class[c])
            img_file = label_id_file.replace("gtFine", "leftImg8bit").replace("_labelTrainIds", "")

            crop_img, crop_condition_img, caption = self.get_crop_rcs(img_file, label_id_file, c)
        else:
            item = self.data[idx]
            img_file = item["image"]
            mask_file = item["conditioning_image"]
            label_id_file = mask_file.replace("color", "labelTrainIds")

            crop_img, crop_condition_img, caption = self.get_crop(img_file, label_id_file)


        # get label statistics for cropped image
        label_stats = get_label_stats(crop_condition_img)

        # process cropped image label into one-hot encoding
        crop_condition_img = make_one_hot(crop_condition_img)

        crop_img = self.img_transforms(crop_img)
        crop_condition_img = self.conditioning_img_transforms(crop_condition_img)
    
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids = inputs.input_ids[0]


        return dict(pixel_values=crop_img, 
                    conditioning_pixel_values=crop_condition_img, 
                    input_ids=input_ids,
                    label_stats=label_stats)
    

  




    



