# Author: Yuru Jia. Last Modified: 05/01/2024

import json
import os.path as osp
from PIL import Image
import numpy as np
import torch


def get_cs_classes():
    """Cityscapes class names for external use."""
    return [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle'
    ]


def get_cs_palette():
    return  [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
            [0, 80, 100], [0, 0, 230], [119, 11, 32]]


def get_class_stacks(label_map):
    if isinstance(label_map, str):
        label_map = np.array(Image.open(label_map))
    elif isinstance(label_map, Image.Image):
        label_map = np.array(label_map)
    else:
        label_map = label_map

    labels = np.unique(label_map)
    cs_classes = get_cs_classes()
    sentence = [cs_classes[i] for i in labels if i!=255 and i!=19]
    sentence = " ".join(sentence)
    return sentence


def make_one_hot(label_map):
    label_map = np.array(label_map) if not isinstance(
        label_map, np.ndarray) else label_map
    num_classes = 20
    label_map[label_map == 255] = 19

    one_hot = np.eye(num_classes)[label_map]

    return one_hot


def get_rcs_class_probs(data_root, temperature):
    with open(osp.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {
        k: v
        for k, v in sorted(
            overall_class_stats.items(), key=lambda item: item[1])
    }
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)

    return list(overall_class_stats.keys()), freq.numpy()


def get_label_stats(label_map):
    label_map = np.array(label_map) if not isinstance(
        label_map, np.ndarray) else label_map
    labels = np.unique(label_map)
    cs_classes = get_cs_classes()
    label_stats = {}
    
    for i in range(len(cs_classes)):
        label_stats[cs_classes[i]] = np.sum(label_map == i)    
    label_stats["others"] = np.sum(label_map == 255)

    return label_stats


def map_label2RGB(label_map):
    """
    args:
        label_map: (H, W)
    return:
        color_map: (H, W, 3), numpy array    
    """
    
    label_map = np.array(label_map) if not isinstance(label_map, np.ndarray) else label_map
    palette = np.array(get_cs_palette())

    color_map = np.zeros((label_map.shape[0], label_map.shape[1], 3))
    for label in range(0, 19):
        color_map[label_map == label] = palette[label]

    color_map = color_map.astype(np.uint8)

    return color_map


def map_RGB2label(color_map):
    """
    args:
        color_map: (H, W, 3), numpy array        
    return:
        label_map: (H, W), numpy array
    """

    palette = np.array(get_cs_palette())
    palette_dict = {tuple(color): label for label, color in enumerate(palette)}

    label_map = np.zeros((color_map.shape[0], color_map.shape[1]), dtype=np.uint8)

    for y in range(color_map.shape[0]):
        for x in range(color_map.shape[1]):
            rgb = tuple(color_map[y, x])
            label_map[y, x] = palette_dict.get(rgb, 255)  # Default to label 255 if color not in palette

    return label_map