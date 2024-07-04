"""
Define the pytorch dataset for image-text pairs
"""

import os
import json
import numpy as np
import imageio.v3 as imageio
import torch

from typing import Literal
from torch.utils.data import Dataset
from transformers import CLIPProcessor

class CLIPDataset(Dataset):
    
    def __init__(
        self,
        root,
        processor: CLIPProcessor,
        split: Literal['train_raw', 'train_sub', 'train_all'] = 'train_raw'
    ):
        
        super().__init__()
        self.data_root = root
        self.split = split
        
        if split != 'train_all':
            img_paths, captions, img_box_paths, clues = self.load_data_for_split(split)
        else:
            img_paths_raw, captions_raw = self.load_data_for_split('train_raw')
            img_paths_sub, captions_sub = self.load_data_for_split('train_sub')
            img_paths = img_paths_raw + img_paths_sub
            captions = captions_raw + captions_sub
        
        self.img_paths = img_paths
        self.captions = captions

        self.img_box_paths = img_box_paths
        self.clues  = clues
        
        self.processor = processor

    def load_data_for_split(self, split):
        pairs_path = os.path.join(self.data_root, split, 'img_text_pair.json')
        with open(pairs_path, 'r') as f:
            pairs = json.load(f)
        img_paths = []
        captions = []

        img_box_paths = []
        clues = []
        for pair in pairs:
            img_path = os.path.join(self.data_root, split, 'imgs', pair['img'])
            img_paths.append(img_path)
            img_box_path = os.path.join(self.data_root, split, 'imgs', pair['img_box'])
            img_box_paths.append(img_box_path)
            captions.append(pair['caption'])
            clues.append(pair['clue'])
        return img_paths, captions, img_box_paths, clues

        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        caption = self.captions[index]
        
        img_box_path = self.img_box_paths[index]
        clue = self.clues[index]


        image = imageio.imread(img_path).astype(np.float32) / 255.0
        image_box = imageio.imread(img_box_path).astype(np.float32) / 255.0
        
        # To CHW
        if image.ndim == 2: # gray scale or binary
            image = np.stack([image]*3)
        else:
            image = image.transpose(2, 0, 1)

        # To CHW
        if image_box.ndim == 2: # gray scale or binary
            image_box = np.stack([image_box]*3)
        else:
            image_box = image_box.transpose(2, 0, 1)
        # image = torch.from_numpy(image) / 255.0 # scale to (0, 1)
        
        if np.random.rand() > 0.5:
            use_text = caption
        else:
            use_text =clue

        if np.random.rand() > 0.5:
            use_image = image
        else:
            use_image = image_box

        inputs = self.processor(
            text=use_text,
            images=use_image,
            return_tensors='pt',
            padding='max_length',
            truncation='longest_first'
        )
        inputs = {k: v[0] for k, v in inputs.items()}

       # inputs_2 = self.processor(
       #     text=clue,
       #     images=image_box,
       #     return_tensors='pt',
       #     padding='max_length',
       #     truncation='longest_first'
       # )
       # inputs_2 = {k: v[0] for k, v in inputs_2.items()}
        
        return inputs
        
