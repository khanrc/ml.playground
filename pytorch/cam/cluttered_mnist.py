""" Cluttered MNIST Dataset """

import torch
import torch.utils.data
import Image # PIL
import numpy as np
import os


def load_image(path):
    image = Image.open(path)
    image = np.array(image).astype(np.float32)
    
    if image.ndim == 2:
        image = np.tile(image[:, :, np.newaxis], 3)

    return image


class ClutteredMNISTDataset(torch.utils.data.Dataset):
    train_fn = 'train.pt'
    test_fn = 'test.pt'
    raw_dir = 'raw'
    processed_dir = 'processed'

    def __init__(self, root, train, transform):
        super(ClutteredMNISTDataset, self).__init__()
        
        self.train_data = []
        self.test_data = []
        self.root = root
        self.train = train
        self.transform = transform
        self.N = 100000
        self.split = self.N * 0.9

        self.process()
        
    def process(self):
        """ process raw data and save processed file """
        # check already processed
        proc_dir = os.path.join(self.root, self.processed_dir)
        train_path = os.path.join(proc_dir, self.train_fn)
        test_path = os.path.join(proc_dir, self.test_fn)
        if os.path.exists(train_path) and os.path.exists(test_path):
            # already exists => load process file
            print("processed dataset already exists; load it")
            self.train_data = torch.load(train_path)
            self.test_data = torch.load(test_path)
            return

        # read and process raw data
        print("read and process raw dataset ...")
        label_path = os.path.join(self.root, self.raw_dir, "labels.txt")
        image_path_format = os.path.join(self.root, self.raw_dir, "img_{}.png")
        
        with open(label_path) as f:
            for line in f:
                if not line.strip():
                    break
                    
                idx, label = map(int, line.strip().split('\t'))
                image_path = image_path_format.format(idx)
                image = load_image(image_path)
                
                if idx <= self.split:
                    self.train_data.append((image, label))
                elif idx > self.split:
                    self.test_data.append((image, label))

        # write processed file
        if not os.path.exists(proc_dir):
            os.mkdir(proc_dir)

        with open(train_path, 'wb') as f:
            torch.save(self.train_data, f)
        with open(test_path, 'wb') as f:
            torch.save(self.test_data, f)

        print("Done!")
    
    def __getitem__(self, idx):
        if self.train:
            img, label = self.train_data[idx]
        else:
            img, label = self.test_data[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
