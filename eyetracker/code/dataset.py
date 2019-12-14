import torch

from os import listdir
from os.path import join
from PIL import Image
import pandas as pd
from skimage import io, transform
import numpy as np
import time

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


class DepthGazeDataset(torch.utils.data.Dataset):
    """Our dataset with gaze depth information."""
    def __init__(self, csv_file, transform):
        """
        Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
        on a sample.
        eye_cams (list): [1,2,8,6,4,7] for 6 cameras, [1,2,6,4] for 4 cameras        
        """
        self.eye_frame = pd.read_csv(csv_file)
        self.root_dir = csv_file[:csv_file.rfind('/')]
        self.transform = transform

    def __len__(self):
        return len(self.eye_frame)

    def __getitem__(self, idx):
        user_name = self.eye_frame.iloc[idx, 1]
        spot_idx = self.eye_frame.iloc[idx, -4]
        image_names = self.eye_frame.iloc[idx, 0]

        image_path = join(self.root_dir, user_name, "spot%d"%(spot_idx+1), image_names)
        image = Image.open(image_path)
        image = self.transform(image) # C X H X W

        gaze = self.eye_frame.iloc[idx, -3:].values
        gaze = gaze.astype('float')
        sample = {'images': image, 'gaze': gaze, 'name': [user_name, "spot%d"%spot_idx, image_names]}

        return sample
