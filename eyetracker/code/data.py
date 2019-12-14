from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
from skimage import io, transform
import torch
from torchvision.transforms import Compose, Resize, ToTensor

from dataset import DepthGazeDataset

# class Rescale(object):
#     """Rescale the image in a sample to a given size.

#     Args:
#     output_size (tuple or int): Desired output size. If tuple, output is
#     matched to output_size. If int, smaller of image edges is matched
#     to output_size keeping aspect ratio the same.
#     """
#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         self.output_size = output_size
#     def __call__(self, sample):
#         images, gaze = sample['images'], sample['gaze']
#         num_images = len(images)
#         h, w = images[0].shape[:2]
#         if isinstance(self.output_size, int):
#             if h > w:
#                 new_h, new_w = self.output_size * h / w, self.output_size
#             else:
#                 new_h, new_w = self.output_size, self.output_size * w / h
#         else:
#             new_h, new_w = self.output_size
#         new_h, new_w = int(new_h), int(new_w)
#         for i in range(num_images):
#             image = images[i]
#             images[i] = transform.resize(image, (new_h, new_w))
#         # h and w are swapped for landmarks because for images,
#         # x and y axes are axis 1 and 0 respectively
#         return {'images': images, 'gaze': gaze}


# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""

#     def __call__(self, sample):
#         images, gaze = sample['images'], sample['gaze']
#         images = np.dstack(image for image in images)
#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C X H X W
#         images = images.transpose((2, 0, 1))
#         return {'images': torch.from_numpy(images),
#                 'gaze': torch.from_numpy(gaze)}


def input_transform(output_size):
    return Compose([
        Resize(output_size),
        ToTensor(),
    ])

def get_dataset(csv_file, eye_cameras, input_size):
    """
    input_size: (36,48) for modified_lenet
                (240,320) for modified_alexnet
    """
    # train_dataset = torch.utils.data.ConcatDataset([
    #     DepthGazeDataset(subject_id, dataset_dir) for subject_id in subject_ids
    #     if subject_id != test_subject_id
    # ])
    dataset = DepthGazeDataset(csv_file, input_transform(tuple(input_size)))
    return dataset
