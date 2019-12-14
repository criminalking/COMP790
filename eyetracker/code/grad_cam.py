import torch
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import cv2
import sys
import numpy as np
import pandas as pd
import argparse
import json
import importlib

from data import get_dataset
from utils import get_instance


class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, target_layers, input_size):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.features1, target_layers)
        self.input_size = input_size

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        x1 = x[:,:self.input_size,:,:]
        x2 = x[:,self.input_size:,:,:]
        target_activations1, output1  = self.feature_extractor(x1)
        output2  = self.model.features2(x2)
        
        import pdb
        pdb.set_trace()
        
        output1 = output1.view(output1.size(0), -1)
        output2 = output2.view(output2.size(0), -1)
        output = torch.cat([output1, output2], dim=1)
        output = self.model.classifier(output)
        return target_activations1, output

def show_cam_on_image(img, mask, cam_id=1):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)

    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    
    cv2.imwrite("heatcam%d.jpg"%cam_id, np.uint8(255 * cam))
    cv2.imwrite("heatmap.jpg", np.uint8(heatmap))

class GradCam:
    def __init__(self, model, target_layer_names, use_cuda, input_size):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names, input_size)

    def forward(self, input):
        return self.model(input) 

    def __call__(self, input, index = None):
        if self.cuda:
            features1, features2, output = self.extractor(input.cuda())
        else:
            features1, features2, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot)
        one_hot.requires_grad = True
        
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features1[-1] # only left
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis = (2, 3))[0, :]
        cam = np.zeros(target.shape[1 : ], dtype = np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (640, 480))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input), torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1), positive_mask_2)

        return grad_input

class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model.features._modules[idx] = GuidedBackpropReLU()

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index = None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot)
        one_hot.requires_grad = True
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0,:,:,:]

        return output

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='Use NVIDIA GPU acceleration')
    parser.add_argument('-c', '--config', type=str, default='train_config.json', help='config file for training')
    args = parser.parse_args()
    args.use_cuda = args.cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

if __name__ == '__main__':
    args = get_args()

    # Load model
    with open(args.config) as handle:
        config = json.load(handle)
    base_name = config['name']
    device = torch.device("cuda" if args.use_cuda else "cpu")

    # load dataset
    dataset = get_dataset(config['data_loader']['dataset'], config['data_loader']['use_cameras'], config['data_loader']['input_size'])
    testing_data_loader = DataLoader(dataset=dataset, num_workers=config['data_loader']['num_workers'], batch_size=config['data_loader']['test_batch_size'], shuffle=True)
        
    # restore checkpoint
    print('===> Restoring model')
    module = importlib.import_module('models.{}'.format(config['arch']['type']))
    model = module.Model(len(config['data_loader']['use_cameras'])//2).to(device)
    model_path = os.path.join(config["trainer"]['checkpoint_dir'], config["trainer"]['restore'])
    model.load_state_dict(torch.load(model_path)['state_dict'])
    criterion = get_instance(nn, "loss", config)

    import pdb
    pdb.set_trace()

    # Can work with any model, but it assumes that the model has a 
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    grad_cam = GradCam(model = model, target_layer_names = ["2"], use_cuda=args.use_cuda, input_size=len(config['data_loader']['use_cameras'])//2)

    # test images
    model.eval()
    targets = np.zeros((len(dataset), 3))
    predictions = np.zeros((len(dataset), 3))
    
    test_batch_size = config['data_loader']['test_batch_size']
    
    avg_error = 0

    csv_file = config['data_loader']['dataset']
    root_dir = csv_file[:csv_file.rfind('/')]
    csv_data = pd.read_csv(csv_file)
    
    cam_ids = config['data_loader']['use_cameras'] # first camera
    
    for i, batch in enumerate(testing_data_loader):
        input, target = batch['images'].to(device, dtype=torch.float), batch['gaze'].to(device, dtype=torch.float) # input are 6 images, left 3 + right 3
        
        prediction = model(input)
        mse = criterion(prediction, target)
        
#        predictions[i*test_batch_size:(i+1)*test_batch_size, :] = prediction.cpu().numpy()
#        targets[i*test_batch_size:(i+1)*test_batch_size, :] = target.cpu().numpy()
        
        avg_error += mse.item()
        
        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested index.
        target_index = 0 # could be 0:x, 1:y, 2:z
        mask = grad_cam(input, target_index)
        
        img1 = cv2.imread(os.path.join(root_dir, batch['name'][0][0], "video%d"%cam_ids[0], batch['name'][1][0], batch['name'][2][0]))
        img1 = img1[:,:,2][:,:,None] * np.ones(3, dtype=int)[None, None, :]
        img2 = cv2.imread(os.path.join(root_dir, batch['name'][0][0], "video%d"%cam_ids[1], batch['name'][1][0], batch['name'][2][0]))
        img2 = img2[:,:,2][:,:,None] * np.ones(3, dtype=int)[None, None, :]
        img3 = cv2.imread(os.path.join(root_dir, batch['name'][0][0], "video%d"%cam_ids[2], batch['name'][1][0], batch['name'][2][0]))
        img3 = img3[:,:,2][:,:,None] * np.ones(3, dtype=int)[None, None, :]
        show_cam_on_image(img1, mask, cam_id=1)
        show_cam_on_image(img2, mask, cam_id=2)
        show_cam_on_image(img3, mask, cam_id=3)
        
        print("MSE error: ", mse.item())
        
        break

        
    print("Distance error: {:.4f}".format(avg_error / len(testing_data_loader)))
    

#    gb_model = GuidedBackpropReLUModel(model = models.vgg19(pretrained=True), use_cuda=args.use_cuda)
#    gb = gb_model(input, index=target_index)
#    utils.save_image(torch.from_numpy(gb), 'gb.jpg')

#    cam_mask = np.zeros(gb.shape)
#    for i in range(0, gb.shape[0]):
#        cam_mask[i, :, :] = mask

#    cam_gb = np.multiply(cam_mask, gb)
#    utils.save_image(torch.from_numpy(cam_gb), 'cam_gb.jpg')
