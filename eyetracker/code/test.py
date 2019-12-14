import argparse
import json
import os
import numpy as np
import importlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import get_dataset
from utils import visualize_2d, get_instance, compute_angle_error, draw_error

# args
parser = argparse.ArgumentParser(description='PyTorch Eye Tracking Test')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('-c', '--config', type=str, default='train_config.json', help='config file for training')
args = parser.parse_args()

# load config file and set some variables
with open(args.config) as handle:
    config = json.load(handle)
base_name = config['name']
if args.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
device = torch.device("cuda" if args.cuda else "cpu")

# load dataset
dataset = get_dataset(config['data_loader']['dataset'], config['data_loader']['use_cameras'], config['data_loader']['input_size'])
testing_data_loader = DataLoader(dataset=dataset, num_workers=config['data_loader']['num_workers'], batch_size=config['data_loader']['test_batch_size'], shuffle=False)

# restore checkpoint
print('===> Restoring model')
module = importlib.import_module('models.{}'.format(config['arch']['type']))
model = module.Model(len(config['data_loader']['use_cameras'])//2).to(device)
model_path = os.path.join(config["trainer"]['checkpoint_dir'], config["trainer"]['restore'])
model.load_state_dict(torch.load(model_path)['state_dict'])
criterion = get_instance(nn, "loss", config)

def validate():
    model.eval()

    targets = np.zeros((len(dataset), 3))
    predictions = np.zeros((len(dataset), 3))

    test_batch_size = config['data_loader']['test_batch_size']
    
    avg_error = 0
    with torch.no_grad():
        for i, batch in enumerate(testing_data_loader):
            input, target = batch['images'].to(device, dtype=torch.float), batch['gaze'].to(device, dtype=torch.float) # input are 6 images, left 3 + right 3

            prediction = model(input)
            mse = criterion(prediction, target)

            predictions[i*test_batch_size:(i+1)*test_batch_size, :] = prediction.cpu().numpy()
            targets[i*test_batch_size:(i+1)*test_batch_size, :] = target.cpu().numpy()
            
            avg_error += mse.item()
        
    print("Distance error: {:.4f}".format(avg_error / len(testing_data_loader)))
    return mse, predictions, targets

# run test and visualize
error, predictions, targets = validate()
angle_errors = np.zeros((len(dataset), 2)) # horizontal and vertical error
for i,(predict, target) in enumerate(zip(predictions, targets)):
    angle_errors[i] = compute_angle_error(predict, target)
print('Angle error: ', np.mean(angle_errors))
print('Horizontal angle error: ', np.mean(angle_errors[:,0]))
print('Vertical angle error: ', np.mean(angle_errors[:,1]))
num_spot = 4
visualize_2d(predictions, targets, num_spot)
#visualize_3d(predictions, targets)
#draw_error(targets, np.mean(angle_errors, 1))

# def checkpoint(epoch, error):
#     state = OrderedDict([
#         ('state_dict', model.state_dict()),
#         ('optimizer', optimizer.state_dict()),
#         ('epoch', epoch),
#         ('error', error),
#     ])
#     model_out_path = os.path.join(checkpoint_dir, "model_epoch_{}.pth".format(epoch))
#     torch.save(state, model_out_path)



