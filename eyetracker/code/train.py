from __future__ import print_function
import argparse
from math import log10
import os
from collections import OrderedDict
import numpy as np
import time
import json
from shutil import copyfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import importlib # import all models in models/

from logger import Logger
from data import get_dataset
from utils import get_instance

# args
parser = argparse.ArgumentParser(description='PyTorch Eye Tracking')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('-c', '--config', type=str, default='config_train.json', help='config file for training')
args = parser.parse_args()

# load config file and set some variables
with open(args.config) as handle:
    config = json.load(handle)
assert os.path.exists(config['data_loader']['train_dataset'])
base_name = config['name']
checkpoint_dir = os.path.join(config['trainer']['checkpoint_dir'], base_name)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
log_dir = os.path.join(config['trainer']['log_dir'], base_name)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
print('===> Copying .json file')
copyfile(args.config, os.path.join(log_dir, args.config))

if args.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
device = torch.device("cuda" if args.cuda else "cpu")

logger = Logger(log_dir)

torch.manual_seed(config['seed'])

print('===> Loading datasets')
train_dataset = get_dataset(config['data_loader']['train_dataset'], config['data_loader']['use_cameras'], config['data_loader']['input_size'])
if config['data_loader']['test_dataset']:
    test_dataset = get_dataset(config['data_loader']['test_dataset'], config['data_loader']['use_cameras'], config['data_loader']['input_size'])

# Creating data indices for training and validation splits:
if not config['data_loader']['test_dataset']:
    train_dataset_size = len(train_dataset)
    indices = list(range(train_dataset_size))
    split = int(np.floor(config['data_loader']['test_split'] * train_dataset_size))
    np.random.seed(config['seed'])
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    # TODO: need to make this part more clean with get_instance
    training_data_loader = DataLoader(dataset=train_dataset, num_workers=config['data_loader']['num_workers'], batch_size=config['data_loader']['batch_size'], sampler=train_sampler)
    testing_data_loader = DataLoader(dataset=train_dataset, num_workers=config['data_loader']['num_workers'], batch_size=config['data_loader']['test_batch_size'], sampler=test_sampler)
else:
    training_data_loader = DataLoader(dataset=train_dataset, num_workers=config['data_loader']['num_workers'], batch_size=config['data_loader']['batch_size'], shuffle=True)
    testing_data_loader = DataLoader(dataset=test_dataset, num_workers=config['data_loader']['num_workers'], batch_size=config['data_loader']['test_batch_size'])


print('===> Building model')
module = importlib.import_module('models.{}'.format(config['arch']['type']))
model = module.Model(len(config['data_loader']['use_cameras'])//2).to(device)
if config["trainer"]['restore']:
    print('===> Restoring model')
    model_path = os.path.join(config["trainer"]['checkpoint_dir'], config["trainer"]['restore'])
    model.load_state_dict(torch.load(model_path)['state_dict'])
criterion = get_instance(nn, "loss", config)
optimizer = get_instance(optim, "optimizer", config, model.parameters())
scheduler = get_instance(optim.lr_scheduler, "lr_scheduler", config, optimizer)

def train(epoch, last_loss):
    epoch_loss = 0
    model.train()
    
    end_total = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        start = time.time()
        end_data = start - end_total

        input, target = batch['images'].to(device, dtype=torch.float), batch['gaze'].to(device, dtype=torch.float) # input are left + right images
        
        optimizer.zero_grad()
        loss = criterion(model(input), target)

        # check abnormal value, skip start value
        if epoch > 15 and loss > last_loss * 5:
            print("Find a abnormal dot here...")
            with open(os.path.join(log_dir, 'abnormal_dot.txt'), 'a') as f:
                f.write("epoch:%d, iter:%d, loss:%f\n" % (epoch, iteration, loss))
                for x,y,z in zip(*batch['name']):
                    f.write("%s" % os.path.join(x, y, z))
                f.write("\n")

        last_loss = loss
        
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        # save log file
        info = {
            'loss': loss.item(),
        }
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch)

        end_total = time.time()
        print("===> Epoch[{}]({}/{}): Loss: {:.4f} Data time: {:.4f} Total time: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item(), end_data, end_total-start))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    return last_loss


def validate():
    model.eval()
    
    avg_error = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch['images'].to(device, dtype=torch.float), batch['gaze'].to(device, dtype=torch.float) # input are left + right images

            prediction = model(input)
            mse = criterion(prediction, target)
            avg_error += mse.item()        
    print("===> Error: {:.4f}".format(avg_error / len(testing_data_loader)))

    info = {
        'valid_loss': avg_error / len(testing_data_loader),
    }
    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch)

    return mse


def checkpoint(epoch, error):
    state = OrderedDict([
        ('state_dict', model.state_dict()),
        ('optimizer', optimizer.state_dict()),
        ('epoch', epoch),
        ('error', error),
    ])
    model_out_path = os.path.join(checkpoint_dir, "model_epoch_{}.pth".format(epoch))
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

last_loss = 0
for epoch in range(1, config['trainer']['epochs'] + 1):
    scheduler.step()

    last_loss = train(epoch, last_loss)
    error = validate()
    if epoch % config['trainer']['save_period'] == 0:
        checkpoint(epoch, error)
