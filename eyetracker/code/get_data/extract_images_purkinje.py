"""
This file is used to update GT with .csv file created by Yujie
Format:
ir_name, frame_name, target_position, screen_index, pupil_coord, purkj_coord
We only use target_position and screen_index.
"""
import numpy as np
import argparse
import os
import datetime
import pandas as pd
import shutil
import cv2
import random

def extract_red_channel(image):
    assert image.shape[2] == 3 # check if this is a color image
    new_image = image[:,:,2]
    return new_image


def preprocess_image(cams, root_path):
    """ Preprocess images. Extract red channel, turn image upside down vertically. """
    for i in cams:
        dir_path = os.path.join(root_path, 'video%d'%i)
        dirs = [x for x in os.listdir(dir_path) if x.startswith('rgb_spot')]
        for directory in dirs:
            full_path = os.path.join(dir_path, directory)
            gray_dir = full_path.replace('rgb_', '')
            if not os.path.exists(gray_dir):
                os.makedirs(gray_dir)
            files = os.listdir(full_path)
            for f in files:
                image = cv2.imread(os.path.join(full_path, f))
                red_image = extract_red_channel(image)
                cv2.imwrite(os.path.join(gray_dir, f), red_image)


def compute_GT(infos):
    """ Preprocess csv files. Compute ground truth data (x,y,z), size: (num_images, 3). """
    # TODO: numbers below are not super accurate, need to think about the influence
    # display settings
    """
                                                  
                             -----------------    
                             -               -    
                  -----      -               -    
           -----  -   -      -               -    
    eye    -   -  -----      -               -  
           -----             -----------------  
                                                
                                                
             0      1                2          
            33cm  50.5cm           119cm           (depth is inaccurate)
    W x H (w x h in pixel):
    0: 20 x 15 cm (2048 x 1536)
    1: 20 x 15 cm (2048 x 1536)
    2: 70.8 x 39.8 cm (2560 x 1440)
    3: 20 x 15 cm (2048 x 1536), move 0 5cm further
    3: 20 x 15 cm (2048 x 1536), move 1 5cm further
    """
    height_px = [1536, 1536, 1440, 1920, 1536]
    width_px = [2048, 2048, 2560, 1080, 2048]
    width_cm = [20, 20, 70.8, 20, 20]
    height_cm = [x*1.0*y/z for x,y,z in zip(width_cm, height_px, width_px)]
    cm_to_pix_scale = [x*1.0/y for x,y in zip(width_px, width_cm)]

    # spot settings of reflected display
    spot_depth = [33, 54.5, 119, 38, 59.5] # cm
    offset_x = [-0.35, -0.35, -0.35, -0.35, -0.35] # cm, offset of eye center and reflected display center in the x direction
    offset_y = [1.075, 1.4, 0, 1.075, 1.4] # cm, offset of eye center and reflected display center in the y direction

    # bench type settings
    # Note: Origin is center of right eye
    """
    xxxxxxxxxxxxxxxxxxxxx      O is the origin
    xx     xx   xx     xx
    xx eye xx   xx 0   xx
    xxxxxxxxx   xxxxxxxxx
    """
    new_target_position = np.zeros((len(infos), 3))
    for i in range(len(infos)):
        screen_id = infos.iloc[i][3]
        target_x = float((infos.iloc[i][2]).split(',')[0][1:])
        target_y = float((infos.iloc[i][2]).split(',')[1][:-1])

        x = target_x / cm_to_pix_scale[screen_id]
        y = target_y / cm_to_pix_scale[screen_id]
        ############ upside down cameras ############
        #if screen_id < 3: # should reflect 0, 1, 2: 
        y = -y
        #############################################
        x = x + offset_x[screen_id]
        y = y + offset_y[screen_id]
        z = spot_depth[screen_id]

        new_target_position[i] = np.array([x,y,z])
    return new_target_position


def write_csv(filename, target_position, infos, dirname):
    """ Write final ground truth data to .csv file. """
    """ Format: (unit is cm)
    name         spot    frame              x     y    z      pupil_x  pupil_y  pupil_radius  purkj1_x purkj1_y purkj3_x purkj3_y
    conny_4_3    1       image_00501.png    5.46  0.0  50.0   23       26       23            12       14       166      277
    """
    num_images = len(infos)
    pupils = np.zeros((num_images, 3))
    purkj1 = np.zeros((num_images, 2))
    purkj3 = np.zeros((num_images, 2))
    for i in range(num_images):
        pupils[i, 0] = float(infos.iloc[i,4].split(',')[0][1:])
        pupils[i, 1] = float(infos.iloc[i,4].split(',')[1])
        pupils[i, 2] = float(infos.iloc[i,4].split(',')[2][:-1])
        purkj1[i, 0] = float(infos.iloc[i,5].split(',')[0][2:])
        purkj1[i, 1] = float(infos.iloc[i,5].split(',')[1][:-1])
        purkj3[i, 0] = float(infos.iloc[i,5].split(',')[2][2:])
        try:
            purkj3[i, 1] = float(infos.iloc[i,5].split(',')[3][:-1])
        except:
            purkj3[i, 1] = float(infos.iloc[i,5].split(',')[3][:-2])
    df = pd.DataFrame({"name":[dirname]*num_images, 
                       "spot":infos.iloc[:,3],
                       "frame":infos.iloc[:,1] + '.png',
                       "x":target_position[:,0],
                       "y":target_position[:,1],
                       "z":target_position[:,2],
                       "pupil_x":pupils[:,0],
                       "pupil_y":pupils[:,1],
                       "pupil_radius":pupils[:,2],
                       "purkj1_x":purkj1[:,0],
                       "purkj1_y":purkj1[:,1],
                       "purkj3_x":purkj3[:,0],
                       "purkj3_y":purkj3[:,1]}) 
    df.to_csv(filename, index=False)
    return df

    
def main(args):
    dirname = os.path.join(args.root, args.dirname)
    csv_file = dirname + '.csv'

    infos = pd.read_csv(csv_file)
    print("===> Computing ground truth")
    gt = compute_GT(infos) # (num_images, 3)
    filename = os.path.join('/data/connylu/eye_data/worldnew.csv')
    print("===> Writing final .csv file")
    df = write_csv(filename, gt, infos, args.dirname)
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Generate ground truth.')
    parser.add_argument('--root', type=str, default='/data/connylu/eye_data',
                        help='root path')
    parser.add_argument('-d', '--dirname', type=str, required=True,
                        help='directory name, world or eye, e.g. world')
#    parser.add_argument('-ipd', type=float, required=True,
#                        help='IPD of user')
#    parser.add_argument('--num_cam', type=int, default = 8,
#                        help='number of cameras')
    args = parser.parse_args()
    main(args)
