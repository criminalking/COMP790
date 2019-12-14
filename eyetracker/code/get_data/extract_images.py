# This file is used to extract images

import numpy as np
import argparse
import os
import datetime
import pandas as pd
import shutil
import cv2
import random


def write_table(intervals, num_spot, num_dot, file):
    file.write('|-------+--------+--------|\n')
    file.write('|       |')
    for i in range(num_spot):
        file.write(' spot_%d |'%(i+1))
    file.write('\n')
    file.write('|-------+--------+--------|\n')
    for j in range(num_dot):
        file.write('| dot_%d |'%(j+1))
        for i in range(num_spot):
            file.write(' {:2d} |'.format(intervals[i][j]))
        file.write('\n')
    file.write('\n')


def process_pi_time(dirname, index):
    pi_time = []
    # read start_time, timestamps
    start_time_filename = os.path.join(dirname, "start_time%d.txt"%index) 
    time_stamp_filename = os.path.join(dirname, "timestamp%d.pts"%index)
    start_time_file = open(start_time_filename, 'r')
    start_time = start_time_file.read() # e.g. 04:11:45:001040
    start_time = datetime.datetime.strptime(start_time, '%I:%M:%S:%f')
    with open(time_stamp_filename, 'r') as time_stamp_file:
        next(time_stamp_file) # skip the first line
        for line in time_stamp_file:
            delta_time = float(line.strip()) * 1000 # miliseconds to microseconds
            time = start_time + datetime.timedelta(microseconds=delta_time)
            pi_time.append(time.strftime('%I:%M:%S:%f'))
    return pi_time


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
                ############ upside down cameras ############
                if i == 7 or i == 8: 
                    red_image = cv2.flip(red_image, 0)
                #############################################
                cv2.imwrite(os.path.join(gray_dir, f), red_image)


def compute_GT(num_spot, num_dot, root_path, infos):
    """ Preprocess csv files. Compute ground truth data (x,y,z), size: (num_spot,num_dot,3). """
    # TODO: numbers below are not super accurate, need to think about the influence
    # display settings
    """
                                                   -----------
                             -----------------     -         -
                             -               -     -         -
                  -----      -               -     -         -
           -----  -   -      -               -     -         -
    eye    -   -  -----      -               -     -         -
           -----             -----------------     -         -
                                                   -         -
                                                   -----------
             0      1                2                  3
            33cm  50.5cm           119cm              280cm      (depth is inaccurate)
    W x H (w x h in pixel):
    0: 20 x 15 cm (2048 x 1536)
    1: 20 x 15 cm (2048 x 1536)
    2: 70.8 x 39.8 cm (2560 x 1440)
    3: 86.5 x 153.8 cm (1080 x 1920)
    """
    height_px = [1536, 1536, 1440, 1920]
    width_px = [2048, 2048, 2560, 1080]
    width_cm = [20, 20, 70.8, 86.5]
    height_cm = [x*1.0*y/z for x,y,z in zip(width_cm, height_px, width_px)]
    cm_to_pix_scale = [x*1.0/y for x,y in zip(width_px, width_cm)]

    # spot settings of reflected display
    spot_depth = [33, 54.5, 119, 280] # cm
    offset_x = [-0.35, -0.35, -0.35, -0.35] # cm, offset of eye center and reflected display center in the x direction
    offset_y = [1.075, 1.4, 0, -2.9] # cm, offset of eye center and reflected display center in the y direction

    # bench type settings
    # Note: Origin is center of eyes
    """
    xxxxxxxxxxxxxxxxxxxxx      O is the origin
    xx     xx   xx     xx
    xx eye xx 0 xx eye xx
    xxxxxxxxx   xxxxxxxxx
    """
    gt = np.zeros((num_spot, num_dot, 3))
    for i in range(num_spot):
        screen_id = infos.iloc[i*num_dot][4] # 0-3
        for j in range(num_dot):
            x = infos.iloc[i*num_dot+j][0] / cm_to_pix_scale[screen_id]
            y = infos.iloc[i*num_dot+j][1] / cm_to_pix_scale[screen_id]
            ############ upside down cameras ############
            if screen_id < 3: # should reflect 0, 1, 2: 
                y = -y
            #############################################
            gt[i][j][0] = x + offset_x[screen_id]
            gt[i][j][1] = y + offset_y[screen_id]
            gt[i][j][2] = spot_depth[screen_id]
    return gt


def write_csv(filename, gt, root, dirname, num_spot, num_dot, cams, intervals):
    """ Write final ground truth data to .csv file. """
    """ Format: (unit is cm)
    Name       Spot  Image1           ...  ImageN           x     y    z
    conny_4_3  1     image_00501.png  ...  image_00502.png  5.46  0.0  50.0
    Note: not including IPD here!!!
    """
    df = pd.DataFrame(columns=['name','spot','x','y','z'])

    num_cam = len(cams)
    for i in range(num_spot):
        image_files = []
        for j in cams:
            path = os.path.join(root, dirname, 'video%d'%j, 'spot%d'%(i+1))
            image_files.append(sorted(os.listdir(path)))
        num_images = len(image_files[0]) # every cam has the same number of images
        x = np.zeros(num_images)
        y = np.zeros(num_images)
        z = np.zeros(num_images)
        last_interval = 0
        for j in range(num_dot):
            this_interval = last_interval + intervals[i][j]
            x[last_interval:this_interval], y[last_interval:this_interval], z[last_interval:this_interval] = gt[i][j]
            last_interval = this_interval

        image_dict = {}
        for j in range(num_cam):
            image_dict['video%d'%cams[j]] = image_files[j]
        df_spot = pd.DataFrame({"name":[dirname]*num_images, 
                                "spot":[i+1]*num_images,
                                **image_dict, 
                                "x": x,
                                "y": y,
                                "z": z}) 
        df = df.append(df_spot, ignore_index = True)
    df.to_csv(filename, index=False)
    return df


def write_org(root_path, org_path, intervals, df, num_spot, num_dot, cams):
    filename = os.path.join(org_path, 'README.org')
    # select image and put in images/
    num_images = len(df)
    for example in range(2):
        index = random.randint(0, num_images-1)
        spot = df.iloc[index][1]
        for i in cams:
            shutil.copy(os.path.join(root_path, 'video%d'%i, 'spot%d'%spot, df.iloc[index]['video%d'%i]),
                        os.path.join(org_path, 'images/example%d_video%d.png'%(example+1, i)))

    # write front
    file = open(filename, 'w')
    file.write('#+TITLE: Visualization\n')
    file.write('#+AUTHOR: criminalking\n')
    file.write('\n')

    file.write('* Image\n')
    file.write('\n')
    file.write('** Example 1\n')
    file.write('\n')
    # write example 1
    file.write('#+BEGIN_HTML\n')
    file.write('<p float="left">\n')
    file.write('<img src="%s" height="240" width="320">\n' % 'images/example1_video2.png')
    file.write('<img src="%s" height="240" width="320">\n' % 'images/example1_video4.png')
    file.write('<img src="%s" height="240" width="320">\n' % 'images/example1_video1.png')
    file.write('<img src="%s" height="240" width="320">\n' % 'images/example1_video6.png')
    file.write('<img src="%s" height="240" width="320">\n' % 'images/example1_video7.png')
    file.write('<img src="%s" height="240" width="320">\n' % 'images/example1_video8.png')
    file.write('<img src="%s" height="240" width="320">\n' % 'images/example1_video3.png')    
    file.write('<img src="%s" height="240" width="320">\n' % 'images/example1_video5.png')
    file.write('</p>\n')
    file.write('#+END_HTML\n')
    file.write('\n')
    
    file.write('** Example 2\n')
    file.write('\n')
    # write example 2
    file.write('#+BEGIN_HTML\n')
    file.write('<p float="left">\n')
    file.write('<img src="%s" height="240" width="320">\n' % 'images/example2_video2.png')
    file.write('<img src="%s" height="240" width="320">\n' % 'images/example2_video4.png')
    file.write('<img src="%s" height="240" width="320">\n' % 'images/example2_video1.png')
    file.write('<img src="%s" height="240" width="320">\n' % 'images/example2_video6.png')
    file.write('<img src="%s" height="240" width="320">\n' % 'images/example2_video7.png')
    file.write('<img src="%s" height="240" width="320">\n' % 'images/example2_video8.png')
    file.write('<img src="%s" height="240" width="320">\n' % 'images/example2_video3.png')    
    file.write('<img src="%s" height="240" width="320">\n' % 'images/example2_video5.png')
    file.write('</p>\n')
    file.write('#+END_HTML\n')
    file.write('\n')

    file.write('* Intervals\n')
    file.write('\n')
    # write table
    write_table(intervals, num_spot, num_dot, file)

    
def main(args):
    cams = [int(x) for x in args.cams]
    num_cam = len(cams)
    num_spot = args.num_spot
    num_dot = args.num_dot
    dirname = os.path.join(args.root, args.dirname)
    csv_file = [os.path.join(dirname, x) for x in os.listdir(dirname) if x.endswith('.csv')][0] # only one csv file in the new version

    # process all pi time
    print("===> Processing all pi time")
    pis_time = []
    for i in cams:
        pi_time = process_pi_time(os.path.join(dirname, 'video%d' % i), i)
        pis_time.append(pi_time)

    #all_indices = np.zeros((num_spot, num_dot, num_cam)).astype(int)
    intervals = np.zeros((num_spot, num_dot)).astype(int)
    # read csv: (x, y, start, end, screen_id)
    infos = pd.read_csv(csv_file)
    assert len(infos) == (num_spot * num_dot) # check if the dots number are correct
    spot_index = [infos.iloc[0][-1], infos.iloc[50][-1], infos.iloc[100][-1], infos.iloc[150][-1]]

    for i in range(num_spot):
        pointers = np.zeros(num_cam).astype(int)
        # start search corresponding images
        print("===> Searching image indices")
        start_indices = np.zeros((num_dot, num_cam)).astype(int)
        end_indices = np.zeros((num_dot, num_cam)).astype(int)
        for j in range(num_dot):
            start_time = infos.iloc[i*num_dot+j][2]
            end_time = infos.iloc[i*num_dot+j][3]
            start_time = datetime.datetime.strptime(start_time, '%I:%M:%S:%f')
            end_time = datetime.datetime.strptime(end_time, '%I:%M:%S:%f')
            for k in range(num_cam):
                pi_time = pis_time[k] # pi_time for each camera
                while datetime.datetime.strptime(pi_time[pointers[k]], '%I:%M:%S:%f') < start_time:
                    pointers[k] += 1
                    continue
                start_index = pointers[k]
                while datetime.datetime.strptime(pi_time[pointers[k]], '%I:%M:%S:%f') <= end_time:
                    pointers[k] += 1
                    continue
                end_index = pointers[k] # should not be included
                start_indices[j][k] = start_index 
                end_indices[j][k] = end_index 
        # synchronize images from all cameras, start indices -> max one, end indices -> min one
        interval = np.min(end_indices - start_indices, 1) 
        end_indices = start_indices + np.tile(interval, (num_cam, 1)).transpose() 
        intervals[i] = interval

        # save images to a new dir, 7*50*15 = 5250 for each camera
        print("===> Copying Data Now")
        for j in range(num_cam):
            new_dir = os.path.join(dirname, 'video%d'%(cams[j]), 'rgb_spot%d'%(spot_index[i]+1))
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            for k in range(num_dot):
                start_index, end_index = start_indices[k][j], end_indices[k][j]
                for index in range(start_index, end_index):
                    shutil.copy(os.path.join(dirname, 'video%d'%(cams[j]), 'png', args.image_format%(index+1)), new_dir)

                    
    root_path = os.path.join(args.root, args.dirname)
    print("===> Starting preprocess images")
    preprocess_image(cams, root_path)
    print("===> Computing ground truth")
    gt = compute_GT(num_spot, num_dot, root_path, infos) # (num_spot, num_dot, 3)
    filename = os.path.join(args.root, '%s.csv'%args.dirname)
    print("===> Writing final .csv file")
    df = write_csv(filename, gt, args.root, args.dirname, num_spot, num_dot, cams, intervals)
    print("===> Writing .org file for visualization")
    org_path = os.path.join(args.org, args.dirname)
    if not os.path.exists(org_path):
        os.makedirs(org_path)
        os.makedirs(os.path.join(org_path, 'images'))
    write_org(root_path, org_path, intervals, df, num_spot, num_dot, cams)
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Extract images.')
    parser.add_argument('--root', type=str, default='/playpen/connylu/eye_data',
                        help='root path')
    parser.add_argument('-d', '--dirname', type=str, required=True,
                        help='directory name, e.g. conny_4_2')
    parser.add_argument('-org', type=str, default='/playpen/connylu/result_eye_images',
                        help='path of org file')
#    parser.add_argument('-ipd', type=float, required=True,
#                        help='IPD of user')
#    parser.add_argument('--num_cam', type=int, default = 8,
#                        help='number of cameras')
    parser.add_argument('-c', '--cams', nargs='+', required=True,
                        help='camera index(1-based)')
    parser.add_argument('--num_spot', type=int, default = 4,
                        help='number of spots')
    parser.add_argument('--num_dot', type=int, default = 50,
                        help='number of dots')
    parser.add_argument('--image_format', type=str, default = 'image_%05d.png',
                        help='image format, e.g, image_%05d.png')
    args = parser.parse_args()
    main(args)
