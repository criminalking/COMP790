#!/bin/bash

# get input: dir name
echo "Please input directory name(user+date), e.g. conny_2019_09_16"
read dirname

prefix=/playpen/connylu/eye_data/$dirname
mkdir $prefix

# loop: get {video*.h264, timestamp*.pts, start_time*.txt} from every ip
declare -i i=1
for HOST in $(cat pi_ips.txt | awk '{ print $1 }') ;
do
    scp -r pi@$HOST:~/synchronization/video$i $prefix
    i=$(( $i + 1 ))
done

# loop: create new dir under video*
for i in {1..8}
do
    mkdir $prefix/video$i/png
done

# ffmpeg images in dir
for i in {1..8}
do
    ffmpeg -i $prefix/video$i/video$i.h264 -qscale:v 2 $prefix/video$i/png/image_%05d.png
done

