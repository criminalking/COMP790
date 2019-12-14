#!/bin/bash

RED='\033[0;31m'
NC='\033[0m' # No Color

# get input: dir name
echo "Please input directory name(user+date), e.g. conny_2019_09_16"
read dirname

prefix=/playpen/connylu/eye_data/$dirname

# loop: create new dir under video*
for i in {1..8}
do
    if [ -d $prefix/video$i ]; then
	mkdir $prefix/video$i/png
	ffmpeg -i $prefix/video$i/video$i.h264 -qscale:v 2 $prefix/video$i/png/image_%05d.png
    else
	echo -e "${RED}There is no video$i in the directory.${NC}"
    fi
done
