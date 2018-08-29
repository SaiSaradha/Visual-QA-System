#!/bin/sh
#change permission
chmod +x image_data/unzip_features.sh
chmod +x raw_images/unzip_imgs.sh
chmod +x text_data/unzip_all.sh
chmod +x pretrained/unzip_wordembed.sh

#download data and features
cd image_data/
./unzip_features.sh
cd ../raw_images/
./unzip_imgs.sh
cd ../text_data/
./unzip_all.sh
cd ../pretrained/
./unzip_wordembed.sh

#invoke the main program
cd ../code
py -3 main.py

echo -n "Press Enter to exit"
read keyval