#!/bin/sh
for file in /home/cvos/PycharmProjects/tf2-yolov3-nano/coco/train/*.jpg
do
  echo $file;
  echo $file >> train.txt;
done
