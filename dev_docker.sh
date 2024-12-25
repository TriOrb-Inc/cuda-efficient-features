#!bin/bash

docker run -it --rm --name dev_cef --runtime=nvidia --gpus all \
               -v $(pwd):/ws \
               -w /ws \
               triorb/ros2/cuda:humble /bin/bash