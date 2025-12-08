#!bin/bash

docker run -it --rm --name dev_cef --runtime=nvidia --gpus all \
               -v $(pwd):/ws \
               -w /ws \
               triorb/l4t-ros2:slam.35.4.3 /bin/bash