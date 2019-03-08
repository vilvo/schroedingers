# Schroedingers

## Overview

## cpp - streamer

* librealsense to read framesets from a RealSense camera.
* Frameset filtering, and simple color frame processing with opencv.
* Streaming processed frames to neural network process (py) with protocol buffers.

### Dependencies and build instructions

On Ubuntu 18.04

#### Realsense dependencies
```
sudo apt-key adv --keyserver keys.gnupg.net --recv-key C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key C8B3A55A6F3EFCDE
sudo add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo bionic main" -u
sudo apt install librealsense2-dkms librealsense2-utils librealsense2-dev
```

#### OpenCV dependencies
```
sudo apt install libopencv*
```

#### Build
```
make
```