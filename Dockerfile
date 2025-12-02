# syntax=docker/dockerfile:1.0.0-experimental
###########################################
# Base image 
###########################################
# Tag for non jetson devices
# FROM nvidia/cuda:11.7.1-devel-ubuntu22.04 AS base
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0 AS base

ENV DEBIAN_FRONTEND=noninteractive

# Install language
RUN apt-get update && apt-get install -y \
  locales \
  && locale-gen en_US.UTF-8 \
  && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
  && rm -rf /var/lib/apt/lists/*
ENV LANG en_US.UTF-8

# Install timezone
RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime \
  && export DEBIAN_FRONTEND=noninteractive \
  && apt-get update \
  && apt-get install -y tzdata \
  && dpkg-reconfigure --frontend noninteractive tzdata \
  && rm -rf /var/lib/apt/lists/*

# Install ROS2
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    sudo \
  && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
  && apt-get update && apt-get install -y \
    ros-humble-ros-base \
    python3-argcomplete \
  && rm -rf /var/lib/apt/lists/*

ENV ROS_DISTRO=humble \
  AMENT_PREFIX_PATH=/opt/ros/humble \
  COLCON_PREFIX_PATH=/opt/ros/humble \
  LD_LIBRARY_PATH=/opt/ros/humble/lib \
  PATH=/opt/ros/humble/bin:$PATH \
  PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages \
  ROS_PYTHON_VERSION=3 \
  ROS_VERSION=2 
ENV DEBIAN_FRONTEND=

###########################################
#  Develop image 
###########################################
FROM base AS dev

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
  bash-completion \
  build-essential \
  cmake \
  gdb \
  git \
  python3-argcomplete \
  python3-colcon-common-extensions \
  python3-pip \
  python3-rosdep \
  python3-vcstool \
  vim \
  wget \
  python3-autopep8 \
  && rm -rf /var/lib/apt/lists/* \
  && rosdep init || echo "rosdep already initialized"

###########################################
#  Full image 
###########################################
FROM dev AS full

ENV DEBIAN_FRONTEND=noninteractive
# Install the full release
RUN apt-get update && apt-get install -y \
  ros-humble-desktop \
  ros-humble-ament-lint \
  ros-humble-launch-testing \
  ros-humble-launch-testing-ament-cmake \
  ros-humble-launch-testing-ros \
  ros-humble-rmw-cyclonedds-cpp \
  ros-humble-ament-clang-format \
  ros-humble-librealsense2* -y \
  ros-humble-realsense2-* -y \
  ros-humble-rosbag2-storage-mcap \
  && rm -rf /var/lib/apt/lists/* \
  && pip install colcon-mixin \
  && colcon mixin add default https://raw.githubusercontent.com/colcon/colcon-mixin-repository/master/index.yaml \
  && colcon mixin update default
ENV DEBIAN_FRONTEND=

################
# Preliminary Machine Learning Dependencies
################
RUN apt-get update \
 && apt-get install -y -qq --no-install-recommends \
  libglvnd0 \
  libgl1 \
  libglx0 \
  libegl1 \
  libxext6 \
  libx11-6 \
  python3-vcstool \
  git \
  software-properties-common \
  libceres-dev \
  nlohmann-json3-dev

# DL/Nvidia dependencies
RUN apt-get install build-essential g++ gcc -y
RUN apt-get install libgl1-mesa-glx libglib2.0-0 -y
RUN apt-get install openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev -y

# Pytorch
RUN apt-get install libopenblas-base libopenmpi-dev libomp-dev -y 
RUN pip3 install torch==2.8.0 torchvision==0.23.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

# Darknet Yolo
RUN apt-get update && apt-get install -y \
  wget \
  && mkdir /opt/cmake \
  && wget -O /opt/cmake/cmake-3.30.0-linux-aarch64.tar.gz https://github.com/Kitware/CMake/releases/download/v3.30.0/cmake-3.30.0-linux-aarch64.tar.gz \
  && tar -xzf /opt/cmake/cmake-3.30.0-linux-aarch64.tar.gz -C /opt/cmake/
ENV PATH="/opt/cmake/cmake-3.30.0-linux-aarch64/bin:${PATH}"

# This will fail at the make package step if cuda is not found at build time
# comment out if needed
RUN apt-get update && sudo apt-get install -y \
  build-essential \
  libopencv-dev \
  git \
  && mkdir /opt/darknet \
  && git clone --branch v5.1 https://codeberg.org/CCodeRun/darknet.git /opt/darknet/ \
  && mkdir /opt/darknet/build \
  && cd /opt/darknet/build \
  && cmake -DCMAKE_BUILD_TYPE=Release .. \
  && make -j $(nrpoc) \
  && make package \
  && dpkg -i darknet*.deb

RUN apt-get update && sudo apt-get install -y \
  build-essential \
  libtclap-dev \
  libmagic-dev \
  libopencv-dev \
  git \
  && mkdir /opt/DarkHelp \
  && git clone https://codeberg.org/CCodeRun/DarkHelp.git /opt/DarkHelp/ \
  && mkdir /opt/DarkHelp/build \
  && cd /opt/DarkHelp/build \
  && cmake -DCMAKE_BUILD_TYPE=Release .. \
  && make -j $(nproc) \
  && make package \
  && dpkg -i darkhelp*.deb

# Megapose (Note: might want to create a fork of this because currently megapose is unmaintained)
RUN apt-get update && sudo apt-get install -y \
  git \
  && mkdir /opt/megapose6d \
  && git clone https://github.com/megapose6d/megapose6d.git /opt/megapose6d/

# Megapose env vars needed
ENV PYTHONPATH /opt/megapose6d/src/:$PYTHONPATH
ENV CONDA_PREFIX $(python3 -c "import sys; print(sys.prefix)")

# Megapose dependencies
RUN pip install bokeh
RUN pip install joblib
RUN pip install pin
RUN pip install transforms3d
RUN pip install webdataset
RUN pip install omegaconf
RUN pip install tqdm
RUN pip install imageio
RUN pip install pypng
RUN pip install trimesh
RUN pip install panda3d
RUN pip install simplejson
RUN pip install open3d
RUN pip install roma
RUN pip install ipython
RUN pip install selenium


RUN pip install pandas
RUN pip install seaborn

RUN pip install albumentations
RUN pip install pycocotools

# Install packages to make ros_build work
RUN pip install empy catkin-pkg lark

RUN apt-get install -y libgtk2.0-dev pkg-config
RUN pip install opencv-python --no-cache-dir --force-reinstall

# Hacky numpy version fix
RUN pip uninstall numpy -y
RUN pip install numpy==1.26

# Install python-lsp-server for completion
RUN pip install python-lsp-server

# Env vars for the nvidia-container-runtime.
ENV QT_X11_NO_MITSHM 1
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# Add line into bashrc so that it automatically source ros2 setup
RUN echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc
