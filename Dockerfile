# syntax=docker/dockerfile:1.0.0-experimental
###########################################
# Base image 
###########################################
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04 AS base

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

###########################################
#  Full+Gazebo image 
###########################################
#FROM full AS gazebo
#
#ENV DEBIAN_FRONTEND=noninteractive
## Install gazebo
#RUN apt-get update && apt-get install -y \
#  ros-humble-gazebo* \
#  && rm -rf /var/lib/apt/lists/*
#ENV DEBIAN_FRONTEND=
#
###########################################
#  Full+Gazebo+Nvidia image 
###########################################
#
#FROM gazebo AS gazebo-nvidia
#
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

# Install python libraries for DL
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install Pillow
RUN pip install tqdm
RUN pip install torchpack
RUN pip install numba
RUN pip install opencv-python
RUN pip install pandas

# Jupyter Notebook
RUN pip install notebook ipykernel
RUN pip install matplotlib

# Install packages to make ros_build work
RUN pip install empy catkin-pkg lark

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
